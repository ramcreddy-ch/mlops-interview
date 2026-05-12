# Section 3: Orchestration & Pipelines
## Tools: Airflow, Kubeflow Pipelines | Build Real Production DAGs

---

## 3.1 Why Orchestration Matters in Production

Without orchestration, ML pipelines look like this:
```bash
# What junior teams do (DANGEROUS):
nohup python preprocess.py && python train.py && python deploy.py &
```
Problems: No retry logic. No SLA tracking. No visibility. A single failure at 3 AM silently stops retraining, model gets stale, accuracy degrades.

**With orchestration:** Every step is a node. Failures trigger alerts. Retries happen automatically. You know exactly which stage failed and why.

---

## 3.2 Apache Airflow — Production DAG for Full ML Pipeline

### Architecture Overview:
```
Airflow Scheduler ──▶ DAG Definition ──▶ Task Queue (Redis/RabbitMQ)
                                               │
                                    ┌──────────┴──────────┐
                               Worker 1               Worker 2
                             (ETL tasks)          (Training tasks)
```

### Real Production DAG: Fraud Detection Pipeline

```python
# dags/fraud_training_pipeline.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.amazon.aws.operators.emr import EmrServerlessStartJobRunOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from airflow.utils.trigger_rule import TriggerRule

# ── Default args apply to ALL tasks in the DAG ──────────────────────────────
default_args = {
    "owner": "ml-platform-team",
    "depends_on_past": False,          # Don't wait for yesterday's run
    "email_on_failure": True,
    "email": ["ml-oncall@company.com"],
    "retries": 2,                       # Retry failed tasks twice
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,  # 5min → 10min → 20min
    "execution_timeout": timedelta(hours=2),  # Kill if running > 2h
}

with DAG(
    dag_id="fraud_model_training_pipeline",
    default_args=default_args,
    description="End-to-end fraud model retraining pipeline",
    schedule_interval="0 2 * * *",   # 2 AM daily
    start_date=datetime(2024, 1, 1),
    catchup=False,                    # Don't run missed schedules
    max_active_runs=1,                # Prevent concurrent runs
    tags=["ml", "fraud", "critical"],
    # SLA: This entire DAG must finish within 4 hours
    dagrun_timeout=timedelta(hours=4),
) as dag:

    # ── Task 1: Validate raw data freshness ─────────────────────────────────
    def check_data_freshness(**context):
        import boto3
        from datetime import timezone

        s3 = boto3.client("s3")
        obj = s3.head_object(
            Bucket="fraud-data-lake",
            Key=f"raw/transactions/{context['ds']}/data.parquet"
        )
        age_hours = (
            datetime.now(timezone.utc) - obj["LastModified"]
        ).total_seconds() / 3600

        if age_hours > 6:
            raise ValueError(
                f"Data is {age_hours:.1f}h old! Expected < 6h. "
                f"Upstream pipeline may have failed."
            )

        return {"data_age_hours": age_hours}

    validate_data = PythonOperator(
        task_id="validate_data_freshness",
        python_callable=check_data_freshness,
        sla=timedelta(minutes=10),   # Task-level SLA
    )

    # ── Task 2: Run Great Expectations validation ────────────────────────────
    run_ge_validation = BashOperator(
        task_id="run_data_validation",
        bash_command="""
            cd /opt/airflow/dags/repo
            python -m great_expectations checkpoint run fraud_training_checkpoint \
                --batch-request '{"date": "{{ ds }}"}'
        """,
        sla=timedelta(minutes=20),
    )

    # ── Task 3: Run Spark feature engineering on EMR Serverless ─────────────
    run_feature_engineering = EmrServerlessStartJobRunOperator(
        task_id="run_feature_engineering",
        application_id="{{ var.value.emr_app_id }}",
        execution_role_arn="{{ var.value.emr_role_arn }}",
        job_driver={
            "sparkSubmit": {
                "entryPoint": "s3://fraud-code/src/featurize.py",
                "entryPointArguments": [
                    "--input-date", "{{ ds }}",
                    "--output-path", "s3://fraud-features/{{ ds }}/",
                    "--window-days", "30",
                ],
                "sparkSubmitParameters": (
                    "--conf spark.executor.cores=4 "
                    "--conf spark.executor.memory=16g "
                    "--conf spark.executor.instances=20 "
                    "--conf spark.dynamicAllocation.enabled=true "
                    "--conf spark.dynamicAllocation.maxExecutors=50"
                ),
            }
        },
        sla=timedelta(hours=1),
    )

    # ── Task 4: Train model on SageMaker or K8s job ─────────────────────────
    def trigger_training(**context):
        import boto3

        sm = boto3.client("sagemaker", region_name="us-east-1")
        job_name = f"fraud-training-{context['ds'].replace('-', '')}"

        sm.create_training_job(
            TrainingJobName=job_name,
            AlgorithmSpecification={
                "TrainingImage": "123456789.dkr.ecr.us-east-1.amazonaws.com/fraud-trainer:latest",
                "TrainingInputMode": "FastFile",
            },
            RoleArn="{{ var.value.sagemaker_role }}",
            InputDataConfig=[{
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3Uri": f"s3://fraud-features/{context['ds']}/train/",
                        "S3DataType": "S3Prefix",
                    }
                },
                "ContentType": "application/x-parquet",
            }],
            OutputDataConfig={"S3OutputPath": f"s3://fraud-models/{job_name}/"},
            ResourceConfig={
                "InstanceType": "ml.g4dn.xlarge",   # 1x T4 GPU
                "InstanceCount": 1,
                "VolumeSizeInGB": 100,
            },
            StoppingCondition={"MaxRuntimeInSeconds": 7200},  # Max 2h
            HyperParameters={
                "n-estimators": "300",
                "max-depth": "6",
                "learning-rate": "0.05",
                "scale-pos-weight": "99",
                "training-date": context["ds"],
            },
        )

        # Wait for job to complete (poll every 60s)
        waiter = sm.get_waiter("training_job_completed_or_stopped")
        waiter.wait(
            TrainingJobName=job_name,
            WaiterConfig={"Delay": 60, "MaxAttempts": 120},
        )

        # Get output location
        response = sm.describe_training_job(TrainingJobName=job_name)
        if response["TrainingJobStatus"] != "Completed":
            raise ValueError(f"Training failed: {response['FailureReason']}")

        return {"model_s3_uri": response["ModelArtifacts"]["S3ModelArtifacts"]}

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=trigger_training,
        sla=timedelta(hours=2),
    )

    # ── Task 5: Validate model quality gate ─────────────────────────────────
    def validate_model_quality(**context):
        import mlflow
        import json

        ti = context["ti"]
        model_uri = ti.xcom_pull(task_ids="train_model")["model_s3_uri"]

        # Load evaluation metrics logged by training script
        metrics = json.loads(
            open(f"/tmp/{context['ds']}_metrics.json").read()
        )

        # Hard gates — fail pipeline if model is bad
        thresholds = {
            "auc_pr": 0.75,         # Precision-Recall AUC (imbalanced data)
            "f1_score": 0.72,
            "false_positive_rate": 0.05,  # Can't annoy too many real users
        }

        failures = []
        for metric, threshold in thresholds.items():
            if metric == "false_positive_rate":
                if metrics[metric] > threshold:
                    failures.append(f"{metric}: {metrics[metric]} > {threshold}")
            else:
                if metrics[metric] < threshold:
                    failures.append(f"{metric}: {metrics[metric]} < {threshold}")

        if failures:
            raise ValueError(
                f"Model quality gate FAILED:\n" + "\n".join(failures)
            )

        # Register model in MLflow if it passes
        mlflow.set_tracking_uri("http://mlflow.internal:5000")
        with mlflow.start_run(run_name=f"fraud-{context['ds']}") as run:
            mlflow.log_metrics(metrics)
            mlflow.log_param("training_date", context["ds"])
            mlflow.log_param("model_s3_uri", model_uri)

            # Transition to Staging in registry
            client = mlflow.tracking.MlflowClient()
            mv = client.create_model_version(
                name="fraud-detection-model",
                source=model_uri,
                run_id=run.info.run_id,
            )
            client.transition_model_version_stage(
                name="fraud-detection-model",
                version=mv.version,
                stage="Staging",
            )

    validate_quality = PythonOperator(
        task_id="validate_model_quality",
        python_callable=validate_model_quality,
        sla=timedelta(minutes=15),
    )

    # ── Task 6: Branch — deploy or halt ─────────────────────────────────────
    def decide_deployment(**context):
        ti = context["ti"]
        # Check if validation passed (XCom from previous task)
        # If validate_model_quality raised, this task never runs
        # Additional business logic can go here
        return "deploy_to_staging"   # Go to deployment task

    branch = BranchPythonOperator(
        task_id="decide_deployment_path",
        python_callable=decide_deployment,
    )

    # ── Task 7: Deploy to staging via ArgoCD ────────────────────────────────
    deploy_staging = BashOperator(
        task_id="deploy_to_staging",
        bash_command="""
            # Update Helm values with new model version
            MODEL_VERSION="{{ ti.xcom_pull(task_ids='validate_model_quality') }}"
            
            # Update GitOps repo to trigger ArgoCD
            cd /tmp/gitops-repo
            git pull origin main
            
            # Update the model tag in Helm values
            yq e ".model.version = \"$MODEL_VERSION\"" \
                -i k8s/fraud-model/staging/values.yaml
            
            git add k8s/fraud-model/staging/values.yaml
            git commit -m "deploy: fraud-model to staging - {{ ds }}"
            git push origin main
            
            # Wait for ArgoCD to sync (poll ArgoCD API)
            argocd app wait fraud-model-staging --health --timeout 300
        """,
    )

    # ── Task 8: Run smoke tests against staging ──────────────────────────────
    smoke_test_staging = BashOperator(
        task_id="smoke_test_staging",
        bash_command="""
            python tests/integration/test_fraud_endpoint.py \
                --endpoint http://fraud-model.staging.internal \
                --test-cases tests/data/smoke_test_transactions.json \
                --expected-auc 0.85
        """,
    )

    # ── Task 9: Notify success ───────────────────────────────────────────────
    notify_success = SlackWebhookOperator(
        task_id="notify_success",
        http_conn_id="slack_webhook",
        message=(
            "✅ *Fraud Model Retraining Complete* ({{ ds }})\n"
            "Model deployed to staging. Awaiting prod promotion."
        ),
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # ── Task 10: Notify failure ──────────────────────────────────────────────
    notify_failure = SlackWebhookOperator(
        task_id="notify_failure",
        http_conn_id="slack_webhook",
        message=(
            "🚨 *Fraud Model Pipeline FAILED* ({{ ds }})\n"
            "Check Airflow: http://airflow.internal/dags/fraud_model_training_pipeline"
        ),
        trigger_rule=TriggerRule.ONE_FAILED,  # Runs if ANY upstream task fails
    )

    # ── Wire up dependencies ─────────────────────────────────────────────────
    (
        validate_data
        >> run_ge_validation
        >> run_feature_engineering
        >> train_model
        >> validate_quality
        >> branch
        >> deploy_staging
        >> smoke_test_staging
        >> notify_success
    )

    # Failure notification runs in parallel with all tasks
    [validate_data, run_ge_validation, run_feature_engineering,
     train_model, validate_quality, deploy_staging] >> notify_failure
```

---

## 3.3 Airflow vs Kubeflow Pipelines — Real Decision Framework

| Dimension | Apache Airflow | Kubeflow Pipelines |
|---|---|---|
| **Primary use case** | General ETL + ML orchestration | Pure ML workflows on K8s |
| **Runs on** | Any server, VM, K8s | Must be on Kubernetes |
| **ML-specific features** | Basic (Python operators) | Native (experiment tracking, artifact lineage) |
| **Multi-tenancy** | Complex setup | Native (namespaces) |
| **Scaling** | Celery workers or K8s executor | Native K8s pods per step |
| **UI** | Good for DAG visibility | Good for pipeline lineage |
| **Learning curve** | Medium | High (K8s knowledge required) |
| **Best for** | Teams with mixed ETL + ML | Pure ML platform teams |
| **Cost** | Cheaper to run | More infra overhead |
| **Prod maturity** | Very mature (10+ years) | Mature (5+ years) |

**When to choose Airflow:** Your team does both data engineering and ML. You need to trigger ML pipelines from ETL completion events.

**When to choose Kubeflow:** Your team is purely ML. You need tight K8s integration, GPU resource scheduling, and native experiment tracking.

---

## 3.4 Kubeflow Pipelines — Real Production Example

```python
# fraud_pipeline_kubeflow.py
import kfp
from kfp import dsl
from kfp.components import func_to_container_op

# ── Define each step as a containerized function ─────────────────────────────

@func_to_container_op
def validate_data(date: str) -> str:
    """Runs Great Expectations against raw data"""
    import great_expectations as ge
    import json

    context = ge.get_context()
    result = context.run_checkpoint(
        checkpoint_name="fraud_checkpoint",
        batch_request={"date": date}
    )
    if not result.success:
        raise ValueError(f"Data validation failed for {date}")
    return date


@func_to_container_op
def run_feature_engineering(date: str, output_path: str) -> str:
    """Runs PySpark feature engineering"""
    import subprocess
    result = subprocess.run([
        "spark-submit",
        "--master", "k8s://https://k8s-api.internal:6443",
        "--conf", "spark.executor.instances=20",
        "--conf", "spark.kubernetes.namespace=ml-spark",
        "s3://fraud-code/featurize.py",
        "--date", date,
        "--output", output_path
    ], check=True, capture_output=True)
    return f"{output_path}/{date}/"


@func_to_container_op
def train_model(
    features_path: str,
    model_output_path: str,
    n_estimators: int = 300,
    max_depth: int = 6,
) -> dict:
    """Trains XGBoost model and logs to MLflow"""
    import xgboost as xgb
    import mlflow
    import pandas as pd
    import json

    df = pd.read_parquet(features_path)
    X = df.drop(["is_fraud", "transaction_id"], axis=1)
    y = df["is_fraud"]

    mlflow.set_tracking_uri("http://mlflow.internal:5000")
    with mlflow.start_run():
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,
            scale_pos_weight=99,
            use_label_encoder=False,
            eval_metric="aucpr",
            tree_method="gpu_hist",  # Use GPU if available
        )
        model.fit(X, y)

        # Log and save
        mlflow.xgboost.log_model(model, "model")
        model.save_model(f"{model_output_path}/model.xgb")

        metrics = {"auc_pr": 0.89}  # Simplified
        mlflow.log_metrics(metrics)
        return metrics


@func_to_container_op
def validate_and_register(metrics: dict, model_path: str) -> str:
    """Quality gate + model registry promotion"""
    import mlflow
    import json

    if isinstance(metrics, str):
        metrics = json.loads(metrics)

    if metrics.get("auc_pr", 0) < 0.75:
        raise ValueError(f"Model failed quality gate: {metrics}")

    client = mlflow.tracking.MlflowClient("http://mlflow.internal:5000")
    mv = client.create_model_version(
        name="fraud-detection-model",
        source=model_path,
        description=f"Auto-registered | AUC-PR={metrics['auc_pr']:.4f}"
    )
    client.transition_model_version_stage(
        name="fraud-detection-model",
        version=mv.version,
        stage="Staging"
    )
    return mv.version


# ── Wire the pipeline ─────────────────────────────────────────────────────────
@dsl.pipeline(
    name="Fraud Detection Training Pipeline",
    description="End-to-end fraud model retraining"
)
def fraud_training_pipeline(
    date: str = "2024-11-15",
    features_output_path: str = "s3://fraud-features",
    model_output_path: str = "s3://fraud-models",
):
    # Step 1: Validate data
    validate_step = validate_data(date=date)
    validate_step.set_retry_count(2)
    validate_step.set_timeout(600)

    # Step 2: Feature engineering (depends on validation)
    features_step = run_feature_engineering(
        date=validate_step.output,
        output_path=features_output_path,
    )
    # Request GPU node for feature engineering (if needed)
    features_step.set_gpu_limit(0)   # CPU-only for Spark
    features_step.set_memory_limit("32G")
    features_step.set_cpu_limit("16")

    # Step 3: Train model on GPU
    train_step = train_model(
        features_path=features_step.output,
        model_output_path=model_output_path,
        n_estimators=300,
        max_depth=6,
    )
    train_step.set_gpu_limit(1)       # Request 1 GPU
    train_step.add_node_selector_constraint(
        "cloud.google.com/gke-accelerator", "nvidia-tesla-t4"
    )
    train_step.set_memory_limit("16G")

    # Step 4: Validate and register
    register_step = validate_and_register(
        metrics=train_step.output,
        model_path=model_output_path,
    )

    # Add experiment tracking metadata
    dsl.get_pipeline_conf().add_op_transformer(
        lambda op: op.add_pod_annotation("mlops.company.com/team", "fraud")
    )


# ── Compile and submit ────────────────────────────────────────────────────────
if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=fraud_training_pipeline,
        package_path="fraud_pipeline.yaml"
    )

    client = kfp.Client(host="http://kubeflow.internal")
    run = client.create_run_from_pipeline_func(
        pipeline_func=fraud_training_pipeline,
        arguments={"date": "2024-11-15"},
        experiment_name="Fraud Detection Experiments",
        run_name="fraud-training-2024-11-15",
    )
```

---

## 3.5 Failure Handling, Retries, and SLAs — Production Patterns

### Pattern 1: Idempotent Tasks

Every Airflow task must be **idempotent** — running it twice produces the same result as running it once.

```python
# BAD: Not idempotent — running twice creates duplicate records
def load_features(**context):
    df = compute_features()
    df.to_sql("features", conn, if_exists="append")  # Duplicates on retry!

# GOOD: Idempotent — upsert or delete-then-insert
def load_features(**context):
    df = compute_features()
    date = context["ds"]
    
    # Delete existing data for this date first (partition)
    conn.execute(f"DELETE FROM features WHERE date = '{date}'")
    df.to_sql("features", conn, if_exists="append")  # Safe to retry
```

### Pattern 2: Checkpointing Long Tasks

```python
def train_with_checkpoints(**context):
    import torch
    import os

    checkpoint_path = f"/shared-storage/checkpoints/{context['ds']}"
    start_epoch = 0

    # Resume from checkpoint if it exists (handles mid-task failures)
    if os.path.exists(f"{checkpoint_path}/latest.pt"):
        checkpoint = torch.load(f"{checkpoint_path}/latest.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, 100):
        train_one_epoch(model, loader, optimizer)

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, f"{checkpoint_path}/epoch_{epoch}.pt")
            torch.save(
                {"epoch": epoch, ...},
                f"{checkpoint_path}/latest.pt"   # Always overwrite latest
            )
```

### Pattern 3: SLA Miss Callbacks

```python
def sla_miss_callback(dag, task_list, blocking_task_list, slas, blocking_tis):
    """Called by Airflow when SLA is breached"""
    import requests

    blocked_tasks = [str(t) for t in task_list]
    message = (
        f"🚨 *SLA BREACH* in DAG `{dag.dag_id}`\n"
        f"Tasks: {blocked_tasks}\n"
        f"This pipeline must complete by 6 AM for batch scoring.\n"
        f"Immediate action required."
    )
    requests.post(
        "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
        json={"text": message}
    )
    # Also page on-call engineer
    requests.post("https://api.pagerduty.com/incidents", ...)

with DAG(
    dag_id="fraud_pipeline",
    sla_miss_callback=sla_miss_callback,
    ...
) as dag:
    pass
```

---

*Next → [Section 4: Experiment Tracking](../experiment-tracking/README.md)*
