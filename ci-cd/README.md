# CI/CD for ML Systems — Production Deep Dive
## GitOps, Model Versioning, Continuous Training, Canary Automation

---

## Part 1: Software CI/CD vs ML CI/CD — The Paradigm Shift

In traditional software, CI/CD is a linear path: `Code → Test → Build → Deploy`.
In ML, the process is multidimensional because **Data + Code = Model**.

```
┌──────────────────────────────────────────────────────────────────┐
│                   THE THREE ML TRIGGERS                           │
├───────────────────────┬──────────────────────┬───────────────────┤
│  1. CODE TRIGGER      │  2. DATA TRIGGER     │ 3. MODEL TRIGGER  │
├───────────────────────┼──────────────────────┼───────────────────┤
│ Data Scientist merges │ Monitoring detects   │ Model finishes    │
│ PR with new feature   │ data drift or new    │ training and      │
│ engineering logic.    │ month of data lands. │ passes eval.      │
├───────────────────────┼──────────────────────┼───────────────────┤
│ Triggers: CI tests    │ Triggers: CT         │ Triggers: CD      │
│ (unit, integration)   │ (Continuous Training)│ (Continuous       │
│ then builds Docker    │ to retrain model     │ Deployment)       │
│ image.                │ with existing code.  │ to Kubernetes.    │
└───────────────────────┴──────────────────────┴───────────────────┘
```

**Rule of Production MLOps:** You NEVER train a model inside your CI pipeline (Jenkins/GitHub Actions). Training takes hours/days and requires GPUs. CI just tests the code syntax and builds the environment. Training happens in the CT pipeline (Airflow/Kubeflow).

---

## Part 2: The ML Continuous Integration (CI) Pipeline

**Goal:** Prevent bad code from breaking the training pipeline.

```yaml
# .github/workflows/ml-ci.yml — Production ML CI Pipeline
name: ML Continuous Integration

on:
  pull_request:
    branches: [ main ]

jobs:
  test-and-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          cache: 'poetry'
          
      - name: Install dependencies
        run: poetry install
        
      - name: Lint and Format (Ruff)
        run: poetry run ruff check . && poetry run ruff format --check .

      - name: Type Checking (MyPy)
        run: poetry run mypy src/
        
      - name: Unit Tests (Fast, no data needed)
        run: poetry run pytest tests/unit/
        
      # CRITICAL ML STEP: Test the training loop on dummy data
      # Verifies shapes match, loss computes, no NaN explosions
      - name: Integration Test - Dummy Training
        run: |
          poetry run python src/train.py \
            --data_path=tests/fixtures/dummy_data.parquet \
            --epochs=2 \
            --batch_size=4 \
            --fast_dev_run=True
            
      - name: Security Scan (Trivy)
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'local-build'
          format: 'table'
          exit-code: '1'
          ignore-unfixed: true
          vuln-type: 'os,library'
          severity: 'CRITICAL,HIGH'

      - name: Build and Push Training Docker Image
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: mycr.io/fraud-trainer:${{ github.sha }}
```

---

## Part 3: Continuous Training (CT) — Automated Retraining

**Goal:** Retrain the model when data drifts or new labels arrive, without human intervention.

### Orchestrator: Apache Airflow

```python
# dags/retrain_fraud_model.py
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml-platform',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ['ml-alerts@company.com'],
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'fraud_model_continuous_training',
    default_args=default_args,
    description='Weekly automated retrain of fraud model',
    schedule_interval='0 2 * * 0', # Every Sunday at 2 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    # Step 1: Validate incoming data quality
    validate_data = KubernetesPodOperator(
        task_id='validate_data_expectations',
        name='gx-validate',
        namespace='ml-training',
        image='mycr.io/fraud-trainer:latest',
        cmds=["great_expectations", "checkpoint", "run", "weekly_fraud_data"],
        get_logs=True,
    )

    # Step 2: Run training job on GPU node
    train_model = KubernetesPodOperator(
        task_id='train_xgboost_model',
        name='fraud-train',
        namespace='ml-training',
        image='mycr.io/fraud-trainer:latest',
        cmds=["python", "src/train.py"],
        env_vars={
            'MLFLOW_TRACKING_URI': 'http://mlflow.ml-platform:5000',
            'MLFLOW_EXPERIMENT_NAME': 'fraud_automated_retraining'
        },
        node_selector={'nvidia.com/gpu': 'true'},
        # VPA will manage resources, but we set requests
        container_resources={
            'request_cpu': '4',
            'request_memory': '16Gi',
            'limit_gpu': '1'
        },
        get_logs=True,
    )

    # Step 3: Evaluate model against golden holdout set
    evaluate_model = KubernetesPodOperator(
        task_id='evaluate_and_register',
        name='fraud-eval',
        namespace='ml-training',
        image='mycr.io/fraud-trainer:latest',
        cmds=["python", "src/evaluate.py"],
        env_vars={
            'MINIMUM_AUC': '0.85', # Fail pipeline if worse than this
            'PROMOTE_TO_STAGING': 'true'
        },
        get_logs=True,
    )

    validate_data >> train_model >> evaluate_model
```

---

## Part 4: Continuous Deployment (CD) — GitOps for ML

**Goal:** The state of production Kubernetes is identical to what is in Git.
**Tool:** ArgoCD

```
Workflow:
1. Airflow evaluates new model -> Registers to MLflow as "Staging"
2. Human reviews metrics -> Transitions MLflow state to "Production"
3. Webhook triggers GitHub Action
4. GitHub Action updates the `model_version` tag in the GitOps repo
5. ArgoCD detects Git change -> Syncs new model to K8s cluster
```

### The GitOps Repository (Separate from code repo!)

```yaml
# gitops-repo/fraud-api/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-api
  namespace: ml-serving
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fraud-api
  template:
    metadata:
      labels:
        app: fraud-api
    spec:
      containers:
      - name: api
        # Image tag is the CODE version
        image: mycr.io/fraud-api:v2.4.1
        env:
        # This is what changes when a new model is promoted!
        - name: MODEL_URI
          value: "s3://mlflow-artifacts/1/abcdef12345/artifacts/model"
        - name: MLFLOW_RUN_ID
          value: "abcdef12345"
```

### The Auto-Promotion Script (Triggered by MLflow webhook):

```bash
# update_gitops_repo.sh
NEW_RUN_ID=$1
MODEL_URI="s3://mlflow-artifacts/1/${NEW_RUN_ID}/artifacts/model"

git clone https://github.com/company/gitops-repo.git
cd gitops-repo

# Use yq to update the YAML file safely
yq e -i "(.spec.template.spec.containers[] | select(.name == \"api\").env[] | select(.name == \"MODEL_URI\").value) = \"${MODEL_URI}\"" fraud-api/deployment.yaml
yq e -i "(.spec.template.spec.containers[] | select(.name == \"api\").env[] | select(.name == \"MLFLOW_RUN_ID\").value) = \"${NEW_RUN_ID}\"" fraud-api/deployment.yaml

git add fraud-api/deployment.yaml
git commit -m "Auto-deploy: Promote fraud model ${NEW_RUN_ID} to production"
git push origin main
# ArgoCD takes over from here!
```

**Why GitOps is mandatory for ML:**
If the new model causes a massive spike in false positives, rollback is instant. Just `git revert <commit-hash>` in the GitOps repo. ArgoCD immediately pulls the old model back into production. No pipeline to re-run.

---

## Part 5: Safe Deployments — Canary & Shadow

Never deploy an ML model to 100% of traffic instantly.

### 1. Shadow Deployment (Zero Risk)
- New model receives 100% of traffic.
- Makes predictions, logs them to database.
- Responses are discarded. User only sees V1's response.
- **Why:** To verify the new model doesn't crash on weird edge-case inputs, and to check latency/memory profiling under real production load.

### 2. Canary Deployment (Low Risk)
- New model receives 5% of traffic.
- Real users see the new model's response.
- **Why:** To verify integration works and catch obvious regressions.

### Automated Canary with Flagger & Istio

```yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: fraud-api
  namespace: ml-serving
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-api
  service:
    port: 8080
  analysis:
    interval: 5m
    threshold: 3
    maxWeight: 100
    stepWeight: 10
    metrics:
    # Standard software metrics
    - name: request-success-rate
      thresholdRange:
        min: 99
      interval: 1m
    - name: request-duration
      thresholdRange:
        max: 200
      interval: 30s
    # ML-specific metric!
    - name: prediction-drift-psi
      thresholdRange:
        max: 0.1
      interval: 5m
```

---

## Interview Questions (30 Q&A) — See Original Below

---

## Q1: Explain the difference between Software CI/CD and ML CI/CD.
**Context:** Team was training models inside Jenkins PR builders.
**Answer:** Software CI/CD builds binaries. ML CI/CD has three triggers: Code (Git PR), Data (Drift alert), and Model (Registry promotion). You cannot train a model on every PR (takes too long). Separate CI (code testing) from CT (Continuous Training - triggered by Airflow/Kubeflow).

## Q2: How do you handle model versioning and rollback strategies?
**Context:** Bad model deployed, took 45 minutes to rollback.
**Answer:** Version Code (Git), Data (DVC), and Model (MLflow run ID). Use ArgoCD for GitOps deployment. The K8s manifest pointing to `s3://model/v4` lives in Git. To rollback, you simply `git revert` the manifest repo. ArgoCD instantly syncs K8s to pull the older model.

## Q3: How do you handle large model (4GB+) timeouts during deployment?
**Context:** Jenkins pipelines timing out waiting for K8s pods to become healthy.
**Answer:** Decouple CD from K8s status. Jenkins should only update the GitOps repo and exit immediately. ArgoCD handles the async pulling and syncing. Ensure K8s readiness probes have a high `initialDelaySeconds` (e.g., 120s) so K8s doesn't kill the pod while loading weights into VRAM.

## Q4: How do you implement automated A/B testing in your CD pipeline?
**Context:** Need to test models safely in production.
**Answer:** CD pipeline updates Istio/KServe routing rules. Deploy Model B alongside Model A. CD script sets Istio to route 10% traffic to B. Wait for metrics collection. If CTR improves, CD script automatically updates Istio to 100%.

## Q5: How do you test ML code in CI without running a full training job?
**Context:** CI pipeline took 8 hours to run on every commit.
**Answer:** Use a "dummy" dataset (100 rows). CI runs the training code on the dummy data just to verify syntax, shape compatibility, and that the loss function doesn't throw a `NaN`. Full training only happens in the CT pipeline.

## Q6: How do you manage infrastructure as code (IaC) alongside ML code?
**Context:** K8s manifests drifting from actual deployed state.
**Answer:** Keep Terraform/K8s manifests in a separate Git repository from the ML training code. This enforces separation of concerns. Data scientists PR the model code; Platform engineers PR the IaC.

## Q7: What are the best practices for using GitHub Actions for MLOps?
**Context:** Moving away from Jenkins.
**Answer:** 1. Use OIDC instead of long-lived AWS/GCP secrets. 2. Use reusable workflows (e.g., `ml-training-workflow.yml`) so multiple teams share the same CI/CD standards. 3. Use self-hosted runners for heavy jobs to avoid GitHub's per-minute billing.

## Q8: How do you implement data quality gates in CI/CD?
**Context:** Corrupt data broke the nightly retraining job.
**Answer:** Before the training step begins, the CT pipeline runs a Great Expectations suite against the S3 dataset. If the schema changed or nulls exceed 5%, the pipeline fails and halts *before* wasting GPU hours on training.

## Q9: How do you handle database migrations linked to ML model deployments?
**Context:** Model required a new feature, which required a new DB column.
**Answer:** Decouple them. Phase 1: Deploy DB schema migration (adds column). Phase 2: Deploy data pipeline (populates column). Phase 3: Deploy ML model (uses column). Never bundle schema migrations and ML deployments in the same CD step.

## Q10: How do you ensure reproducibility in your CI/CD pipelines?
**Context:** Model trained in CI had different accuracy than model trained locally.
**Answer:** Dockerize the training environment. The exact `Dockerfile` (with pinned `poetry.lock` and CUDA versions) must be used locally and in the CI pipeline. Use MLflow to track the Git commit hash of the code that triggered the run.

## Q11: What is a Shadow Deployment in the context of CD?
**Context:** Evaluating a high-risk medical model.
**Answer:** CD pipeline deploys the model, but sets Istio to "mirror" traffic. The model receives 100% of prod requests, makes predictions, and logs them to Kafka, but the responses are discarded. Allows safe validation of latency and accuracy.

## Q12: How do you test inference endpoints in CI?
**Context:** Model deployed but failed due to JSON serialization errors.
**Answer:** CI pipeline builds the inference Docker image, spins it up using `docker-compose` or Minikube, and fires synthetic payload tests (`pytest`) against the REST API to ensure input/output schemas match expectations.

## Q13: How do you manage multi-environment progression?
**Context:** Pushing from Dev -> Staging -> Prod.
**Answer:** Model artifact is built *once* in Dev. The exact same Docker image/S3 artifact is promoted through environments. Environment-specific configs (DB URLs) are injected via K8s ConfigMaps. Never rebuild the image for Prod.

## Q14: How do you use Feature Flags in ML deployments?
**Context:** Need to instantly disable a new model feature.
**Answer:** Wrap the model invocation or specific preprocessing steps in a feature flag (LaunchDarkly/Split). If the model fails, toggle the flag to bypass it without needing a full K8s rollback.

## Q15: How do you handle CI/CD for edge devices (IoT/Mobile)?
**Context:** Deploying models to 10k cameras.
**Answer:** CI pipeline converts the model (ONNX/TensorRT). CD pipeline pushes the artifact to an IoT Device Management service (AWS IoT Greengrass), which handles OTA (Over-The-Air) updates to the fleet asynchronously.

## Q16: Explain "Continuous Training" (CT).
**Context:** Models were decaying in production.
**Answer:** An Airflow DAG scheduled weekly, or triggered by an Evidently AI drift alert. It pulls the latest data, runs the training code, evaluates against a holdout set, and if better, registers the new model version automatically.

## Q17: How do you track model lineage in CI/CD?
**Context:** Audit required knowing exactly how a model was built.
**Answer:** MLflow tracks the S3 URI of the dataset, the Git commit of the code, the hyperparameter dictionary, and the Docker image digest used for training. This creates an immutable chain of custody.

## Q18: Using Spinnaker vs ArgoCD for ML CD.
**Context:** Choosing a CD tool.
**Answer:** Spinnaker is powerful for complex, multi-cloud imperative pipelines (e.g., deploying to AWS and GCP simultaneously). ArgoCD is declarative GitOps, deeply integrated with Kubernetes. For K8s-native MLOps, ArgoCD is generally preferred.

## Q19: How do you optimize Docker image builds in CI?
**Context:** Jenkins taking 20 minutes to build PyTorch images.
**Answer:** Use Docker build caching (`--cache-from`). Separate `requirements.txt`/`poetry.lock` copy steps from the rest of the code. Put heavy base images (CUDA) in a lower layer so they are cached.

## Q20: How do you handle secrets in ML CI/CD?
**Context:** AWS keys hardcoded in Jenkinsfiles.
**Answer:** Use HashiCorp Vault. Jenkins authenticates to Vault via an AppRole, fetches a short-lived, dynamically generated AWS STS token valid for 15 minutes, deploys the model, and the token expires.

## Q21: Implementing Blue/Green Deployments.
**Context:** Zero-downtime ML updates.
**Answer:** Deploy Model V2 alongside Model V1. Both are fully scaled. Switch the K8s Service selector (or Ingress route) to point 100% of traffic to V2 instantly. Keep V1 running for 10 minutes as a fast fallback, then terminate V1.

## Q22: What is the role of DVC in CI/CD?
**Context:** Versioning datasets.
**Answer:** DVC creates a `.dvc` file containing the MD5 hash of the dataset in S3. You commit the `.dvc` file to Git. In the CI pipeline, `dvc pull` fetches the exact data version tied to that Git commit.

## Q23: How do you handle configuration drift?
**Context:** Someone manually edited a K8s deployment via `kubectl`.
**Answer:** ArgoCD detects that the live cluster state differs from the Git state. It marks the app as "OutOfSync" and can be configured to automatically overwrite the manual changes, enforcing Git as the source of truth.

## Q24: Automating hyperparameter tuning in CI/CD.
**Context:** Need to tune automatically on retrain.
**Answer:** The CT pipeline spawns an Optuna/Ray Tune job. It runs 50 trials in parallel on a K8s GPU cluster. The orchestrator tracks the best trial in MLflow and promotes that specific run to the Model Registry.

## Q25: How do you handle schema evolution in ML endpoints?
**Context:** V2 of the model requires a new input field.
**Answer:** Version the API route (`/v1/predict` vs `/v2/predict`). Deploy V2. Have clients migrate to `/v2`. Once V1 traffic hits zero, deprecate V1. Never introduce breaking schema changes to a live endpoint.

## Q26: Testing Data Pipelines in CI.
**Context:** Spark job logic was flawed.
**Answer:** Use Pytest with local Spark sessions. Create small, hardcoded Parquet files with known edge cases (nulls, extreme values). The CI pipeline runs the Spark transformations locally to assert the output matches expected results.

## Q27: How do you integrate model cards into CI/CD?
**Context:** Compliance team needed documentation for every deployment.
**Answer:** The CI pipeline automatically generates a markdown Model Card containing training metrics, dataset distributions, and bias test results, and attaches it as an artifact to the MLflow registry.

## Q28: Canary Analysis automation.
**Context:** Waiting 24 hours to manually check a canary is slow.
**Answer:** Use tools like Kayenta or Flagger. They automatically query Datadog metrics (latency, error rate) comparing the canary vs baseline. If metrics deviate significantly, it automatically triggers a rollback.

## Q29: Deploying multiple models to a single endpoint.
**Context:** Triton inference server managing 5 models.
**Answer:** The CD pipeline doesn't deploy a new container. It uploads the ONNX model to an S3 bucket and triggers a Triton API call to dynamically load the new model into GPU memory without restarting the pod.

## Q30: Security scanning in the ML CI pipeline.
**Context:** Preventing CVEs from reaching production.
**Answer:** In Jenkins, run Trivy or Clair against the built Docker image. If CRITICAL or HIGH vulnerabilities are found in the Python packages (e.g., a vulnerable version of `requests`), the build fails and blocks deployment.

---
*Next → [Cost Optimization](../cost-optimization/README.md)*
