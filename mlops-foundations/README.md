# Section 1: MLOps Foundations — Production Reality

---

## 1.1 What is MLOps in Real Companies?

MLOps is NOT about tools. It is about **reducing the time from a model idea to production value, safely, repeatably, at scale**.

At a mid-size bank with 40 data scientists and 80 production models, MLOps meant:
- A data scientist pushes code → model is tested, validated, containerized, and deployed to K8s **without involving a single platform engineer**
- Every model in production has its data lineage, training code, and drift metrics visible in a single dashboard
- A bad model is **rolled back in 90 seconds**, not 3 hours

### What MLOps is NOT (common misconceptions):
| Misconception | Reality |
|---|---|
| "We use MLflow so we do MLOps" | That is one tool. MLOps is the full workflow |
| "MLOps = DevOps for ML" | Close, but model behavior is non-deterministic — new failure modes |
| "We need MLOps only at scale" | You need it the moment you have >1 model in production |
| "MLOps is a data scientist's job" | It requires platform engineers, data engineers, ML engineers |

---

## 1.2 DevOps vs MLOps vs LLMOps — Real Differences

```
                    DEVOPS
  Code → Build → Test → Deploy → Monitor
  (Deterministic: same code = same output)

                    MLOPS
  Data + Code → Train → Validate → Deploy → Monitor → Retrain
  (Non-deterministic: same code + different data = different model behavior)

                    LLMOPS
  Data + Code + Prompts → Fine-tune/RAG → Evaluate → Deploy → Monitor → Prompt-tune
  (Non-deterministic + stochastic outputs + cost-per-token concerns)
```

### Detailed Comparison Table:

| Dimension | DevOps | MLOps | LLMOps |
|---|---|---|---|
| Artifact | Docker image | Docker image + Model weights | Docker image + Model weights + Prompt templates |
| Testing | Unit/Integration tests | Offline metrics + data validation | Offline eval + LLM-as-judge + human eval |
| Rollback trigger | 5xx rate, latency | Model drift, business metric drop | Hallucination rate, cost spike, toxicity |
| Primary cost | Compute | GPU training compute | Inference token costs (LLMs charge per token) |
| Reproducibility | Pin library versions | Pin library + data + random seed | Pin library + data + model version + prompt version |
| Monitoring | CPU, Memory, Latency | CPU + Data drift + Prediction drift | CPU + Hallucination rate + Context length + Latency |
| Team involved | DevOps/SRE | Platform Eng + ML Eng + Data Eng | All above + Prompt Engineers + AI Safety |

### Real Failure Example: Why MLOps ≠ DevOps

**Incident:** A DevOps team "owned" ML deployments. They used standard blue-green deploys. Model V2 was deployed. All infra metrics were green (CPU ✅, Memory ✅, Latency ✅). But 3 days later, revenue dropped 8%.

**Root cause:** Model V2 had a training-serving skew. It was trained on Q3 data but deployed in Q4 with different user behavior patterns. DevOps monitoring had zero visibility into this because they monitored infrastructure, not model behavior.

**Fix:** Dedicated MLOps monitoring with **prediction distribution drift** alerting using Evidently AI.

---

## 1.3 The ML Lifecycle — Every Stage with Real-World Pipelines

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION ML LIFECYCLE                               │
├──────────────┬──────────────┬──────────────┬──────────────┬─────────────┤
│   DATA ZONE  │  TRAINING    │  VALIDATION  │  SERVING     │  FEEDBACK   │
│              │  ZONE        │  ZONE        │  ZONE        │  LOOP       │
├──────────────┼──────────────┼──────────────┼──────────────┼─────────────┤
│ 1. Ingestion │ 5. Feature   │ 8. Offline   │ 11. CI/CD    │ 14. Label   │
│   (Kafka/S3) │    Engineering│    Metrics   │    Pipeline  │    Store    │
│              │    (Spark)   │    Gate      │              │             │
│ 2. Validation│ 6. Experiment│ 9. Shadow    │ 12. Canary   │ 15. Drift   │
│   (Great     │    Tracking  │    Deploy    │    Deploy    │    Monitor  │
│   Expectations│   (MLflow)  │              │              │             │
│              │              │ 10. A/B Test │ 13. Full     │ 16. Auto-   │
│ 3. Versioning│ 7. Training  │    Validation│    Rollout   │    Retrain  │
│   (DVC/      │    Pipeline  │              │              │    Trigger  │
│    LakeFS)   │    (Airflow/ │              │              │             │
│              │    Kubeflow) │              │              │             │
│ 4. Feature   │              │              │              │             │
│    Store     │              │              │              │             │
│    (Feast)   │              │              │              │             │
└──────────────┴──────────────┴──────────────┴──────────────┴─────────────┘
```

### Stage 1: Data Ingestion

**Real setup at a fintech company:**
```yaml
# Kafka topic for transaction events
Topic: txn-raw
  Partitions: 64
  Retention: 7 days
  Replication factor: 3

# Flink streaming job consumes and validates
Flink Job: txn-validator
  Parallelism: 32
  Checkpointing: every 60 seconds to S3
  Output: s3://data-lake/raw/transactions/year=2024/month=11/day=15/

# Airflow DAG triggers daily batch processing
DAG: daily-feature-pipeline
  Schedule: 0 2 * * *  (2 AM daily)
  Tasks: validate → deduplicate → feature-compute → store-to-feast
```

**Why this matters:** Raw data arrives with schema violations every day. A transaction might come in with `amount: null` or `currency: "XYZ"` (invalid ISO code). The validator catches this BEFORE it contaminates the training pipeline.

### Stage 2: Data Validation with Great Expectations

```python
# ge_suite.py — Real production validation suite
import great_expectations as ge

datasource = ge.get_context().sources.add_pandas("transactions")
suite = ge.get_context().add_expectation_suite("transaction_suite")

# Business rules encoded as expectations
suite.add_expectation(
    ge.core.ExpectationConfiguration(
        expectation_type="expect_column_values_to_not_be_null",
        kwargs={"column": "user_id"}
    )
)
suite.add_expectation(
    ge.core.ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={"column": "amount", "min_value": 0, "max_value": 1_000_000}
    )
)
suite.add_expectation(
    ge.core.ExpectationConfiguration(
        expectation_type="expect_column_values_to_match_regex",
        kwargs={"column": "currency", "regex": "^[A-Z]{3}$"}  # ISO 4217
    )
)

# If validation fails → Airflow task fails → Pipeline halts → Alert fires
```

**Production rule:** A failed data validation should NEVER silently pass. It should alert and halt the downstream pipeline. Silent data corruption is the #1 cause of stealth model degradation.

### Stage 3: Feature Engineering (The Hard Part)

```python
# pyspark_features.py — Production feature computation
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("fraud-feature-pipeline") \
    .config("spark.sql.shuffle.partitions", "400")  # tune for cluster size
    .getOrCreate()

txn = spark.read.parquet("s3://data-lake/raw/transactions/")

# Window functions for time-based features
user_30d_window = Window.partitionBy("user_id") \
    .orderBy("timestamp") \
    .rangeBetween(-30 * 86400, 0)  # 30 days in seconds

features = txn.withColumn(
    "avg_spend_30d",
    F.avg("amount").over(user_30d_window)
).withColumn(
    "txn_count_30d",
    F.count("*").over(user_30d_window)
).withColumn(
    "max_spend_30d",
    F.max("amount").over(user_30d_window)
)

# Write to feature store's offline store (Snowflake/S3 Parquet)
features.write \
    .mode("overwrite") \
    .partitionBy("year", "month", "day") \
    .parquet("s3://feature-store/offline/fraud_features/")
```

**CRITICAL TRAP:** The window function above uses `rangeBetween` on timestamps. If you use `rowsBetween`, you get a different definition of "30 days" at serving time. This is training-serving skew. The feature store exists to prevent this.

### Stage 4: The Full CI/CD Pipeline Flow

```
Data Scientist pushes code to Git
        │
        ▼
GitHub Actions / Jenkins CI
├── Lint Python (flake8/black)
├── Unit tests (pytest)
├── Integration tests (test against staging feature store)
├── Build Docker image
├── Push to ECR/GCR
        │
        ▼
MLflow Model Registry (Staging)
├── Log model with all metrics
├── Run validation suite against golden dataset
├── Gate: AUC must be > 0.85, Precision@K > 0.7
        │
        ▼
(Human approval OR automated gate)
        │
        ▼
ArgoCD detects Registry change → Updates K8s manifests
        │
        ▼
Canary Deploy (5% traffic → 25% → 100%)
        │
        ▼
Monitoring alerts activated (drift, latency, business KPIs)
        │
        ▼
Ground truth arrives (after 24-72h) → Compare with predictions
        │
        ▼
If drift detected → Auto-retrain trigger → Restart loop
```

---

## 1.4 The Three Maturity Levels of MLOps

### Level 0: Manual (Most teams here)
- Data scientists train models in Jupyter notebooks
- Models are deployed manually via SSH/FTP
- No reproducibility, no monitoring
- **Symptom:** "It worked on my laptop but fails in production"

### Level 1: Automated Training (Most interview targets)
- Training pipeline is automated (Airflow/Kubeflow)
- Model registry exists (MLflow)
- Basic monitoring (latency, error rates)
- Manual triggering of retraining
- **Symptom:** "We retrain monthly, which takes a full day"

### Level 2: Fully Automated (What FAANG runs)
- Continuous training triggers when drift detected
- Shadow deployments are standard
- Full lineage tracked end-to-end
- Models retrain, validate, and promote automatically
- Humans approve only major architecture changes
- **Symptom:** "Our fraud model retrained 14 times last week due to changing patterns"

---

## 1.5 Key Terms Every MLOps Engineer Must Own

| Term | Production Definition |
|---|---|
| **Training-Serving Skew** | Feature computation logic differs between training (Spark) and serving (Python). Silent accuracy killer. |
| **Data Drift** | Input feature distribution shifts (PSI > 0.2 is alarm threshold) |
| **Concept Drift** | The relationship between X and y changes (fraud patterns evolve). Detected only via ground truth. |
| **Model Staleness** | Model was trained on old data. Alert if training date > 30 days ago. |
| **Shadow Deployment** | New model receives traffic but responses are discarded. Safe evaluation. |
| **Canary Deployment** | New model receives % of live traffic with real impact on users. |
| **Point-in-Time Correctness** | Training features reflect values AT THE TIME of the event, not current values. |
| **Feature Store** | Central system managing feature definitions, offline (training) and online (serving) stores. |
| **Model Registry** | Version-controlled catalog of models with lifecycle states (None → Staging → Production). |
| **Data Lineage** | Audit trail of which data fed which features, which trained which model. |

---

*Next → [Section 2: Versioning & Data Management](../versioning/README.md)*
