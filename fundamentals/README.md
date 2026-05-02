# MLOps Fundamentals - Interview Questions

## Q1: Walk me through the end-to-end ML lifecycle as you've implemented it in production.

**Real-world context:**
In my last role, we had 25+ models running across fraud detection, recommendation, and risk scoring. The lifecycle wasn't a clean circle - it was messy, with different teams owning different stages and handoff problems everywhere.

**Answer:**

The lifecycle I've operated in production looks like this:

```
Data Ingestion (Kafka/S3)
    |
    v
Data Validation (Great Expectations / custom checks)
    |
    v
Feature Engineering (Spark batch + Flink streaming --> Feast)
    |
    v
Experimentation (Jupyter on K8s + MLflow tracking)
    |
    v
Training Pipeline (Airflow DAG --> K8s training job)
    |
    v
Model Validation Gate (champion vs challenger on holdout data)
    |
    v
Model Registry (MLflow -- version, stage, metadata)
    |
    v
CI/CD Pipeline (Jenkins/GH Actions --> build container --> deploy)
    |
    v
Deployment (KServe/custom Flask on K8s -- canary rollout)
    |
    v
Monitoring (Datadog infra + Evidently drift + business metrics)
    |
    v
Retrain Trigger (drift alert / scheduled / manual)
```

The parts that break most often in production:
1. **Data validation** - upstream teams change schemas without telling you. A column rename at 2 AM breaks your nightly training pipeline.
2. **Feature consistency** - features computed differently in training (Pandas on a notebook) vs serving (Java microservice). This is training-serving skew, and it's the silent killer.
3. **Model validation gate** - teams skip this when they're rushing. A model goes live, metrics look fine for a week, then revenue drops 3% and nobody connects it to the model change.

**Tools:** Kafka, Airflow, Spark, Feast, MLflow, Jenkins, Kubernetes, Datadog

---

## Q2: Explain the difference between a feature store's offline and online layer. When did you actually need both?

**Real-world context:**
We had a fraud detection model that needed features from two sources: historical aggregates (customer's average transaction over 90 days) and real-time signals (transaction count in the last 5 minutes). Without a feature store, we had Pandas code computing features for training and a completely separate Java service computing them for serving. They drifted apart, and the model started performing 8% worse in production.

**Answer:**

The offline store is a batch-oriented data store (Parquet on S3, BigQuery, Hive table) that serves historical feature values for training. When you build a training dataset, you query the offline store with point-in-time correctness: "give me the features for customer X as they were on January 15th at 2pm," not as they are today.

The online store is a low-latency key-value store (Redis, DynamoDB) that serves the latest feature values for real-time inference. When a scoring request comes in, the serving endpoint queries Redis for pre-computed features.

```
Training Time:
  Offline Store (S3/BigQuery) --batch query--> Training Dataset
  Features are materialized by a Spark job (nightly)

Serving Time:
  Request --> Online Store (Redis) --<5ms--> Feature Vector --> Model --> Response
  Features are materialized by Flink streaming job (real-time)
```

**Trade-offs:**
- Running Feast adds operational overhead (Redis cluster, materialization jobs, schema management)
- Worth it when you have 5+ models sharing features; overkill for 1-2 models

---

## Q3: How do you detect and handle data drift in production? Give me a real example.

**Real-world context:**
Our credit scoring model ran fine for 6 months. Then Q4 hit, and the approval rate started climbing while default rates also climbed. Turned out, a new customer acquisition channel was bringing in a completely different demographic. The feature distributions shifted, but nobody was watching.

**Answer:**

I monitor drift at three levels:

**1. Feature-level drift (input monitoring):**
I compute Population Stability Index (PSI) on every input feature, comparing the current week's distribution against the training data distribution.

```python
def compute_psi(reference, current, bins=10):
    ref_hist = np.histogram(reference, bins=bins)[0] / len(reference)
    cur_hist = np.histogram(current, bins=bins)[0] / len(current)
    ref_hist = np.clip(ref_hist, 0.001, None)
    cur_hist = np.clip(cur_hist, 0.001, None)
    psi = np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))
    return psi
```

**2. Prediction-level drift (output monitoring):**
If the model suddenly starts predicting "approve" 15% more often than last month, something changed.

**3. Performance-level drift (outcome monitoring):**
When ground truth labels arrive (30-60 days later for credit defaults), compare actual vs predicted. 

**How I set this up in production:**
- Prediction + feature logging goes to Kafka, then lands in S3
- A daily Airflow DAG runs PSI on each feature against the reference dataset
- Results feed into Datadog as custom metrics

---

## Q4: What's training-serving skew, and how have you dealt with it?

**Real-world context:**
Our recommendation model had 0.94 AUC offline. In production, click-through rate was worse than random. We spent two weeks debugging the model itself before discovering the problem was in the features.

**Answer:**

Training-serving skew is when the features a model sees during training are computed differently from the features it gets during inference. The model is mathematically correct, but getting different inputs.

**How it happens in practice:**
A data scientist writes feature computation in Pandas (`df.rolling(30).mean()`). A backend engineer re-implements it in Java for the serving path, but implements the sliding window slightly differently. A 10% difference in a single feature cascaded into completely wrong predictions.

**How I fix it:**
1. **Feature Store (prevention):** Define features once in Feast. Both training and serving use the same computation logic.
2. **Feature validation (detection):** For every prediction, I log the features alongside the prediction. A daily batch job re-computes features offline for a sample of recent predictions and compares them against what was used at serving time. If the values differ by more than 1%, we have skew.

---

## Q5: Batch inference vs real-time inference - how do you decide, and what are the operational differences?

**Real-world context:**
A product team wanted real-time churn predictions for a weekly email campaign. I pushed back and saved us 6 months of engineering and $40k/month in infrastructure.

**Answer:**

**My decision framework:**
```
Does the business need results in < 1 second?
  YES --> Real-time (K8s, KServe, Redis feature store)
  NO  --> Does the business need results in < 1 hour?
          YES --> Near-real-time (Spark Structured Streaming or micro-batch)
          NO  --> Batch (Airflow + Spark, run overnight)
```

**Operational differences that matter:**
For batch: I need good Airflow monitoring, data quality checks before scoring, and idempotent writes to the output table.
For real-time: I need HPA autoscaling tuned to custom RPS metrics, circuit breakers for feature store timeouts, async prediction logging to Kafka, and readiness probes that verify the model is loaded into VRAM.

---

## Q6: How do you design an effective Model Registry workflow to prevent untested models from hitting production?

**Real-world context:**
A junior data scientist accidentally deployed a model that was trained on only 10% of the dataset because they overrode a tag in S3. Production failed for an hour.

**Answer:**

A Model Registry (like MLflow or Vertex AI Registry) must act as a strict state machine, not just a file storage bucket.

**The Workflow:**
1. **Registration:** Training pipeline completes and registers the model in MLflow as `Stage: None`. 
2.  **Validation Gate (Automated):** A CI pipeline is triggered. It downloads the model, runs it against a Golden holdout dataset, and asserts that Accuracy > 0.85 and Latency < 50ms. 
3. **Promotion:** If tests pass, CI automatically promotes the model to `Stage: Staging`.
4. **Approval Gate (Human):** A senior engineer or PM reviews the metrics in MLflow. They click a button in the UI (or merge a PR in GitOps) to transition the model to `Stage: Production`.
5. **Deployment:** The CD pipeline *only* listens for the `Transition to Production` event. It packages the artifact tied to that specific registry version and deploys it.

**Crucially:** Human operators are completely stripped of IAM permissions to deploy to the K8s cluster directly. They can only click "Approve" in the registry.

---
*[Back to README](../README.md)*
