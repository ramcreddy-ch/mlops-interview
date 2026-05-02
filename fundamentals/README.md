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

**Trade-offs:**
- Fully automated retraining is risky for regulated use cases (finance, healthcare) where human approval is required
- Over-validating slows down iteration speed; under-validating causes production incidents

**Common mistakes:**
- Treating ML deployment like application deployment (no validation gate)
- Not versioning training data alongside model code
- Monitoring only infrastructure metrics and ignoring model quality metrics

**Follow-up questions interviewers ask:**
- "How do you handle data versioning?"
- "What happens when the model validation gate fails?"
- "How do you decide between scheduled vs triggered retraining?"

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

The key property is that both layers use the same feature definitions written once in code. Feast does this well - you define a feature in Python, and Feast handles materializing it to both offline (batch) and online (streaming).

**When you don't need both layers:**
- If you only do batch scoring (nightly predictions stored in a database), you only need the offline store
- If your model has no historical features (purely request-time features like "is this URL valid"), you don't need a feature store at all

**Trade-offs:**
- Running Feast adds operational overhead (Redis cluster, materialization jobs, schema management)
- Worth it when you have 5+ models sharing features; overkill for 1-2 models

**Common mistakes:**
- Using different code paths for offline and online feature computation (the whole point of a feature store is to prevent this)
- Not implementing point-in-time correctness, leading to data leakage during training

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

# PSI < 0.1 = stable
# PSI 0.1-0.2 = moderate drift, investigate
# PSI > 0.2 = significant drift, trigger retrain
```

**2. Prediction-level drift (output monitoring):**
If the model suddenly starts predicting "approve" 15% more often than last month, something changed - even if we don't know which feature drifted.

**3. Performance-level drift (outcome monitoring):**
When ground truth labels arrive (30-60 days later for credit defaults), compare actual vs predicted. This is the ultimate measure but comes with a delay.

**How I set this up in production:**
- Prediction + feature logging goes to Kafka, then lands in S3
- A daily Airflow DAG runs PSI on each feature against the reference dataset
- Results feed into Datadog as custom metrics
- Alert thresholds: PSI > 0.2 on any feature = PagerDuty alert

**Trade-offs:**
- Not all drift matters. Seasonal changes are expected (e-commerce spikes in December). Setting thresholds too tight causes alert fatigue
- PSI works for numeric features. For categorical features, I use chi-squared tests
- Ground truth delay means you're always reacting late; feature drift monitoring gives you early warning

**Common mistakes:**
- Only monitoring infrastructure (latency, error rate) and ignoring data quality
- Using the wrong reference period (comparing against all-time data instead of the period the model was trained on)
- Alerting on every small PSI change without a stability window (5-minute blips vs sustained drift)

---

## Q4: What's training-serving skew, and how have you dealt with it?

**Real-world context:**
Our recommendation model had 0.94 AUC offline. In production, click-through rate was worse than random. We spent two weeks debugging the model itself before discovering the problem was in the features.

**Answer:**

Training-serving skew is when the features a model sees during training are computed differently from the features it gets during inference. The model is mathematically correct - it's doing exactly what it was trained to do - but it's getting different inputs.

**How it happens in practice:**

A data scientist writes feature computation in Pandas:
```python
# Training: Pandas
df['avg_session'] = df.groupby('user_id')['duration'].rolling(30).mean()
```

A backend engineer re-implements it in Java for the serving path:
```java
// Serving: Java
double avgSession = totalDuration / sessionCount;
// But sessionCount excludes sessions < 5 seconds...
```

That "sessions < 5 seconds" filter introduces a 10-20% difference in the feature value. The model sees inputs it was never trained on.

**How I fix it:**

1. **Feature Store (prevention):** Define features once in Feast. Both training and serving use the same computation logic. This is the best long-term fix.

2. **Feature validation (detection):** For every prediction, I log the features alongside the prediction. A daily batch job re-computes features offline for a sample of recent predictions and compares them against what was used at serving time. If the values differ by more than 1%, we have skew.

3. **Policy (process):** No feature can be implemented in two separate codebases. If a data scientist adds a feature, the serving-side computation must use the same library or the feature store's online materialization.

**Common mistakes:**
- Assuming skew only happens with complex features. Even simple things like timestamp handling (timezone differences between training data and serving) cause skew
- Not logging serving-time feature values, which makes skew impossible to detect after the fact

---

## Q5: Batch inference vs real-time inference - how do you decide, and what are the operational differences?

**Real-world context:**
I've been asked to build "real-time" inference for use cases that absolutely didn't need it. A product team wanted real-time churn predictions for a weekly email campaign. I pushed back and saved us 6 months of engineering and $40k/month in infrastructure.

**Answer:**

| Aspect | Batch Inference | Real-time Inference |
|--------|----------------|---------------------|
| Latency need | Minutes to hours OK | Milliseconds required |
| Example | Score all customers overnight for tomorrow's marketing campaign | Score this credit card transaction before authorization |
| Infra | Spark/Airflow job on schedule | K8s deployment with autoscaling, Redis for features |
| Cost | Pay per job (predictable) | Pay per pod-hour (24/7) |
| Failure impact | Job fails, retry next morning | Endpoint down = business impact now |
| Complexity | Low (stateless batch job) | High (load balancing, circuit breakers, health checks, autoscaling) |
| GPU usage | Burst (high for 2 hours, then idle) | Sustained (right-sized for 24/7) |

**My decision framework:**

```
Does the business need results in < 1 second?
  YES --> Real-time (K8s, KServe, Redis feature store)
  NO  --> Does the business need results in < 1 hour?
          YES --> Near-real-time (Spark Structured Streaming or micro-batch)
          NO  --> Batch (Airflow + Spark, run overnight)
```

**Operational differences that matter:**

For batch: I need good Airflow monitoring (task duration, SLA misses), data quality checks before scoring, and idempotent writes to the output table (so retries don't create duplicates).

For real-time: I need HPA/VPA autoscaling tuned to custom metrics (requests per second, not CPU), circuit breakers for feature store timeouts, async prediction logging to Kafka (never log synchronously in the hot path), and readiness probes that verify the model is loaded.

**Follow-up questions:**
- "Have you seen teams build real-time when batch was sufficient?"
- "How do you handle the transition from batch to real-time for a model that outgrows batch?"

---

*[Back to README](../README.md)*
