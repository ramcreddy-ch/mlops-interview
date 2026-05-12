# Section 13: Interview Prep — Deep Scenario-Based Questions
## Staff/Principal Level | Real Production Depth Required

---

## How to Answer MLOps Interview Questions

The formula that wins at senior levels: **S.T.A.R.D.**
- **S**ituation — real context (scale, constraints)
- **T**ool — what you chose and why (including what you rejected)
- **A**rchitecture — the design decision with trade-offs
- **R**esult — measurable outcome
- **D**ebug — what went wrong and how you fixed it

Never give a textbook definition. Always anchor to a real or realistic scenario.

---

## Section A: System Design Questions

### Q1: "Design a real-time ML inference system for 10,000 requests/second"

**Weak answer:** "I'd use Kubernetes and a REST API."

**Strong answer (full depth):**

```
My design for 10K RPS fraud detection:

CAPACITY PLANNING FIRST:
- 10K RPS × 50ms avg latency = 500 concurrent requests in flight
- Each pod handles ~100 concurrent requests safely
- Need minimum 5 pods (× 2 for redundancy = 10 baseline)

ARCHITECTURE:
1. Kong API Gateway → rate limiting, auth, routing
2. FastAPI pods (10-30 based on HPA) on Kubernetes
3. Redis Cluster for feature store (<5ms reads)
4. XGBoost model loaded in-process (not a remote call)

FEATURE FETCHING OPTIMIZATION:
- Redis pipeline for batching multiple GET calls
- Circuit breaker (50ms timeout) with fallback features
- Result: 3ms P99 feature fetch (not 30ms with naive approach)

AUTOSCALING:
- HPA based on requests_per_second (NOT CPU — ML is not CPU-correlated)
- Pre-scale before traffic peaks (scheduler: 8AM/12PM/6PM)
- KEDA for Kafka-based scaling when consuming from event streams

FAILURE MODES I'VE HANDLED:
1. Redis down: Fallback feature vector keeps serving (degraded accuracy)
2. Model version mismatch: Blue-green deploy, both versions always running
3. Traffic spike (3x): Pre-provisioned capacity + fast HPA response

RESULT: 8,000 TPS with P99 < 80ms, 99.99% availability
```

---

### Q2: "How would you detect and handle model drift in production?"

**Strong answer:**

```
I monitor three separate drift signals, each with different tooling:

1. DATA DRIFT (input features):
   Tool: Evidently AI, running hourly batch jobs in Airflow
   Metric: PSI (Population Stability Index) per feature
   Threshold: PSI > 0.1 = warning, PSI > 0.2 = alert + auto-retrain trigger
   
   Example: In Q4, PSI for "transaction_amount" spiked to 0.45 because
   holiday shopping inflated average transaction sizes. Model retrained
   within 4 hours automatically.

2. PREDICTION DRIFT (output distribution):
   Metric: Compare histogram of today's fraud probabilities vs 30-day baseline
   Detection: KL divergence or Population Stability Index on predictions
   Why useful: Detects issues even before ground truth arrives
   
   Example: Prediction drift alert fired 6 hours before business noticed
   fraud rate was dropping — model was becoming too conservative.

3. CONCEPT DRIFT (model accuracy):
   Only measurable when ground truth arrives (e.g., chargebacks, 30-90 days later)
   Metric: Track precision, recall, F1 on labeled outcome data
   Trigger: F1 drops >5% from baseline → alert + mandatory retrain review
   
AUTOMATED RETRAIN TRIGGER:
   if psi_max > 0.2 or f1_drop > 0.05:
       trigger_airflow_dag("fraud_model_retrain")
   
HUMAN IN THE LOOP:
   Model is automatically retrained and sent to Staging.
   Platform engineer reviews the validation metrics.
   Automated promotion only if (new_auc - prod_auc) > 0.005.
```

---

### Q3: "Walk me through a production ML incident you've debugged"

**Strong answer structure:**

```
INCIDENT: Fraud model false positive rate jumped from 2% to 18% overnight.
Time: 2:34 AM. PagerDuty woke me up.

INITIAL SYMPTOMS:
- Datadog alert: "Approval rate dropped from 94% to 82%"
- No infrastructure alerts (CPU, memory, latency all normal)
- No deployment in last 24 hours

INVESTIGATION (chronological):

Step 1 (2:36 AM): Check if model changed
  mlflow ui → Production model is v47, same as yesterday. ✅ Not the model.

Step 2 (2:38 AM): Check feature drift
  Grafana → "null_feature_count" metric jumped 8x for "merchant_category_code"
  This is the upstream data quality signal! 🔍

Step 3 (2:40 AM): Trace upstream
  ELK query: filter logs where "merchant_category_code" IS NULL
  Found: all transactions from Partner API "RetailX" sending null MCCs
  Confirmed: RetailX updated their API at 2:00 AM, dropped the MCC field

Step 4 (2:43 AM): Root cause confirmed
  Model was trained with MCC as feature. When null, XGBoost follows default
  tree branch → biases toward "is_fraud=True" for many valid users.

IMMEDIATE FIX:
  - Hot-patch inference service: impute null MCC with mode value "5411" (grocery)
  - This brought FPR from 18% back to 3.5% within 5 minutes
  - kubectl set env deployment/fraud-api MCC_FALLBACK=5411

PERMANENT FIX (deployed next day):
  1. Great Expectations check added: "mcc must not be null > 0.5% of requests"
  2. Upstream contract with RetailX formalized and tested
  3. Model retrained with null-robust features (removed MCC from critical features)

POSTMORTEM:
  Timeline, root cause, contributing factors, 5 action items
  Shared with engineering org (blameless)
```

---

## Section B: Kubernetes + Infrastructure Questions

### Q4: "How do you manage GPU resources for multiple ML teams on a shared cluster?"

```
REAL SCENARIO: 3 ML teams, 20 GPUs, constant conflicts

SOLUTION IMPLEMENTED:

1. NAMESPACE ISOLATION:
   kubectl create namespace ml-risk
   kubectl create namespace ml-marketing
   kubectl create namespace ml-search

2. RESOURCE QUOTAS (hard limits per namespace):
   ml-risk:      requests.nvidia.com/gpu: "8"   # Risk gets 8 GPUs
   ml-marketing: requests.nvidia.com/gpu: "4"   # Marketing gets 4
   ml-search:    requests.nvidia.com/gpu: "4"   # Search gets 4
   (4 GPUs kept as shared burst pool)

3. PRIORITY CLASSES (real-time inference beats batch training):
   PriorityClass "realtime-inference": 1000
   PriorityClass "batch-training":      100
   PriorityClass "dev-experiments":      10
   
   K8s will evict a batch training job to make room for a real-time
   inference pod. Data scientists hate this. Platform engineers require it.

4. SCHEDULING:
   Real-time inference: Kubernetes default scheduler (fast scheduling)
   Distributed training: Volcano gang scheduler (all-or-nothing allocation)

5. COST CHARGEBACK:
   Kubecost reads labels (team: risk, workload_type: training)
   Monthly report to each team's cost center

RESULT:
  Zero incidents of team A blocking team B for 8 months.
  Real-time inference SLA maintained during peak training periods.
```

---

### Q5: "What's the difference between HPA, VPA, and KEDA? When do you use each?"

```
HPA (Horizontal Pod Autoscaler):
  - Adds/removes pod REPLICAS
  - Triggers on: CPU, Memory, custom metrics (RPS, queue depth)
  - Best for: Stateless inference services
  - Limitation: Triggers on average metric — slow to respond to sudden spikes
  
  Example (inference):
    scaleUp on: requests_per_second > 500 per pod
    scaleDown after: 5 minutes below threshold (prevents flapping)

VPA (Vertical Pod Autoscaler):
  - Changes CPU/Memory limits on existing pods (restarts them)
  - Use mode: "Recommend" only — not "Auto" for production ML
  - Why not Auto: Restarting a pod that holds a 5GB model in VRAM = 2-min outage
  - Best for: Right-sizing initial resource requests

  Example (right-sizing):
    Before VPA: DS requested 64GB RAM (conservative)
    VPA Recommendation: 12GB (actual usage)
    Saved: $4,000/month in wasted memory

KEDA (Kubernetes Event-Driven Autoscaler):
  - Adds/removes REPLICAS based on external event sources
  - Can scale TO ZERO (unlike HPA which has minimum 1)
  - Triggers on: Kafka consumer lag, SQS queue depth, cron schedule
  - Best for: Async ML workloads (batch scoring from queue)

  Example (batch scoring):
    Scale based on: kafka_consumergroup_lag for "batch-scoring-requests" topic
    If lag = 0: scale to 0 (no cost when no work)
    If lag > 1000: scale to 10 pods immediately

PRODUCTION PATTERN:
  Real-time serving: HPA (RPS-based) + always >= 2 replicas
  Batch scoring: KEDA (queue-depth-based) + scale to zero
  Training jobs: Fixed resource request + VPA recommendations for future jobs
```

---

## Section C: ML Engineering Questions

### Q6: "How do you prevent training-serving skew?"

```
TRAINING-SERVING SKEW: Silent killer of production ML models.

WHAT IT IS:
  Training: feature computed in Python Pandas
    avg_spend = df.groupby('user_id')['amount'].rolling('30D').mean()
  
  Serving: feature computed in Java (legacy service)
    avgSpend = transactions.stream()
      .filter(t -> t.date.isAfter(now.minusDays(30)))  // BUG: calendar days, not rolling
      .mapToDouble(t -> t.amount).average().orElse(0);
  
  These compute DIFFERENT VALUES. Model was trained on one definition,
  served with another. AUC dropped 8% in production vs offline.

HOW TO PREVENT:

1. FEATURE STORE (primary fix):
   Single definition of every feature in Feast/Tecton.
   Training reads from offline store, serving reads from online store.
   Both computed by the same Flink job.

2. LOG AND COMPARE:
   In production, log the serving-time feature vector for each request.
   Nightly Airflow job re-computes those features from offline store.
   Alert if difference > 0.5%:
   
   serving_value = redis.get("user:123:avg_spend_30d")  # 150.0
   offline_value = snowflake.query("SELECT avg_spend_30d FROM features WHERE user_id='123'")  # 148.7
   skew = abs(serving_value - offline_value) / offline_value = 0.9% → ALERT!

3. SCHEMA VALIDATION:
   Training data schema is tested in CI.
   Serving input schema is validated at API layer.
   If schema changes → pipeline fails before reaching production.

REAL EXAMPLE:
  Found skew in "days_since_last_login" feature:
  Training: computed at time of label assignment
  Serving: computed at time of prediction (could be hours later)
  Fix: Feature store materialized this with exact timestamp semantics
```

---

### Q7: "A/B testing vs Shadow deployments — when do you use which?"

```
SHADOW DEPLOYMENT:
  - New model gets 100% of traffic, responses are DISCARDED (user never sees them)
  - Production users are completely unaffected
  - You observe: latency, resource usage, output distribution
  - You DON'T observe: business impact (clicks, revenue)
  
  Use when:
    - Model is risky (medical, financial — can't risk wrong answers on users)
    - You need to understand resource profile before committing
    - Testing infrastructure (is new Triton config stable?)
  
  Duration: 7-14 days
  Risk: Zero user impact. Only cost is compute for the shadow model.

A/B TESTING:
  - Split users (deterministically!) between old and new model
  - Both responses reach real users
  - You observe: click rates, revenue, conversion, NPS
  - This is the only way to measure TRUE business impact
  
  Use when:
    - Shadow deployment confirmed model is safe
    - You need statistical proof of improvement (not just offline AUC)
    - Business stakeholders need to see ROI before full rollout
  
  Duration: Until statistical significance (minimum 1000 events per variant)
  Risk: A% of users get potentially worse experience

MY PRODUCTION WORKFLOW:
  1. Shadow (7 days): Confirm latency, stability, no crashes
  2. A/B 5% (3 days): Sanity check on business metrics
  3. A/B 25% (7 days): Gather statistical significance
  4. Promote to 100% if metric improvement is significant (p < 0.05)

CRITICAL: Deterministic routing for A/B
  BAD:  random.choice() per request (user gets A sometimes, B sometimes)
  GOOD: hash(user_id + experiment_name) % 100 → stable assignment
```

---

## Section D: Behavioral + Leadership Questions

### Q8: "Tell me about a time you drove MLOps adoption against resistance."

```
SITUATION:
  Joined a team where 20 data scientists deployed models by emailing zip files
  to a sysadmin who manually SSH'd into servers. Leadership said "it works,
  don't break it." I was brought in to modernize the stack.

APPROACH:

1. BUILT TRUST FIRST (Month 1):
   Did NOT immediately say "your process is wrong."
   Embedded with the DS team for 2 weeks.
   Understood their actual pain points:
   - "It takes 3 days to get a model into production" ← THEIR problem
   - "I can never reproduce last quarter's model" ← THEIR problem
   Framed MLOps as solving THEIR problems, not following MY standards.

2. STARTED WITH A GOLDEN PATH (Month 2):
   Built a one-click deploy pipeline for the simplest model type (XGBoost).
   Made it FASTER than the old process (hours vs days).
   Demonstrated it publicly. Let one DS try it first. They became the advocate.

3. GRADUATED REQUIREMENTS (Month 3-6):
   Didn't mandate everything at once.
   Started with: "All new models must use the pipeline. Existing models grandfathered."
   Added requirements one at a time as the team got comfortable.

4. MADE FAILURE VISIBLE:
   Set up a dashboard showing: "Models with no drift monitoring" and "Models
   not retrained in 90 days." Made the risk visible. Let the DSs self-identify
   their tech debt rather than being told.

RESULT:
  - 6 months: 80% of new models on the pipeline
  - 12 months: 95% adoption, including migrated legacy models
  - DS team cited "deployment confidence" as top improvement in eng survey
  - Myself? Promoted to Staff Engineer.

KEY LESSON: MLOps adoption is a change management problem, not a technical one.
```

---

## Section E: Cost Optimization Questions

### Q9: "How do you reduce ML infrastructure costs by 60%?"

```
REAL ENGAGEMENT: $120K/month → $48K/month in 4 months.

AUDIT (Month 1) — found:
  - Training: 10 p3.8xlarge On-Demand ($12.24/hr × 24/7 = $8.8K/month)
  - Serving: 30 pods all running, even at 3 AM (zero traffic)
  - Data: 50TB in S3 Standard (never accessed after 30 days)
  - Experiments: 500 abandoned MLflow runs leaving 2TB of artifacts

FIXES:

1. SPOT INSTANCES FOR TRAINING (saves ~70% on training):
   - Implemented checkpointing in all training scripts
   - Switched to Spot p3 instances
   - Added retries for spot interruptions
   - Savings: $8.8K → $2.6K/month for training

2. SCALE-TO-ZERO FOR SERVING (saves ~40% on serving):
   - Implemented KEDA for batch scoring services (0 pods when no work)
   - KServe serverless for 8 low-traffic models (scale to zero at night)
   - Kept 2 min-replicas for critical real-time models
   - Savings: 30 pods → average 12 pods (60% reduction)

3. S3 LIFECYCLE RULES (saves on storage):
   s3://ml-data/raw/: Standard → Infrequent Access after 30d → Glacier after 90d
   s3://mlflow-artifacts/: Delete old experiment artifacts (> 180 days, non-Production)
   Savings: $3,200/month → $900/month

4. RIGHT-SIZING WITH VPA:
   Average GPU utilization was 18% (over-provisioned).
   Moved 6 models from ml.g4dn.2xlarge to ml.g4dn.xlarge.
   Savings: $2,100/month

5. RESERVED INSTANCES for baseline load:
   3 always-running critical models → 1-year Reserved Instances (40% discount)

TOTAL SAVINGS: $72K/month → $48K/month. 60% reduction.
```

---

*Next → [Section 14: Advanced Topics](../advanced-topics/README.md)*
