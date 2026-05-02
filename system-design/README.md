# System Design - Interview Questions

## Q1: Design a real-time ML inference system using Kafka + Kubernetes

**Context:** Built this for fraud detection at 8K TPS, P99 < 80ms SLA.

**Architecture:**
```
Transaction --> Kafka (64 partitions) --> Flink (feature enrichment)
  --> K8s Inference Service (12 replicas, HPA on RPS)
  --> Decision Service (block/review/approve)
  --> Kafka predictions topic --> S3 Data Lake
```

**Scaling:** HPA on custom `requests_per_second` metric (not CPU). Redis feature store with 3-node cluster. Flink with unaligned checkpoints every 60s.

**Failures I dealt with:**
- Kafka consumer lag from aggressive Flink checkpointing. Fix: increased interval to 60s
- Redis connection pool exhaustion under load. Fix: `socket_timeout=0.5`, `max_connections=50`
- Model reload race condition during canary. Fix: readiness probe that waits for model load + `maxUnavailable: 0`

**Cost:** ~$4,150/month for 8K TPS (12 inference pods + Redis cluster + MSK)

**Trade-offs:** Chose synchronous over async for predictable latency. XGBoost on CPU over deep learning on GPU to save $3K/month.

---

## Q2: Design a model deployment platform for 50 data scientists

**Context:** Data scientists were deploying via `nohup python serve.py &`. No rollback, no monitoring.

**Architecture:**
```
Self-Service CLI (ml-deploy create/status/rollback)
  --> MLflow Registry (versioned artifacts on S3)
  --> Jenkins CI/CD (build image, validate, deploy)
  --> KServe on K8s (canary rollout, autoscaling)
  --> Datadog monitoring (infra + model + business metrics)
```

**Key decisions:**
- KServe over custom Flask: canary, autoscaling, GPU support out of the box
- Model baked into Docker image (not runtime S3 load): deterministic deployments, no S3 dependency at startup
- RBAC: data scientists get deploy/status/rollback only, not cluster-admin

**Common mistakes:** No resource limits (one model consumed 32GB/replica), skipping validation gate, giving cluster-admin to everyone.

---

## Q3: Design a feature store (batch + streaming)

**Context:** Needed historical aggregates (avg txn 30d, batch) and real-time signals (txn count 5min, streaming) in same feature vector.

**Architecture:**
```
Batch: S3 --> Spark (nightly via Airflow) --> Offline Store (S3 Parquet) --> Redis
Stream: Kafka --> Flink (continuous) --> Redis (with TTL) --> S3 (append)
Serving: Request --> Feast --> Redis --> Unified feature vector
Training: Feast offline retrieval --> Point-in-time join from S3
```

**The hard part:** Point-in-time correctness. Training data for March 15 must use features as of March 15, not today. Without this, you get data leakage.

**Failures:** Redis OOM from long TTLs. Fix: aggressive TTLs (10min for streaming, 25hr for batch). Flink crash causing stale features. Fix: `feature_freshness_seconds` metadata with fallback.

---

## Q4: Design a multi-tenant ML platform on Kubernetes

**Context:** Three teams sharing GPU cluster. Need data/compute/deployment isolation.

**Isolation:**
```
Per team: Namespace + ResourceQuota + NetworkPolicy + ServiceAccount + RBAC
Shared: MLflow, Feast, Prometheus in ml-shared namespace
GPU: Time-slicing for dev, dedicated for prod, MIG for training
Cost: Kubecost labels by namespace for chargeback
```

**Common mistakes:** No ResourceQuotas (one sweep consumes all GPUs), shared ServiceAccounts, over-isolating shared services.

---

## Q5: Design a multi-region, high-availability ML inference architecture for a global application.

**Context:** Our image classification API needed to serve users in the US, Europe, and Asia with < 200ms latency, surviving a complete AWS region failure.

**Architecture:**
```
Global Traffic Manager (Route 53 Latency-Based Routing)
  |--> US-East-1 Region
  |      |--> ALB --> EKS Cluster --> KServe Pods
  |      |--> Local ElastiCache (Redis) for features
  |--> EU-West-1 Region
  |      |--> ALB --> EKS Cluster --> KServe Pods
  |      |--> Local ElastiCache (Redis) for features
```

**The Challenge: Model and Data Synchronization**
You cannot copy a 5GB model to 3 regions over the internet every time a pod scales up.
1. **Model Distribution:** CI/CD pipeline pushes the Docker image to a global ECR registry (which replicates to all regions automatically). 
2. **Feature Synchronization:** We use DynamoDB Global Tables or Redis Enterprise Active-Active. If a user updates their profile in Europe, the feature store replicates that change to the US in <1 second so the model has the freshest data regardless of where the traffic routes.

**Failover Testing:**
We run chaos engineering tests ("Chaos Gorilla") monthly where we sever the network connection to `us-east-1`. Route 53 detects the health check failure and shifts traffic to `eu-west-1` within 60 seconds. The European EKS cluster HPA scales up the pods automatically to handle the 2x traffic load.

---

## Q6: Design an automated A/B testing pipeline for ML models.

**Context:** The marketing team wanted to test 5 different recommendation models concurrently, measuring actual CTR (Click-Through Rate) before declaring a winner.

**Architecture:**
```
Client Request --> API Gateway
  |--> A/B Routing Service (Redis backed)
        - Looks up user_id
        - Deterministically hashes user_id to select a model variant (A, B, C, D, E)
        - E.g., user_123 -> Model C
  |--> K8s Inference Cluster
        - Deployment A (Control)
        - Deployment B, C, D, E (Challengers)
```

**Tracking the winner:**
1. The Inference Service logs the prediction to Kafka, tagged with `variant: C`.
2. When the user eventually clicks a recommended item (or doesn't), the client app fires an event to Kafka.
3. A Spark Streaming job joins the "Prediction" stream with the "Click" stream on `request_id`.
4. It calculates the CTR for each variant and pushes it to a Grafana dashboard.

**Crucial detail:** The assignment of users to models MUST be deterministic. If a user gets Model A at 9:00 AM, they must get Model A at 9:05 AM. If you use round-robin load balancing, the user experience will be chaotic and the A/B test data will be statistically invalid. We use a hashing function: `hash(user_id + experiment_id) % 100`.

---
*[Back to README](../README.md)*
