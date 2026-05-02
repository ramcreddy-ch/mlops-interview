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

*[Back to README](../README.md)*
