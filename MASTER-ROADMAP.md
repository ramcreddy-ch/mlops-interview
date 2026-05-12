# 🚀 THE COMPLETE MLOps PRODUCTION ROADMAP
## One-Stop Documentation | 12+ Years Production Experience | Print-Ready

> **Purpose:** Complete production MLOps reference — architecture decisions, real code, failure modes, interview answers  
> **Print Tip:** Print each section double-sided. Each README is self-contained.  
> **Navigation:** Use the table below to jump to any topic

---

## 📚 COMPLETE CONTENT DIRECTORY

### 🆕 New Sections (Complete Production Documentation)

| # | Section | Topics Covered | Size |
|---|---------|----------------|------|
| 1 | [MLOps Foundations](./mlops-foundations/README.md) | DevOps vs MLOps vs LLMOps, lifecycle stages, maturity levels, key terms | Deep |
| 2 | [Versioning & Data Management](./versioning/README.md) | DVC, LakeFS, Git branching, data contracts, large dataset handling | Deep |
| 3 | [Orchestration & Pipelines](./orchestration/README.md) | Airflow DAGs, Kubeflow Pipelines, SLAs, failure handling, retries | Deep |
| 4 | [Experiment Tracking](./experiment-tracking/README.md) | MLflow full setup, W&B, model registry lifecycle, champion-challenger | Deep |
| 5 | [Model Training at Scale](./model-training/README.md) | DDP distributed training, Ray Tune HPO, mixed precision, GPU cost guide | Deep |
| 6 | [Model Serving](./model-serving/README.md) | FastAPI production server, KServe, Triton, ONNX, REST vs gRPC | Deep |
| 10 | [LLMOps](./llmops/README.md) | vLLM on K8s, RAG pipelines, prompt versioning, cost optimization | Deep |
| 12 | [Case Studies](./case-studies/README.md) | Fraud Detection, Recommendation System, Enterprise LLM Chatbot | Deep |
| 13 | [Interview Prep](./interview-prep/README.md) | Scenario Q&A, system design, incident debugging, cost reduction | Deep |
| 14 | [Advanced Topics](./advanced-topics/README.md) | Feast feature store, canary deployments, A/B testing, MAB | Deep |
| 15 | [Learning Roadmap](./learning-roadmap/README.md) | 24-month month-by-month plan, hands-on projects, skills checklist | Deep |

### 📁 Existing Sections (Production Q&A — 30 Questions Each)

| # | Section | Topics Covered |
|---|---------|----------------|
| — | [System Design](./system-design/README.md) | Real-time inference, multi-region HA, A/B testing, shadow deploy, fraud & reco |
| — | [Fundamentals](./fundamentals/README.md) | ML lifecycle, drift types, feature stores, batch vs real-time, class imbalance |
| — | [Kubernetes & Platform](./kubernetes/README.md) | GPU limits, HPA, gang scheduling, KEDA, Karpenter, cold starts |
| — | [Cloud MLOps](./cloud-ml/README.md) | SageMaker vs EKS, Vertex AI, spot instances, cross-account, cost |
| — | [Monitoring & Observability](./monitoring/README.md) | Three layers, drift detection, Prometheus, SLIs/SLOs, golden signals |
| — | [Security & Compliance](./security/README.md) | Vault, PII, RBAC, supply chain, adversarial attacks, GDPR |
| — | [Real-Time Scenarios](./real-time-scenarios/README.md) | Kafka lag debugging, feature store stampedes, false alarms |
| — | [GPU Engineering](./gpu-engineering/README.md) | CUDA OOM, MIG vs time-slicing, NCCL bottlenecks, TensorRT |
| — | [Python for MLOps](./python-mlops/README.md) | GIL bypass, memory leaks, fast DataLoaders, pickle risks |
| — | [CI/CD for ML](./ci-cd/README.md) | ML vs software CI/CD, model versioning, GitOps rollback |
| — | [Cost Optimization](./cost-optimization/README.md) | Spot instances, scale-to-zero, cost attribution, FinOps |
| — | [Behavioral & Leadership](./behavioral/README.md) | Incident leadership, adoption strategy, exec communication |

---

## 🗺️ READING PATHS — Choose Your Journey

### Path A: "I have an interview next week"
```
Day 1: MLOps Foundations → System Design (Q1-Q5)
Day 2: Fundamentals (Q1-Q10) → Interview Prep (Section A + B)
Day 3: Monitoring → Kubernetes → Interview Prep (Section C + D)
Day 4: Case Studies (Fraud + Reco) → Interview Prep (Section E)
Day 5: Mock answers out loud using STARD framework
```

### Path B: "I'm building my first production ML system"
```
Month 1: Foundations → Versioning → Experiment Tracking
Month 2: Orchestration → Model Serving → Monitoring
Month 3: Kubernetes → CI/CD → Case Studies
Month 4: Advanced Topics → Security → Cost Optimization
```

### Path C: "I'm a DS transitioning into MLOps"
```
Start:  Learning Roadmap (read the full 24-month plan)
Then:   MLOps Foundations (understand what you're signing up for)
Then:   Fundamentals (bridge your DS knowledge to MLOps)
Then:   Follow Phase 1 of the Learning Roadmap month by month
```

### Path D: "I want to understand LLMs in production"
```
LLMOps → Model Serving → Kubernetes → Case Studies (LLM Chatbot)
```

---

## ⚡ THE 10 THINGS EVERY MLOPS ENGINEER MUST KNOW COLD

These come up in **every** senior MLOps interview:

### 1. Training-Serving Skew
```
What: Feature computation is DIFFERENT between training (Spark) and serving (Python)
Why: Silent accuracy killer. Can drop AUC by 5-15% with no infra alerts.
Fix: Feature Store (Feast) with single definition. Log+compare features in production.
```

### 2. The Three Drift Types
```
Data Drift:    Input distribution shifts. Detect: PSI > 0.2. Fix: Retrain.
Concept Drift: X→y relationship changes. Detect: Ground truth labels. Fix: Retrain + reweight.
Model Drift:   Model output distribution shifts. Detect: KL divergence. Proxy for accuracy.
```

### 3. Point-in-Time Correctness
```
Problem: Training on Nov 15 event? Feature values must reflect Nov 14, not today.
Fix: Feast/Tecton enforces PITC during get_historical_features().
Without it: Data leakage. Model learns future. Perfect offline AUC, random production AUC.
```

### 4. HPA Anti-Pattern for ML
```
CPU-based HPA FAILS for ML: XGBoost/PyTorch designed to max out CPU.
Fix: HPA on requests_per_second or concurrency (custom metric via Prometheus adapter).
```

### 5. The Model Registry State Machine
```
None → Staging → Production → Archived
Key rule: CD pipeline ONLY triggers on registry state transitions, never on raw Git commits.
This prevents un-validated models from reaching production.
```

### 6. Spot Instance Strategy for Training
```
Always use Spot for training (70-80% cost reduction).
Requirements: Checkpointing every N epochs. Retry logic in orchestrator.
AWS interruption warning: 2 minutes. Use Node Termination Handler.
```

### 7. Kafka Consumer Lag ≠ Consumer Dead
```
Temporary lag spike during traffic burst = normal.
Continuously GROWING lag = consumer is dead or too slow.
Alert on: rate of change (derivative), not absolute lag value.
```

### 8. OOMKilled Exit Code
```
Exit code 137 = OOMKilled = pod exceeded limits.memory
Common causes: Large batch size × model size, Gunicorn N workers × model size
Fix: Reduce workers, or increase limit, or switch to pre-fork loading pattern.
```

### 9. Shadow vs Canary vs A/B
```
Shadow:  100% traffic mirrored, user sees OLD model only. Zero risk. No business metric.
Canary:  X% traffic to new model, user sees real response. Low risk. Limited business signal.
A/B:     50/50 split (or other ratio). Both groups real users. Full business metric. Takes time.
Order: Shadow → Canary → A/B → Full rollout.
```

### 10. Cost Per Inference Calculation
```
GPU cost = (instance_cost_per_hour) / (requests_per_hour)
           E.g.: $3.06/hr ÷ 36,000 req/hr = $0.000085 per request
Add: feature store fetch, logging, monitoring overhead
Alert if cost per request increases >20% (indicates regression or traffic pattern change)
```

---

## 🔧 QUICK COMMAND REFERENCE

### DVC
```bash
dvc init && dvc remote add -d s3remote s3://bucket/path
dvc add data/large_file.csv && dvc push
dvc repro && dvc metrics diff HEAD~1
dvc dag  # Visualize pipeline
```

### MLflow
```bash
mlflow server --backend-store-uri postgresql://... --default-artifact-root s3://...
mlflow models serve -m "models:/fraud-model/Production" -p 5001
mlflow run . -P n_estimators=300
```

### Airflow
```bash
airflow db init && airflow webserver -p 8080 & airflow scheduler &
airflow dags trigger fraud_model_training_pipeline
airflow tasks test fraud_pipeline validate_data 2024-11-15
```

### Kubernetes
```bash
kubectl get pods -n ml-serving -w
kubectl describe pod <pod-name> -n ml-serving   # Debug pod issues
kubectl top pods -n ml-serving                   # Resource usage
kubectl logs <pod> -n ml-serving --previous      # Logs from crashed pod
kubectl exec -it <pod> -n ml-serving -- /bin/bash
```

### KServe
```bash
kubectl apply -f inference-service.yaml
kubectl get inferenceservice -n ml-serving
kubectl get pods -n ml-serving -l serving.kserve.io/inferenceservice=fraud-model
```

### vLLM
```bash
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --quantization awq
```

### Feast
```bash
feast init my_feature_repo && cd my_feature_repo
feast apply                                 # Apply feature definitions
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
feast serve                                 # Start online feature server
```

---

## 📊 ARCHITECTURE DECISION QUICK-GUIDE

| Decision | Option A | Option B | Choose A when | Choose B when |
|---|---|---|---|---|
| Serving | FastAPI custom | KServe | Full control needed | Standard framework, scale-to-zero |
| Orchestration | Airflow | Kubeflow | Mixed ETL+ML team | Pure ML, K8s-native team |
| Tracking | MLflow | W&B | Self-hosted, privacy | DL teams, visual comparison |
| Feature store | Feast (self-managed) | Vertex/SageMaker FS | Multi-cloud, cost control | Fully managed, vendor OK |
| Batching | Airflow+Spark | SageMaker Batch | On-prem/multi-cloud | AWS only, managed |
| GPU sharing | MIG | Time-slicing | Production, isolation | Dev, flexible |
| Model serving GPU | Triton | vLLM | Traditional DL | LLMs only |
| Auto-scaling | HPA | KEDA | Stateless, CPU/RPS metric | Queue-based, scale-to-zero |
| Deployment | Rolling | Canary+Flagger | Low risk change | New model, ML risk |

---

*Repository created for production MLOps mastery. All code examples are production-tested patterns.*  
*Total documentation: ~15,000 lines of code examples and production guidance.*
