# Section 15: Learning Roadmap
## 24-Month Plan to 12+ Years Equivalent MLOps Mastery

---

## The Philosophy

You cannot shortcut 12 years of experience. But you CAN compress the **learning** into 24 months by:
1. Building real projects (not tutorials)
2. Learning failure modes before you encounter them in prod
3. Going deep on one thing at a time, not shallow on everything

The roadmap below builds skills in dependency order. Each phase unlocks the next.

---

## Phase 1: Foundations (Months 1-3)
### "Make something work end-to-end before optimizing anything"

### Month 1 — ML & Python Foundations

**Goal:** Be able to train a model, evaluate it correctly, and understand WHY it works.

**Week 1-2: Python for production ML**
```
Topics:
  - Python virtual environments (venv, conda, poetry)
  - Type hints, dataclasses, pydantic
  - asyncio basics (for serving)
  - Profiling: cProfile, line_profiler
  - Memory: tracemalloc, objgraph

Hands-on:
  Build a FastAPI endpoint that serves predictions from a pre-trained model.
  Profile the endpoint to find the bottleneck.
```

**Week 3-4: ML fundamentals that matter in production**
```
Topics:
  - Classification metrics: why accuracy is wrong for fraud/medical
  - Precision-Recall tradeoff: set threshold based on business cost
  - Cross-validation: why time-series is different
  - Feature engineering: creating lag features, rolling windows
  - Class imbalance: SMOTE pitfalls, scale_pos_weight

Hands-on:
  Train an XGBoost fraud classifier on the Kaggle Credit Card Fraud dataset.
  Beat the naive accuracy by using PR-AUC correctly.
  Write a CI test that fails if PR-AUC drops below 0.70.
```

**Resources:**
- Hands-On Machine Learning (Aurélien Géron) — Ch 1-7
- FastAPI docs: https://fastapi.tiangolo.com
- Kaggle Credit Card Fraud dataset

---

### Month 2 — Experiment Tracking + Versioning

**Goal:** Every experiment is reproducible. Every model version is traceable.

**Week 1-2: MLflow**
```
Hands-on Project:
  Build a complete experiment tracking system for the fraud model:
  - Log all hyperparameters
  - Log metrics per fold (cross-validation)
  - Log SHAP plots as artifacts
  - Set up Model Registry with Staging → Production workflow
  - Build a script that compares last 10 runs and picks the best

Deliverable: Running MLflow server (Docker Compose) + 20+ logged experiments
```

**Week 3-4: DVC**
```
Hands-on Project:
  Take the fraud project and add full DVC pipeline:
  - dvc init, remote setup (S3 or local)
  - dvc.yaml pipeline: prepare → featurize → train → evaluate
  - params.yaml for all hyperparameters
  - dvc repro runs correctly from a fresh clone
  - dvc metrics diff shows metric changes between branches

Deliverable: GitHub repo where ANY commit is fully reproducible
```

**Resources:**
- MLflow docs: mlflow.org
- DVC docs: dvc.org/doc
- "Made With ML" course (free): madewithml.com

---

### Month 3 — Containers + Kubernetes Basics

**Goal:** Package your model and run it reliably on Kubernetes.

**Week 1: Docker for ML**
```
Topics:
  - Multi-stage Dockerfiles (builder → runtime, save 80% image size)
  - Layer caching optimization (dependencies first, code second)
  - Non-root users in containers (security requirement)
  - .dockerignore (exclude data, .git, __pycache__)

Hands-on:
  Build a production Docker image for the fraud model:
  - < 500MB total (use python:3.11-slim base)
  - Non-root user
  - Health check endpoint baked in
  - Push to Docker Hub or GitHub Container Registry
```

**Week 2-4: Kubernetes fundamentals**
```
Topics:
  - Core objects: Pod, Deployment, Service, ConfigMap, Secret
  - Namespaces, resource requests/limits
  - Readiness vs Liveness probes (critical for ML cold starts)
  - HPA setup (CPU first, then custom metrics)
  - Basic kubectl commands

Hands-on (use minikube or kind locally):
  Deploy the fraud model to local K8s:
  - Deployment with 2 replicas
  - Service (ClusterIP)
  - ConfigMap for non-secret config
  - Secret for API keys
  - HPA targeting 70% CPU
  - Readiness probe with 30s initial delay (model load time)

Deliverable: fraud-model running on local K8s, accessible via kubectl port-forward
```

**Resources:**
- "Kubernetes in Action" (Marko Luksa) — Ch 1-9
- Play with Kubernetes: labs.play-with-k8s.com
- minikube: minikube.sigs.k8s.io

---

## Phase 2: Core MLOps (Months 4-8)
### "Build what production teams actually run"

### Month 4 — Orchestration (Airflow)

**Goal:** Build a fully automated, fault-tolerant ML training pipeline.

```
Hands-on Project: Fraud Model Retraining Pipeline

DAG structure:
  check_data_freshness
    → validate_data (Great Expectations)
    → compute_features (PySpark local mode)
    → train_model (with MLflow logging)
    → quality_gate (fail if AUC-PR < 0.70)
    → register_model (MLflow staging)
    → notify_slack

Requirements:
  - All tasks have 2 retries with exponential backoff
  - SLA callback sends Slack alert if DAG exceeds 2 hours
  - All tasks are IDEMPOTENT (safe to retry)
  - Schedule: 2 AM daily
  - Max active runs: 1

Run Airflow with Docker Compose (official image).
Trigger the DAG manually and watch it succeed.
Introduce a data validation failure and confirm the alert fires.

Deliverable: Working Airflow DAG running in Docker Compose
```

---

### Month 5 — Monitoring + Observability

**Goal:** Know when your model degrades BEFORE the business complains.

```
Hands-on Project: Complete Monitoring Stack for Fraud Model

Components to build:
  1. Prometheus metrics in FastAPI:
     - prediction_score histogram
     - feature_null_rate gauge
     - fallback_triggered counter
     - inference_latency histogram
  
  2. Grafana dashboards:
     - Latency: P50, P95, P99 over time
     - Prediction distribution (histogram) vs baseline
     - Feature null rates per feature
     - Error rate (5xx)
  
  3. Drift detection (Evidently):
     - Daily batch job: compare last 24h predictions vs training baseline
     - PSI for top 10 features
     - Output: HTML report + metrics to Prometheus
  
  4. Alert rules:
     - P99 > 200ms for 5 minutes → page
     - Error rate > 1% → page
     - PSI > 0.2 for any feature → Slack warning
     - Model not retrained in 30 days → Slack warning

Run using Docker Compose (Prometheus + Grafana + FastAPI + Evidently).
Generate artificial drift by sending modified requests.
Confirm alerts fire correctly.

Deliverable: Dashboard showing live model health
```

---

### Month 6 — Cloud MLOps (AWS SageMaker)

**Goal:** Run the full ML lifecycle on a managed cloud platform.

```
Hands-on Project: Fraud Pipeline on AWS (use free tier where possible)

1. SageMaker Training Job:
   - Package training code as SageMaker estimator
   - Run on ml.m5.xlarge (cheapest GPU-optional)
   - Log to MLflow (hosted locally or on SageMaker)
   - Output model to S3

2. SageMaker Real-Time Endpoint:
   - Deploy model to ml.c5.xlarge endpoint
   - Configure auto-scaling (min 1, max 5)
   - Enable Data Capture (for drift monitoring)

3. SageMaker Model Monitor:
   - Create baseline from training data
   - Schedule hourly monitoring job
   - View drift report in SageMaker Studio

4. GitHub Actions CI/CD:
   - Push to main → trigger SageMaker training job
   - If AUC-PR > threshold → deploy to endpoint
   - Uses OIDC authentication (no hardcoded keys)

Budget: < $50/month (use small instances, delete endpoints when not testing)

Deliverable: Fully automated ML pipeline on AWS
```

---

### Month 7 — Feature Store (Feast)

**Goal:** Eliminate training-serving skew with a proper feature store.

```
Hands-on Project: Feast Feature Store for Fraud Model

1. Feast setup:
   - feast init fraud_feature_repo
   - Define User entity
   - Define 3 FeatureViews (30d aggregates, 5m streaming, device features)
   - Configure offline store: local Parquet (or Redshift if on AWS)
   - Configure online store: Redis

2. Feature materialization:
   - Batch materialize: feast materialize for last 90 days
   - Write streaming features with a simple script (simulate Flink)

3. Training with PITC:
   - Generate training data using store.get_historical_features()
   - Confirm features are point-in-time correct

4. Serving:
   - Update FastAPI to fetch features via store.get_online_features()
   - Compare latency vs direct Redis access

5. Skew test:
   - Log serving features and offline features for same user
   - Confirm they match (zero skew)

Deliverable: Feature store with zero training-serving skew
```

---

### Month 8 — CI/CD for ML (GitOps + ArgoCD)

**Goal:** Model deployments are triggered by Git commits, never by SSH.

```
Hands-on Project: Full GitOps ML Deployment Pipeline

1. GitHub Actions CI:
   - On push to main:
     a. Run unit tests (pytest)
     b. Run integration tests (test feature store, test model endpoint)
     c. Build Docker image + push to ECR
     d. DVC repro check (pipeline unchanged)
     e. Trigger training if data changed (dvc status)

2. MLflow as gating mechanism:
   - Training job registers model to Staging
   - Automated test suite runs against Staging model
   - If all pass: model transitions to Production

3. ArgoCD (GitOps):
   - ArgoCD watches a "deploy" Git repo
   - CI commits updated image tag to deploy repo
   - ArgoCD auto-syncs K8s cluster with deploy repo state
   - Rollback = git revert + auto-sync

4. Canary with Flagger (bonus):
   - Configure Flagger to automatically promote if latency + error rate OK
   - Verify rollback triggers on simulated error

Deliverable: Zero-touch deployment pipeline — code push to prod model serving in < 30 minutes
```

---

## Phase 3: Advanced Topics (Months 9-16)

### Month 9-10 — GPU Engineering + Distributed Training

```
Hands-on Projects:

1. Distributed XGBoost training with Dask (single machine multi-core)
2. PyTorch DDP on 2 GPUs (use cloud GPU instance: g4dn.12xlarge)
3. Implement mixed precision training (AMP)
4. Export to ONNX + benchmark vs PyTorch inference (measure speedup)
5. INT8 quantization — measure accuracy loss vs latency gain

Resources:
- PyTorch DDP tutorial: pytorch.org/tutorials
- "Efficient Deep Learning" book
- NVIDIA DLI course: courses.nvidia.com
```

### Month 11-12 — LLMOps

```
Hands-on Projects:

1. Deploy Mistral-7B on local GPU with vLLM
   - Benchmark: tokens/second, latency vs throughput
   - Test different quantization levels (FP16, AWQ, GGUF)

2. Build RAG pipeline:
   - Index 100 PDF documents with pgvector
   - Query interface with LangChain
   - Add LLM-as-judge evaluation

3. Prompt versioning:
   - 5 prompts in Git (YAML format)
   - CI: automated testing of prompt outputs
   - A/B test two prompt versions

4. Cost optimization:
   - Implement model routing (cheap model for simple queries)
   - Add semantic caching (Redis)
   - Measure token cost before/after

Resources:
- vLLM docs: docs.vllm.ai
- LangChain: python.langchain.com
- "Building LLMs for Production" (Chip Huyen's blog posts)
```

### Month 13-14 — Multi-Cloud + Platform Engineering

```
Hands-on Projects:

1. Deploy same ML pipeline to both AWS (EKS) and GCP (GKE)
   - Terraform modules for each cloud
   - Abstract storage (S3 vs GCS) via abstraction layer
   - Compare costs at 1K TPS

2. Build a self-service ML platform:
   - Helm chart that data scientists fill in (model name, docker image, resources)
   - Automated namespace + RBAC creation for new teams
   - MLflow + Airflow shared infrastructure

3. Multi-tenant K8s:
   - Implement ResourceQuotas, NetworkPolicies, RBAC
   - Kubecost chargeback per team
   - Karpenter for efficient GPU node provisioning
```

### Month 15-16 — Security + Compliance

```
Hands-on Projects:

1. Secrets management:
   - HashiCorp Vault in K8s (Vault Agent Injector)
   - External Secrets Operator + AWS Secrets Manager
   - Rotate a secret and confirm no downtime

2. Supply chain security:
   - Integrate Trivy into CI pipeline
   - Sign Docker images with Cosign (Sigstore)
   - Set up JFrog Artifactory as PyPI proxy

3. Data privacy:
   - Implement field-level encryption for PII in feature store
   - Build PII scrubber for logging pipeline (regex + model-based)
   - Document GDPR data flow for a model
```

---

## Phase 4: Leadership & Mastery (Months 17-24)

### Month 17-18 — Incident Response + SRE for ML

```
Study:
  - "Site Reliability Engineering" (Google SRE Book) — free at sre.google
  - Blameless postmortems: how to write them, lead them

Practice:
  - Chaos engineering: use Chaos Monkey or Chaos Toolkit
  - Inject failures into your staging environment:
    * Redis down (confirm fallback features work)
    * Model loading fails (confirm readiness probe prevents traffic)
    * Kafka lag spike (confirm KEDA scales up)
  - Write a postmortem for each simulated incident

Deliverable: 3 written postmortems + runbook for each failure mode
```

### Month 19-20 — Cost Engineering + FinOps

```
Hands-on:
  - Deploy Kubecost on your K8s cluster
  - Create per-team cost dashboards
  - Implement Spot instance strategy for training (80% cost reduction)
  - Calculate cost per inference request (GPU time + memory + networking)
  - Build alerting for cost anomalies (>200% spend increase in 1 hour)

Deliverable: Dashboard showing cost per model, per team, per request
```

### Month 21-22 — Open Source Contribution

```
This is the best way to level up and get visibility:

1. Find a bug or missing feature in:
   - MLflow, DVC, Feast, KServe, or Evidently
   
2. Submit a PR with:
   - Tests
   - Documentation update
   - Changelog entry

3. Write a blog post about what you built/fixed
   - LinkedIn + Medium/Substack
   - Real engineers read these and it directly leads to job opportunities

Even a small documentation fix counts. The goal is to understand production
codebases and get comfortable contributing to large Python projects.
```

### Month 23-24 — Interview Preparation + Portfolio

```
Technical portfolio (GitHub):
  1. fraud-detection-mlops: Complete fraud system (all 15 sections implemented)
  2. llm-rag-production: Enterprise RAG with all safety features
  3. ml-platform-k8s: Multi-tenant ML platform with Terraform + K8s

Write about each project:
  - README explaining architecture decisions
  - Blog post: "How I built X and what I learned"
  - Cost estimates, latency benchmarks, lessons learned

Interview prep:
  - Practice all questions in Section 13 out loud
  - Do 3-4 mock interviews (pramp.com, interviewing.io)
  - For each answer, ensure you have a REAL example (even from projects)
  - Read 1 paper per week from MLSys (mlsys.org)

Target roles at this stage:
  - Senior MLOps Engineer
  - ML Platform Engineer
  - Staff ML Engineer
  Expected salary: $180K-$280K depending on company/location (2025)
```

---

## Skills Checklist — Am I Interview-Ready?

Use this before applying to senior roles:

### Core MLOps (Must have ALL)
- [ ] Built and operated a training pipeline with automated retraining
- [ ] Implemented data drift monitoring that caught a real (or simulated) issue
- [ ] Can explain training-serving skew from a personal experience
- [ ] Deployed a model with canary deployment and rollback tested
- [ ] Set up CI/CD pipeline where model goes from code push to serving in < 1 hour
- [ ] Designed a Feature Store and can explain offline vs online with real trade-offs
- [ ] Handled a production incident and written a postmortem

### Kubernetes (Must have ALL)
- [ ] Can write a Deployment YAML from scratch with proper probes, limits, labels
- [ ] Can debug OOMKilled, CrashLoopBackOff, Pending (GPU quota)
- [ ] Can configure HPA based on custom metrics (not CPU)
- [ ] Implemented KEDA for scale-to-zero on a batch workload
- [ ] Understand Gang Scheduling vs standard scheduling for distributed training

### Cloud (Need 1 deeply, 2 broadly)
- [ ] SageMaker: full end-to-end pipeline (training, serving, monitoring)
- [ ] Vertex AI: training job + endpoint + feature store comparison
- [ ] Azure ML: pipeline + model registry + deployment
- [ ] Cost optimization: Spot instances, reserved, right-sizing

### Advanced (Good to have)
- [ ] LLMOps: deployed vLLM, built RAG pipeline, tracked cost per query
- [ ] Distributed training: run DDP on 2+ GPUs, understand NCCL
- [ ] Security: Vault secrets injection, mTLS between pods, supply chain security
- [ ] A/B testing: set up experiment with proper statistical analysis

---

## The 3 Things That Separate Senior from Staff MLOps Engineers

**1. You think in systems, not tools.**
   - Junior: "I use MLflow for tracking"
   - Senior: "I designed a model registry workflow that eliminated unauthorized deployments"

**2. You have failure mode intuition.**
   - Junior: "I set up monitoring"
   - Staff: "The first thing I check when a model degrades is feature null rates, then prediction distribution, then look at recent data pipeline changes"

**3. You quantify trade-offs.**
   - Junior: "We should use Redis for the feature store"
   - Staff: "Redis gives us 2ms P99, DynamoDB gives us 15ms. At 10K TPS with an 80ms budget, the 13ms difference is the margin between meeting SLA and missing it. The $40K/year Redis cost is justified."

---

*This roadmap is complete. You now have the equivalent of 12+ years of production MLOps knowledge structured and accessible for deep reading.*

*Go build something.*
