# Section 12: Production Case Studies
## Real-World Architecture: Fraud Detection | Recommendation System | LLM Chatbot

---

## Case Study 1: Real-Time Fraud Detection System

### Business Context
- **Company type:** E-commerce platform, 50M active users
- **Scale:** 8,000 transactions/second peak, $2B GMV/month
- **SLA:** P99 latency < 80ms, availability 99.99%
- **Challenge:** Existing rule-based system had 40% false positive rate (blocking real users) and missing 15% of actual fraud

---

### Full Architecture Diagram

```
                        FRAUD DETECTION — PRODUCTION ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════

INGESTION LAYER
───────────────────────────────────────────────────────────────────────────────
 Mobile/Web App  ──HTTP──▶  API Gateway (Kong)  ──▶  Payment Service
                                                           │
                                              POST /transactions
                                                           │
                                                           ▼
                                              Fraud API (FastAPI K8s)
                                                           │
                                     ┌─────────────────────┼──────────────────┐
                                     ▼                     ▼                  ▼
TIER 1: RULES ENGINE          TIER 2: ML MODEL       TIER 3: HUMAN REVIEW
(Redis + Python, <2ms)    (XGBoost, 20-50ms)        (prob 0.6–0.85 → queue)
                                     │
                              Feature Fetch (Redis)
                              ┌─────────────────────┐
                              │ avg_spend_30d        │
                              │ txn_count_30d        │
                              │ failed_logins_5m     │
                              │ cross_border_flag    │
                              │ device_fingerprint   │
                              │ ... 145 more         │
                              └─────────────────────┘
                                     │
FEATURE STORE LAYER                  │
───────────────────────────────────────────────────────────────────────────────
                        ┌────────────┴────────────┐
                        ▼                         ▼
              ONLINE STORE (Redis)        OFFLINE STORE (Snowflake)
              - Real-time features        - Historical features
              - Flink streaming job       - Spark batch (nightly)
              - Sub-5ms reads             - Training data
              - 7-day TTL                 - Point-in-time correct

STREAMING LAYER
───────────────────────────────────────────────────────────────────────────────
Kafka (64 partitions, 3 replicas)
├── Topic: txn-raw          (raw transactions)
├── Topic: txn-enriched     (with online features)
├── Topic: fraud-decisions  (model outputs)
├── Topic: chargebacks      (ground truth, delayed)
└── Topic: audit-log        (all decisions for compliance)

TRAINING LAYER
───────────────────────────────────────────────────────────────────────────────
Airflow DAG (2 AM daily):
  validate_data → spark_features → train_xgboost → validate_model → deploy

MLflow: Experiment tracking + Model Registry
S3: Feature snapshots (DVC versioned)
SageMaker: Training compute (Spot instances, saves 75% cost)

MONITORING LAYER
───────────────────────────────────────────────────────────────────────────────
Prometheus → Grafana:
  - P99 latency, Error rate, Fallback rate
  - Prediction score distribution (drift proxy)
  - Feature null rates (upstream health)

Evidently AI (batch, hourly):
  - PSI for each feature (alert if PSI > 0.2)
  - Prediction drift

Datadog:
  - Business metrics: Fraud rate, False positive rate
  - Ground truth: Chargeback rate vs model score

═══════════════════════════════════════════════════════════════════════════════
```

### Tools Stack

| Layer | Tool | Why |
|---|---|---|
| Ingestion | Kafka (64 partitions) | Handles 8K TPS with horizontal scale |
| Stream processing | Apache Flink | Low-latency stateful stream ops |
| Online features | Redis Cluster | Sub-5ms P99 feature reads |
| Offline features | Snowflake + Spark | Point-in-time correct training sets |
| Feature store | Feast | Unified feature definitions |
| ML Framework | XGBoost | Fast inference (2ms), handles imbalance |
| Serving | FastAPI + KServe | Autoscale on K8s, scale-to-zero |
| Training | SageMaker Spot | 75% cheaper, checkpointed |
| Orchestration | Airflow | Daily retraining DAG |
| Tracking | MLflow | Model registry + experiment tracking |
| Monitoring | Evidently + Prometheus + Datadog | Three layers |
| GitOps | ArgoCD | Automated K8s deployments |
| Service mesh | Istio | mTLS, canary, shadow deployments |

### Key Architecture Decisions & Trade-offs

**Decision 1: Two-Tier Scoring**
```
Option A: ML model for all transactions
  Pros: More accurate
  Cons: Adds 20-50ms to every transaction, expensive at 8K TPS

Option B: Rules engine (Tier 1) + ML (Tier 2)
  Pros: Rules handle obvious cases in 2ms, ML only runs when needed
  Cons: Rules must be maintained, can become stale
  
CHOSEN: Option B. Rules handle 60% of cases in <2ms. ML runs on remaining 40%.
```

**Decision 2: Redis for Feature Store vs DynamoDB**
```
Redis:
  Pros: Sub-millisecond, supports complex data types, in-memory
  Cons: Expensive at scale, data loss risk without persistence
  
DynamoDB:
  Pros: Managed, cheap at scale, durable
  Cons: 5-15ms latency, single-key lookup only

CHOSEN: Redis. At 8K TPS × 50ms budget for features, Redis at <5ms was required.
DynamoDB would push us over SLA. Added Redis Sentinel for HA.
```

### Failure Scenarios & Mitigations

| Failure | Detection | Mitigation |
|---|---|---|
| Redis down | Feature null rate spikes | Circuit breaker → fallback features |
| Model degrades | Business metrics drop | Auto-rollback via MLflow registry |
| Kafka lag grows | Consumer lag > 10K | KEDA auto-scales inference pods |
| Feature pipeline fails | Feature freshness > 15min | Alert + fallback to 24h-old features |
| Training fails | Airflow DAG fails | Alert, keep current model in prod |
| Region outage | Route53 health check fails | Automatic failover to us-west-2 |

### Real Incident: The "Q4 Shopping Season Collapse"

**What happened:** November 25 (Black Friday). Transaction volume jumped from 8K to 24K TPS. Fraud API P99 latency hit 800ms. 15% of transactions were timing out. Fraud was getting through.

**Root causes (found in postmortem):**
1. Redis connection pool maxed out (only 100 connections configured, needed 300)
2. HPA scaling threshold was CPU-based, but ML workloads don't spike CPU proportionally
3. New model pods took 45 seconds to start (loading 500MB XGBoost model)

**Fixes applied during incident:**
```bash
# Emergency: Scale up immediately
kubectl scale deployment fraud-api --replicas=50 -n ml-serving

# Patch Redis connection pool
kubectl set env deployment/fraud-api REDIS_MAX_CONNECTIONS=500 -n ml-serving

# Emergency: Switch HPA to request-per-second metric
kubectl patch hpa fraud-api-hpa -n ml-serving -p '
{"spec": {"metrics": [{"type": "External", "external": {"metric":
{"name": "requests_per_second"}, "target": {"type": "AverageValue", "averageValue": "500"}}}]}}'
```

**Post-incident fixes:**
- HPA switched from CPU to RPS metric
- Model pre-loaded in InitContainer to reduce cold start from 45s to 5s
- KEDA configured to scale on Kafka lag BEFORE capacity is exhausted
- Load tested to 3x expected peak

---

## Case Study 2: Netflix-Scale Recommendation System

### Business Context
- **Scale:** 10M active users, 500K item catalog
- **Volume:** 2M recommendation requests/minute
- **SLA:** P99 < 100ms, fresh recommendations every 30 minutes

### Architecture — Two-Tower + ANN

```
TWO-TOWER RECOMMENDATION ARCHITECTURE
═══════════════════════════════════════════════════════════════════════

OFFLINE (runs continuously on Spark/Ray):
┌──────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│  User Events │    │  Two-Tower Model     │    │  Vector Database    │
│  (clicks,    │───▶│  (DLRM-style)       │───▶│  (Milvus)          │
│  purchases,  │    │  User Tower: 256-dim │    │  10M item vectors   │
│  dwell time) │    │  Item Tower: 256-dim │    │  HNSW index         │
└──────────────┘    └─────────────────────┘    └─────────────────────┘

ONLINE (per-user request, < 100ms total):

User logs in
     │
     ▼
1. Fetch user embedding from Redis      [~2ms]
     │
     ▼
2. ANN search in Milvus                 [~20ms]
   Returns top 500 candidate items
     │
     ▼
3. Ranking model scores 500 items       [~50ms]
   (heavy DLRM model on GPU)
     │
     ▼
4. Business rules filter                [~2ms]
   (remove out-of-stock, apply boost)
     │
     ▼
5. Return top 50 items to user          [~3ms network]

Total: ~77ms ✅ (under 100ms SLA)

CACHING LAYER:
- Pre-compute top-50 for daily-active users (Spark, nightly)
- Cache result in Redis with 30-min TTL
- 80% of requests served from cache at <5ms
```

### Training Pipeline

```python
# two_tower_model.py — Production two-tower recommendation model
import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(self, n_users: int, n_features: int, embedding_dim: int = 256):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, 128)
        self.feature_proj = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.output_proj = nn.Linear(128 + 256, embedding_dim)

    def forward(self, user_ids: torch.Tensor, user_features: torch.Tensor):
        user_emb = self.user_embedding(user_ids)
        feature_emb = self.feature_proj(user_features)
        combined = torch.cat([user_emb, feature_emb], dim=-1)
        # L2 normalize for cosine similarity ANN search
        return F.normalize(self.output_proj(combined), p=2, dim=-1)


class ItemTower(nn.Module):
    def __init__(self, n_items: int, n_features: int, embedding_dim: int = 256):
        super().__init__()
        self.item_embedding = nn.Embedding(n_items, 128)
        self.feature_proj = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.output_proj = nn.Linear(128 + 256, embedding_dim)

    def forward(self, item_ids: torch.Tensor, item_features: torch.Tensor):
        item_emb = self.item_embedding(item_ids)
        feature_emb = self.feature_proj(item_features)
        combined = torch.cat([item_emb, feature_emb], dim=-1)
        return F.normalize(self.output_proj(combined), p=2, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, n_users, n_items, user_feat_dim, item_feat_dim):
        super().__init__()
        self.user_tower = UserTower(n_users, user_feat_dim)
        self.item_tower = ItemTower(n_items, item_feat_dim)

    def forward(self, user_ids, user_features, item_ids, item_features):
        user_emb = self.user_tower(user_ids, user_features)
        item_emb = self.item_tower(item_ids, item_features)
        # Dot product similarity
        scores = (user_emb * item_emb).sum(dim=-1)
        return scores

    def get_user_embedding(self, user_ids, user_features):
        """Used at serving time to get user embedding for ANN search."""
        return self.user_tower(user_ids, user_features)

    def get_item_embedding(self, item_ids, item_features):
        """Run offline to index all items in Milvus."""
        return self.item_tower(item_ids, item_features)
```

---

## Case Study 3: Enterprise LLM Chatbot with RAG

### Business Context
- **Use case:** Internal HR chatbot (policies, benefits, PTO)
- **Users:** 15,000 employees
- **Documents:** 2,400 PDF/Word policy documents
- **Constraint:** All data must remain on-premise (HIPAA + SOC2)

### Architecture

```
ENTERPRISE LLM CHATBOT — ON-PREMISE ARCHITECTURE
═══════════════════════════════════════════════════════════════════════

DOCUMENT INGESTION (offline, weekly):
  SharePoint/Confluence → PDF Parser → Chunker (512 tokens, 64 overlap)
  → BGE-Large Embedder (self-hosted) → pgvector (PostgreSQL)

RUNTIME (per query):
Employee → Slack/Teams Bot → API Gateway (internal)
                                    │
                                    ▼
                          LangChain RAG Service (K8s)
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          pgvector ANN Search              Mistral-7B on vLLM
          (top-5 policy chunks)            (self-hosted, on-prem GPU)
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                          Answer + Source Citations

SAFETY LAYER (every response goes through):
  → Input guardrails: block PII, jailbreak attempts
  → Output guardrails: check for hallucinations vs retrieved context
  → Audit log: all Q&A pairs logged to Splunk (SOC2 compliance)

MONITORING:
  - LLM-as-judge quality scoring (hourly sample)
  - Retrieval hit rate: % of queries that found relevant docs
  - Employee satisfaction: thumbs up/down feedback
  - Cost tracking: tokens per query, cost per department
```

### Security Implementation

```python
# guardrails.py — Input/output safety for enterprise LLM
import re
from typing import Optional

class InputGuardrails:
    """Filter dangerous or policy-violating inputs."""

    # Patterns that indicate jailbreak attempts
    JAILBREAK_PATTERNS = [
        r"ignore (previous|all) instructions",
        r"you are now (DAN|JAILBREAK|unrestricted)",
        r"pretend (you are|to be) (a human|not an AI)",
        r"act as if you have no restrictions",
    ]

    # PII patterns (HIPAA compliance)
    PII_PATTERNS = {
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    }

    def check(self, text: str) -> tuple[bool, Optional[str]]:
        """Returns (is_safe, reason_if_not)."""
        text_lower = text.lower()

        # Check jailbreak
        for pattern in self.JAILBREAK_PATTERNS:
            if re.search(pattern, text_lower):
                return False, "potential_jailbreak"

        # Check PII in input (log and warn, don't block)
        for pii_type, pattern in self.PII_PATTERNS.items():
            if re.search(pattern, text):
                # Log for compliance, but still process (employee may need help)
                import logging
                logging.warning(f"PII detected in input: {pii_type}")

        # Check length
        if len(text) > 5000:
            return False, "input_too_long"

        return True, None


class OutputGuardrails:
    """Validate LLM output before returning to user."""

    def check(
        self,
        response: str,
        retrieved_context: str,
        question: str,
    ) -> tuple[str, dict]:
        """Validate response and add warnings if needed."""
        warnings = []

        # Check if response is very long (truncation might have occurred)
        if len(response) > 2000:
            warnings.append("response_truncated")

        # Check for common hallucination signals
        hallucination_signals = [
            "I don't have access to",
            "As of my knowledge cutoff",
            "I cannot verify",
        ]
        if any(signal in response for signal in hallucination_signals):
            warnings.append("possible_knowledge_limit")

        # Add disclaimer for sensitive topics
        sensitive_topics = ["disciplinary", "termination", "lawsuit", "legal"]
        if any(topic in question.lower() for topic in sensitive_topics):
            response += (
                "\n\n> ⚠️ **Note:** For matters involving employment actions or legal questions, "
                "please consult with HR directly or contact the legal team."
            )

        return response, {"warnings": warnings}
```

---

*Next → [Section 13: Interview Prep](../interview-prep/README.md)*
