# MLOps Quick Reference — Print & Keep
## One-Page Cheat Sheet for Production MLOps Engineers

---

## CORE CONCEPTS

| Concept | Definition | Production Signal |
|---|---|---|
| Training-Serving Skew | Features computed differently in training vs serving | Offline AUC >> Online AUC |
| Data Drift | Input feature distribution shifts | PSI > 0.2 |
| Concept Drift | X→y relationship changes | F1 drop after labels arrive |
| Model Staleness | Model not retrained recently | Alert if > 30 days old |
| PITC | Feature value at label time, not current | Enforced by feature store |
| Degenerate Feedback | Model outputs bias future training data | Exploration (ε-greedy) |
| Training-Serving Gap | Latency between training data cutoff and serving | Monitoring freshness |
| Class Imbalance | Extreme ratio (99:1 for fraud) | Use PR-AUC, not Accuracy |

---

## TOOL SELECTION MATRIX

### Serving
```
< 100 req/s + simple model → FastAPI (2 pods)
> 1K req/s + standard framework → KServe (Knative, scale-to-zero)
GPU + multiple models → Triton Inference Server
LLMs → vLLM (PagedAttention, 24x throughput vs naive)
```

### Orchestration
```
ETL + ML mixed team → Airflow (mature, flexible, non-K8s)
Pure ML on K8s → Kubeflow Pipelines (native K8s, lineage tracking)
Simple cron + Python → Prefect or Dagster (simpler than Airflow)
```

### Monitoring
```
Infra: Prometheus + Grafana (latency, error rate, GPU util)
Model: Evidently AI (drift, data quality, model performance)
Business: Datadog custom events (CTR, revenue, conversion)
Tracing: OpenTelemetry → Jaeger/Datadog APM
```

---

## KUBERNETES QUICK PATTERNS

### Resource Requests for ML
```yaml
resources:
  requests:
    cpu: "2"
    memory: "8Gi"
    nvidia.com/gpu: "1"    # Requires NVIDIA Device Plugin DaemonSet
  limits:
    memory: "16Gi"         # ALWAYS set memory limit (OOMKilled prevention)
    nvidia.com/gpu: "1"
    # Don't set CPU limit for inference (throttling causes latency spikes)
```

### HPA for ML Inference (use RPS, not CPU)
```yaml
metrics:
- type: External
  external:
    metric:
      name: requests_per_second
      selector:
        matchLabels: {deployment: fraud-api}
    target:
      type: AverageValue
      averageValue: "500"   # 500 RPS per pod
```

### KEDA for Batch (scale to zero)
```yaml
triggers:
- type: kafka
  metadata:
    topic: batch-scoring-requests
    consumerGroup: batch-scorer
    lagThreshold: "100"     # Scale when 100+ messages pending
    activationLagThreshold: "1"   # Wake from zero at 1 message
```

---

## DRIFT DETECTION FORMULAS

```python
# PSI (Population Stability Index)
def psi(expected, actual, buckets=10):
    expected_pct = np.histogram(expected, bins=buckets)[0] / len(expected)
    actual_pct   = np.histogram(actual,   bins=buckets)[0] / len(actual)
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / (expected_pct + 1e-8)))
    return psi
# PSI < 0.1: No significant change
# PSI 0.1-0.2: Moderate shift, investigate
# PSI > 0.2: Significant shift, retrain

# KL Divergence for prediction drift
kl_div = stats.entropy(current_preds_hist, baseline_preds_hist)

# Kolmogorov-Smirnov test (two-sample)
ks_stat, p_value = stats.ks_2samp(current_scores, baseline_scores)
# p_value < 0.01: distributions are significantly different
```

---

## COST REDUCTION QUICK WINS

| Action | Typical Savings | Effort |
|---|---|---|
| Training → Spot instances | 70-80% on training | Medium (add checkpointing) |
| Serving → Scale-to-zero (KEDA) | 40-60% on idle models | Low |
| S3 Lifecycle (IA after 30d, Glacier 90d) | 40-70% on storage | Low |
| Right-size with VPA recommendations | 20-40% on compute | Low |
| Reserved instances for baseline load | 40% vs On-Demand | Low |
| Mixed precision training (AMP) | 40% less GPU time | Low |
| Quantize inference model (INT8) | 2-4x faster, same GPU | Medium |

---

## SECURITY CHECKLIST

- [ ] No hardcoded secrets (use Vault + External Secrets Operator)
- [ ] OIDC for CI/CD → Cloud auth (no long-lived keys)
- [ ] Trivy scanning in CI (reject Critical/High CVEs)
- [ ] All pods run as non-root (Pod Security Admission: restricted)
- [ ] NetworkPolicy: default deny-all, explicit allows
- [ ] mTLS between services (Istio)
- [ ] PII scrubbed from logs before emission
- [ ] RBAC: DS team cannot touch prod namespace
- [ ] Model files in Safetensors, not pickle
- [ ] WAF in front of public inference endpoints

---

## INCIDENT RESPONSE PLAYBOOK

```
T+0:  Alert fires. Acknowledge within 5 minutes.
T+5:  Check: Is it infra? (CPU, memory, pod count) → NO
T+10: Check: Did model change? (MLflow registry) → NO
T+15: Check: Did data change? (feature null rates, PSI) → INVESTIGATE
T+20: Check: Did upstream pipeline fail? (Airflow DAG status)
T+30: Implement workaround (rollback model or fallback features)
T+60: Incident resolved. Write timeline in shared doc.
T+48h: Blameless postmortem with 5 action items.

Debug order:
  1. Feature null rates (upstream data quality)
  2. Model version in production (registry)
  3. Recent deployments (K8s rollout history)
  4. Upstream Kafka lag (data pipeline health)
  5. Redis feature store freshness
  6. External API dependencies
```

---

## A/B TEST SAMPLE SIZE FORMULA

```python
from scipy import stats
import numpy as np

def required_n(baseline_rate, min_effect=0.10, alpha=0.05, power=0.80):
    p1 = baseline_rate
    p2 = baseline_rate * (1 + min_effect)
    z_a = stats.norm.ppf(1 - alpha/2)
    z_b = stats.norm.ppf(power)
    p_avg = (p1 + p2) / 2
    n = ((z_a * np.sqrt(2*p_avg*(1-p_avg)) + z_b * np.sqrt(p1*(1-p1)+p2*(1-p2)))**2) / (p2-p1)**2
    return int(np.ceil(n))

# Fraud rate 6%, detect 10% relative improvement:
n = required_n(0.06, 0.10)  # → ~14,235 per variant
```

---

## INTERVIEW ANSWER FRAMEWORK: STARD

- **S**ituation — real scale (10K TPS, 50 models, 20 GPUs)
- **T**ool — what you chose AND what you rejected and why
- **A**rchitecture — the design with trade-offs
- **R**esult — measurable (AUC improved 4%, latency reduced 60%, cost saved $120K)
- **D**ebug — what went wrong, how you found it, what you fixed

---

*Print this page and keep it at your desk. Know every row of every table cold.*
