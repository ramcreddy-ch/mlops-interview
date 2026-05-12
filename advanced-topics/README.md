# Section 14: Advanced Topics
## Feature Stores | Canary Deployments | A/B Testing | Online vs Offline Features

---

## 14.1 Feature Stores — Production Deep Dive with Feast

### What a Feature Store Actually Solves

Without a feature store, you have this mess:
```
Team A computes "user_age" as: current_date - birth_date (in years, rounded down)
Team B computes "user_age" as: (current_date - birth_date).days / 365.25
Team C doesn't compute it — they use age from the profile (which might be fake)

Result: Three models use "the same feature" but with different values.
        Debugging a production issue becomes a nightmare.
```

With Feast:
```python
# Single source of truth — one definition, used by ALL teams
from feast import Feature, FeatureView, Entity, ValueType

user = Entity(name="user_id", value_type=ValueType.STRING)

user_features = FeatureView(
    name="user_fraud_features",
    entities=["user_id"],
    ttl=timedelta(days=7),   # Features expire after 7 days (forces freshness)
    features=[
        Feature(name="avg_spend_30d", dtype=ValueType.FLOAT),
        Feature(name="txn_count_30d", dtype=ValueType.INT64),
        Feature(name="max_spend_30d", dtype=ValueType.FLOAT),
        Feature(name="failed_logins_5m", dtype=ValueType.FLOAT),
        Feature(name="cross_border_flag", dtype=ValueType.BOOL),
        Feature(name="device_count_7d", dtype=ValueType.INT64),
    ],
    online=True,        # Available in Redis for real-time serving
    # Batch source: where to read historical data for training
    batch_source=BigQuerySource(
        table_ref="company.features.user_fraud_features",
        event_timestamp_column="feature_timestamp",
        created_timestamp_column="created_timestamp",
    ),
)
```

### Feast Materialization — Syncing Offline to Online

```bash
# Materialize: reads from BigQuery/Snowflake, writes to Redis
# Run daily via Airflow

feast materialize \
    2024-10-01T00:00:00 \
    2024-11-01T00:00:00   # Materialize last month's worth of features

# For incremental materialization (faster, just new data):
feast materialize-incremental 2024-11-01T00:00:00
```

### Training with Point-in-Time Correct Features

```python
# training_dataset.py — CRITICAL: No data leakage via Feast PITC
from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path="feast_repo/")

# Historical events: each row has (user_id, event_timestamp, is_fraud)
fraud_events = pd.read_parquet("s3://data/fraud_labels.parquet")

# Feast generates the EXACT feature values as they existed at event time
# If fraud happened on Nov 5 at 2pm, Feast uses feature values from Nov 5 at 1:59pm
# NOT the feature value computed later (which would be data leakage)
training_df = store.get_historical_features(
    entity_df=fraud_events,
    features=[
        "user_fraud_features:avg_spend_30d",
        "user_fraud_features:txn_count_30d",
        "user_fraud_features:failed_logins_5m",
        "user_fraud_features:cross_border_flag",
    ],
).to_df()

# training_df now has EXACTLY the right feature values at prediction time
# This is why you don't need to manually join and filter by timestamp
```

### Serving with Online Features (Sub-5ms)

```python
# serve.py — Online feature retrieval at inference time
from feast import FeatureStore

store = FeatureStore(repo_path="feast_repo/")

def get_features_for_user(user_id: str) -> dict:
    """Fetch pre-computed features from Redis via Feast."""
    feature_vector = store.get_online_features(
        features=[
            "user_fraud_features:avg_spend_30d",
            "user_fraud_features:txn_count_30d",
            "user_fraud_features:failed_logins_5m",
            "user_fraud_features:cross_border_flag",
        ],
        entity_rows=[{"user_id": user_id}],
    ).to_dict()

    return {k: v[0] for k, v in feature_vector.items()}

# Usage in FastAPI handler:
features = get_features_for_user("user_12345")
# Returns: {"avg_spend_30d": 145.23, "txn_count_30d": 28, ...}
# Latency: 2-5ms from Redis
```

---

## 14.2 Online vs Offline Feature Engineering — Key Differences

```
┌──────────────────────────────────────────────────────────────────────┐
│                FEATURE CLASSIFICATION GUIDE                           │
├─────────────────────────┬────────────────────────────────────────────┤
│  OFFLINE-ONLY FEATURES  │  ONLINE + OFFLINE FEATURES                 │
│  (Training only)        │  (Must exist at serving time)              │
├─────────────────────────┼────────────────────────────────────────────┤
│ Historical aggregates   │ Real-time aggregates (<5 min window)       │
│ computed over years     │ User's current session features            │
│                         │ Device/IP reputation scores                 │
│ Examples:               │                                            │
│ - "avg_spend_5yr"       │ Examples:                                  │
│ - "lifetime_value"      │ - "failed_logins_last_5m"                  │
│ - "account_age_days"    │ - "current_session_txn_count"              │
│ - "CLTV_percentile"     │ - "ip_reputation_score" (external API)     │
│                         │ - "avg_spend_30d" (materialized nightly)   │
├─────────────────────────┼────────────────────────────────────────────┤
│ Compute: Spark batch    │ Compute: Flink streaming jobs              │
│ Store: S3/Snowflake     │ Store: Redis (online) + S3 (offline)       │
│ Freshness: Daily        │ Freshness: Real-time (seconds)             │
└─────────────────────────┴────────────────────────────────────────────┘
```

### Streaming Feature Computation with Flink:

```python
# flink_streaming_features.py — Real-time feature computation
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment
from pyflink.table.descriptors import Schema, Kafka, Json

env = StreamExecutionEnvironment.get_execution_environment()
env.set_parallelism(32)
t_env = StreamTableEnvironment.create(env)

# Read from Kafka
t_env.execute_sql("""
    CREATE TABLE login_events (
        user_id STRING,
        event_type STRING,
        event_timestamp TIMESTAMP(3),
        WATERMARK FOR event_timestamp AS event_timestamp - INTERVAL '5' SECOND
    ) WITH (
        'connector' = 'kafka',
        'topic' = 'login-events',
        'properties.bootstrap.servers' = 'kafka:9092',
        'format' = 'json'
    )
""")

# Compute failed logins in last 5 minutes using tumbling window
t_env.execute_sql("""
    CREATE TABLE failed_login_features (
        user_id STRING,
        failed_logins_5m BIGINT,
        window_end TIMESTAMP(3),
        PRIMARY KEY (user_id) NOT ENFORCED
    ) WITH (
        'connector' = 'redis',
        'host' = 'redis-feature-store',
        'key-prefix' = 'user:',
        'key-suffix' = ':failed_logins_5m',
        'ttl' = '300'   -- 5 minute TTL matches window
    )
""")

# Stream the computation
t_env.execute_sql("""
    INSERT INTO failed_login_features
    SELECT
        user_id,
        COUNT(*) AS failed_logins_5m,
        TUMBLE_END(event_timestamp, INTERVAL '5' MINUTE) AS window_end
    FROM login_events
    WHERE event_type = 'LOGIN_FAILED'
    GROUP BY
        user_id,
        TUMBLE(event_timestamp, INTERVAL '5' MINUTE)
""")
```

---

## 14.3 Canary Deployments for ML Models — Full Production Playbook

### Why Canary is Different for ML vs Regular Software

```
Regular Software Canary:
  - Deploy new version to 5% of servers
  - Monitor: error rates, latency
  - If stable → promote to 100%
  - Rollback time: seconds

ML Canary:
  - Deploy new model to 5% of traffic
  - Monitor: error rates, latency, PLUS prediction distribution, business KPIs
  - Business KPIs (conversion, revenue) need 3-7 days to show statistically
  - Rollback time: seconds (infra) but model impact may linger (users affected)
```

### Istio-Based Canary Deployment:

```yaml
# canary-virtual-service.yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: fraud-model-routing
  namespace: ml-serving
spec:
  hosts:
    - fraud-model-service
  http:
    # Sticky sessions: Same user always hits same model version
    # Uses hash of user_id header for consistency
    - match:
        - headers:
            x-user-id:
              regex: ".*"
      route:
        - destination:
            host: fraud-model-v3    # Old model (champion)
            port:
              number: 8080
          weight: 95
          headers:
            request:
              set:
                x-model-variant: "v3"
        - destination:
            host: fraud-model-v4    # New model (challenger)
            port:
              number: 8080
          weight: 5
          headers:
            request:
              set:
                x-model-variant: "v4"
```

### Automated Canary Promotion with Flagger:

```yaml
# flagger-canary.yaml — Auto-promotes if metrics are healthy
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: fraud-model
  namespace: ml-serving
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-model
  progressDeadlineSeconds: 3600   # Promote or rollback within 1 hour
  service:
    port: 8080
    targetPort: 8080
  analysis:
    interval: 5m        # Check metrics every 5 minutes
    threshold: 5        # Fail after 5 consecutive bad checks
    maxWeight: 50       # Maximum traffic % to send to canary
    stepWeight: 10      # Increase by 10% each iteration
    metrics:
      # Standard metrics
      - name: request-success-rate
        thresholdRange:
          min: 99       # Minimum 99% success rate
        interval: 1m
      - name: request-duration
        thresholdRange:
          max: 500      # Max 500ms P99 latency
        interval: 30s
      # Custom ML metrics from Prometheus
      - name: ml-prediction-drift
        templateRef:
          name: psi-score
          namespace: flagger-system
        thresholdRange:
          max: 0.15     # Rollback if PSI > 0.15 (high drift vs champion)
        interval: 5m
    webhooks:
      # Notify Slack on promotion/rollback
      - name: slack-notification
        type: event
        url: https://hooks.slack.com/services/YOUR/WEBHOOK
```

### Canary Analysis — What to Check:

```python
# canary_analyzer.py — Decides whether to promote or rollback
import requests
import scipy.stats as stats
import numpy as np
from dataclasses import dataclass

@dataclass
class CanaryDecision:
    should_promote: bool
    confidence: float
    reasons: list

def analyze_canary(
    champion_metrics: dict,
    challenger_metrics: dict,
    min_samples: int = 1000,
) -> CanaryDecision:
    """Statistical analysis to decide if challenger should replace champion."""

    reasons = []

    # 1. Check sufficient sample size
    if challenger_metrics["request_count"] < min_samples:
        return CanaryDecision(False, 0.0, ["insufficient_samples"])

    # 2. Error rate comparison (champion must not be worse)
    if challenger_metrics["error_rate"] > champion_metrics["error_rate"] * 1.1:
        reasons.append(f"error_rate_regression: {challenger_metrics['error_rate']:.4f} vs {champion_metrics['error_rate']:.4f}")
        return CanaryDecision(False, 0.99, reasons)

    # 3. Latency comparison (P99 must not regress by >20%)
    if challenger_metrics["p99_latency_ms"] > champion_metrics["p99_latency_ms"] * 1.2:
        reasons.append(f"latency_regression: {challenger_metrics['p99_latency_ms']}ms vs {champion_metrics['p99_latency_ms']}ms")
        return CanaryDecision(False, 0.95, reasons)

    # 4. Prediction drift check (challenger predictions similar to champion?)
    # Use two-sample KS test on prediction score distributions
    ks_stat, p_value = stats.ks_2samp(
        challenger_metrics["prediction_scores"],
        champion_metrics["prediction_scores"],
    )
    if p_value < 0.01:  # Distributions are statistically different
        reasons.append(f"prediction_distribution_shift: KS={ks_stat:.4f}, p={p_value:.6f}")
        # Don't automatically rollback — this might be intentional improvement
        reasons.append("WARNING: investigate prediction shift before promoting")

    # 5. Business metric comparison (if ground truth available quickly)
    if "approval_rate" in challenger_metrics:
        approval_diff = (
            challenger_metrics["approval_rate"] - champion_metrics["approval_rate"]
        ) / champion_metrics["approval_rate"]

        if approval_diff < -0.05:  # >5% drop in approval rate
            reasons.append(f"approval_rate_drop: {approval_diff:.2%}")
            return CanaryDecision(False, 0.90, reasons)

        if approval_diff > 0.02:  # >2% improvement
            reasons.append(f"approval_rate_improvement: +{approval_diff:.2%}")

    # 6. All checks passed — promote!
    confidence = 0.85 + (
        min(challenger_metrics["request_count"] / 10000, 0.1)
    )  # Higher confidence with more samples, max 0.95
    reasons.append("all_quality_gates_passed")

    return CanaryDecision(True, confidence, reasons)
```

---

## 14.4 A/B Testing in ML Systems — Statistical Rigor

### The Critical Difference: Statistical vs Practical Significance

```
Statistical significance (p < 0.05) means:
  "The probability this result happened by chance is < 5%"

Practical significance means:
  "The improvement is large enough to matter for business"

You can have BOTH wrong:
  - p < 0.001 but effect size is 0.01% (statistically significant, practically useless)
  - p = 0.07 but effect size is 15% (not "statistically significant" but practically huge)
```

### Production A/B Test Implementation:

```python
# ab_testing.py — Rigorous A/B testing for ML models
import hashlib
import scipy.stats as stats
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class ExperimentConfig:
    name: str
    model_a: str         # Champion model name
    model_b: str         # Challenger model name
    traffic_split: float # Fraction going to B (0.0 to 1.0)
    min_samples_per_variant: int = 5000
    significance_level: float = 0.05
    min_effect_size: float = 0.01  # Minimum meaningful improvement


class ABTestRouter:
    """Deterministic, user-sticky A/B test routing."""

    def __init__(self, experiment: ExperimentConfig):
        self.exp = experiment

    def get_variant(self, user_id: str) -> str:
        """Deterministic assignment: same user ALWAYS gets same variant."""
        # Hash combines user_id + experiment name → stable across sessions
        hash_input = f"{user_id}:{self.exp.name}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_value % 10000) / 10000.0  # 0.0 to 1.0

        if bucket < self.exp.traffic_split:
            return "B"  # Challenger
        return "A"  # Champion


class ABTestAnalyzer:
    """Analyze experiment results with proper statistical tests."""

    def analyze_binary_metric(
        self,
        a_successes: int,
        a_total: int,
        b_successes: int,
        b_total: int,
        alpha: float = 0.05,
    ) -> dict:
        """
        Analyze a binary metric (e.g., click, conversion, fraud flag).
        Uses chi-squared test for proportions.
        """
        a_rate = a_successes / a_total
        b_rate = b_successes / b_total
        relative_lift = (b_rate - a_rate) / a_rate

        # Chi-squared test
        contingency_table = np.array([
            [a_successes, a_total - a_successes],
            [b_successes, b_total - b_successes],
        ])
        chi2, p_value, dof, _ = stats.chi2_contingency(contingency_table)

        # Effect size (Cohen's h for proportions)
        effect_size = abs(
            2 * np.arcsin(np.sqrt(b_rate)) - 2 * np.arcsin(np.sqrt(a_rate))
        )

        # Confidence intervals for each rate
        def wilson_ci(successes, total, z=1.96):
            p = successes / total
            n = total
            center = (p + z**2/(2*n)) / (1 + z**2/n)
            margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / (1 + z**2/n)
            return center - margin, center + margin

        a_ci = wilson_ci(a_successes, a_total)
        b_ci = wilson_ci(b_successes, b_total)

        # Statistical power check
        power = self._compute_power(a_rate, b_rate, a_total, b_total, alpha)

        is_significant = p_value < alpha and a_total >= 1000 and b_total >= 1000

        return {
            "a_rate": round(a_rate, 6),
            "b_rate": round(b_rate, 6),
            "relative_lift": round(relative_lift, 4),
            "absolute_lift": round(b_rate - a_rate, 6),
            "p_value": round(p_value, 6),
            "chi2_statistic": round(chi2, 4),
            "effect_size_cohens_h": round(effect_size, 4),
            "statistical_power": round(power, 3),
            "is_statistically_significant": is_significant,
            "a_confidence_interval": [round(x, 6) for x in a_ci],
            "b_confidence_interval": [round(x, 6) for x in b_ci],
            "recommendation": self._get_recommendation(
                is_significant, relative_lift, power
            ),
        }

    def _compute_power(self, p1, p2, n1, n2, alpha):
        """Compute statistical power of the test."""
        effect = abs(p2 - p1) / np.sqrt(p1 * (1-p1))
        # Simplified power calculation
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_power = effect * np.sqrt(min(n1, n2)) - z_alpha
        return stats.norm.cdf(z_power)

    def _get_recommendation(
        self,
        significant: bool,
        lift: float,
        power: float,
    ) -> str:
        if power < 0.8:
            return "INSUFFICIENT_POWER: Collect more data before concluding"
        if not significant:
            return "NO_WINNER: No statistically significant difference detected"
        if lift > 0.02:
            return "PROMOTE_B: Challenger is significantly better (+{:.1%})".format(lift)
        if lift < -0.02:
            return "KEEP_A: Champion is significantly better"
        return "INCONCLUSIVE: Difference is statistically significant but not practically meaningful"
```

### Sample Size Calculator:

```python
def required_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Calculate required sample size per variant.
    
    Args:
        baseline_rate: Current conversion rate (e.g., 0.06 for 6%)
        minimum_detectable_effect: Minimum relative lift to detect (e.g., 0.1 for 10%)
        alpha: Significance level (0.05 standard)
        power: Statistical power (0.80 standard, 0.90 for critical decisions)
    """
    p1 = baseline_rate
    p2 = baseline_rate * (1 + minimum_detectable_effect)

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)

    p_avg = (p1 + p2) / 2

    n = (
        (z_alpha * np.sqrt(2 * p_avg * (1 - p_avg))
         + z_power * np.sqrt(p1 * (1-p1) + p2 * (1-p2))) ** 2
    ) / (p2 - p1) ** 2

    return int(np.ceil(n))

# Example:
# Baseline fraud rate: 6%, want to detect 10% relative improvement
n = required_sample_size(baseline_rate=0.06, minimum_detectable_effect=0.10)
print(f"Need {n:,} samples per variant = {2*n:,} total")
# Output: Need 14,235 samples per variant = 28,470 total
```

---

## 14.5 Multi-Armed Bandit — When A/B Testing is Too Slow

A/B testing wastes traffic on the losing variant. MAB dynamically allocates more traffic to the winning variant while the experiment is running.

```python
# thompson_sampling.py — Thompson Sampling Multi-Armed Bandit
import numpy as np
from dataclasses import dataclass, field
from typing import List

@dataclass
class BanditArm:
    name: str
    successes: int = 0
    failures: int = 0

    def sample(self) -> float:
        """Sample from Beta distribution (posterior over true success rate)."""
        return np.random.beta(self.successes + 1, self.failures + 1)

    @property
    def estimated_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.5


class ThompsonSamplingBandit:
    """
    Multi-Armed Bandit using Thompson Sampling.
    Automatically allocates more traffic to better models.
    """
    def __init__(self, model_names: List[str]):
        self.arms = {name: BanditArm(name=name) for name in model_names}

    def select_model(self) -> str:
        """Select the model to use for this request."""
        # Sample from each arm's posterior
        samples = {
            name: arm.sample()
            for name, arm in self.arms.items()
        }
        # Select arm with highest sample (exploration + exploitation)
        return max(samples, key=samples.get)

    def update(self, model_name: str, success: bool):
        """Update bandit with observed outcome."""
        arm = self.arms[model_name]
        if success:
            arm.successes += 1
        else:
            arm.failures += 1

    def get_allocation(self) -> dict:
        """Show current effective traffic allocation."""
        # Simulate 10K selections to estimate allocation
        selections = [self.select_model() for _ in range(10000)]
        return {
            name: selections.count(name) / len(selections)
            for name in self.arms
        }

# Usage:
bandit = ThompsonSamplingBandit(["model-v3", "model-v4"])

# On each request:
selected_model = bandit.select_model()
result = call_model(selected_model, features)
reward = 1 if result.approved and not chargeback else 0  # Business outcome
bandit.update(selected_model, success=bool(reward))

# After 1000 requests, check allocation:
print(bandit.get_allocation())
# {"model-v3": 0.23, "model-v4": 0.77}
# Bandit automatically sending more to v4 because it's performing better!
```

---

*Next → [Section 15: Learning Roadmap](../learning-roadmap/README.md)*
