# Real-Time Production Issues — Production Deep Dive
## Incident Response, Debugging Kafka/Redis/K8s at Scale

---

## Part 1: The MLOps Incident Response Playbook

When an ML system breaks in production, it usually fails silently. The API still returns `200 OK`, but the predictions are garbage (e.g., approving all fraud, recommending empty items).

### The First 15 Minutes (Triage)

```
Symptom: Business metric drop (e.g., conversion rate down 15%).

1. Check Infrastructure (Is the code running?)
   - Are pods crashing? (OOMKilled)
   - Is latency spiking? (Timeout cascades)
   - Is Kafka lag growing? (Data not reaching model)
   → If YES: This is a DevOps issue. Scale up, restart, or rollback.
   → If NO: Proceed to step 2.

2. Check the Model (Did the code/weights change?)
   - Look at the Model Registry: Was a new model deployed today?
   - Look at GitOps: Were there infra/config changes?
   → If YES: Rollback immediately to the previous version. Investigate later.
   → If NO: Proceed to step 3.

3. Check the Data (Garbage in, garbage out)
   - Feature Null Rates: Did an upstream table drop a column?
   - Feature Distributions (PSI): Did a partner change their API format?
   → If YES: Implement a hotfix (impute nulls) or disable the model (fallback to rules).
```

---

## Part 2: Kafka & Streaming Bottlenecks

### The Problem: Consumer Lag Death Spiral

**Scenario:** 
- Traffic spikes by 3x.
- Inference pods process requests.
- Redis feature store gets overwhelmed by the 3x read volume.
- Redis response time goes from 5ms to 150ms.
- Inference pods wait 150ms per request.
- Inference pods stop pulling from Kafka fast enough.
- Kafka lag grows to 50,000 messages. Predictions are now 10 minutes old.

### The Fix: Circuit Breakers & Fallbacks

You cannot let a slow database take down your real-time stream.

```python
# circuit_breaker.py — Protects the inference loop from slow feature stores
import pybreaker
import redis
from contextlib import contextmanager

# Trip circuit if 5 consecutive errors occur, wait 30s before retrying
redis_breaker = pybreaker.CircuitBreaker(fail_max=5, reset_timeout=30)
redis_client = redis.Redis(host='feature-store', socket_timeout=0.05) # 50ms timeout!

def get_fallback_features():
    """Safe, neutral features that allow the model to make a conservative guess."""
    return {"avg_spend_30d": 50.0, "failed_logins": 0, "is_known_device": True}

@redis_breaker
def fetch_features(user_id):
    """Attempt to fetch from Redis."""
    return redis_client.hgetall(f"user:{user_id}")

def process_kafka_message(msg):
    try:
        features = fetch_features(msg.user_id)
    except (redis.exceptions.TimeoutError, pybreaker.CircuitBreakerError):
        # Redis is slow or down. Do not wait!
        # Log metric for alerting:
        # statsd.increment("feature_fallback_triggered")
        features = get_fallback_features()
    
    # Model executes instantly because we didn't block on network I/O
    prediction = model.predict(features)
    return prediction
```

---

## Part 3: Memory Leaks & OOMKilled in Python Services

### The Problem: Memory Growth Over Time

**Scenario:** Your FastAPI inference service starts at 2GB RAM. Over 24 hours, it grows to 8GB and K8s kills it (`OOMKilled`).

**Why this happens in Python MLOps:**
1. **Global Variables:** Appending predictions to a global list for "logging".
2. **PyTorch Graph Retention:** Saving `loss` instead of `loss.item()` keeps the entire computational graph in memory.
3. **Pandas Memory:** Loading large Parquet files into Pandas inside the request handler and not garbage collecting them.

### The Fix: Memory Profiling & Architecture

```python
# memory_debug.py — Finding the leak
import objgraph
import gc

def debug_memory_leak():
    # Force garbage collection
    gc.collect()
    
    # Print the top 10 most common objects in memory
    print("Top objects in memory:")
    objgraph.show_most_common_types(limit=10)
    
    # Usually you will find a massive `list` or `dict` that shouldn't be there.
    # If you see `torch.Tensor` growing continuously, you have a PyTorch graph leak.
```

**Architectural Fix:** Use a pre-forking server like Gunicorn/Uvicorn with a **Max Requests** limit. This is a common production hack:

```bash
# Restart the worker cleanly after every 10,000 requests.
# This completely flushes any memory leaks before they hit the K8s OOM limit.
gunicorn serve:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --max-requests 10000 --max-requests-jitter 1000
```

---

## Part 4: The Cache Stampede (Thundering Herd)

### The Problem: Synchronized Expiration

**Scenario:**
- You pre-compute recommendations for 10M users every night at 2 AM.
- You cache them in Redis with a TTL of 24 hours.
- Exactly 24 hours later (2 AM next day), ALL 10M keys expire at the exact same millisecond.
- Suddenly, every user who logs in gets a cache miss.
- The system attempts to run the heavy recommendation model on the fly for 50,000 concurrent users.
- The database and GPU cluster instantly crash.

### The Fix: Probabilistic Early Expiration (PER) & Jitter

Never let all cache keys expire at the same time. Add random jitter to the TTL.

```python
import random
import redis

client = redis.Redis()

def cache_recommendations(user_id, recs):
    # Base TTL is 24 hours (86400 seconds)
    # Add +/- 2 hours of random jitter
    jitter = random.randint(-7200, 7200)
    ttl = 86400 + jitter
    
    client.setex(f"recs:{user_id}", ttl, recs)
    # Now the expirations are smeared smoothly across a 4-hour window!
```

---

## Interview Questions (30 Q&A) — See Original Below

---

## Q1: Kafka lag is spiking, and your real-time inference SLA is failing. How do you debug and fix this?
**Context:** Our transaction scoring system read from a Kafka topic with 32 partitions. Suddenly, the consumer lag started climbing by 1,000 messages per second.
**Answer:** Lag means production > consumption. First check if traffic spiked. If not, check if consumer is bottlenecked. In my case, Redis feature store latency spiked to 150ms. The inference threads blocked waiting for Redis. Fix: Added a 50ms circuit breaker to Redis fetch, falling back to safe default features so the model could keep scoring and clear the Kafka queue.

## Q2: A model's inference latency suddenly spikes in production, but traffic is normal. What do you check?
**Context:** PyTorch model P99 latency jumped from 40ms to 800ms.
**Answer:** Check network, compute, and data. K8s showed CPU actually dropped. The bottleneck was data loading — a database index was accidentally dropped, turning a 2ms feature fetch into a 780ms full table scan. Another possibility: NLP dynamic padding. A user sent a massive text block, exploding the tensor size. Fix: Strict input length validation at API gateway.

## Q3: K8s pods are crash-looping (OOMKilled) under load. How do you stabilize the system?
**Context:** Serving pods restarting continuously.
**Answer:** OOMKilled (Exit Code 137). Checked memory profile. Gunicorn was spawning 8 workers, each loading a 1GB Scikit-learn model, exceeding the 4GB pod limit. Immediate fix: hot-patch deployment to increase memory to 10GB or reduce workers to 2. Long-term fix: Switch to Triton/KServe which use shared memory for models.

## Q4: Datadog alerts are noisy, and your team is experiencing alert fatigue. How do you fix your monitoring strategy?
**Context:** 500+ Slack alerts a day.
**Answer:** We were alerting on *causes* (CPU high) rather than *symptoms* (users affected). Deleted static CPU/Mem thresholds. Created P1 alerts on Symptoms (P99 Latency > 100ms, 5xx Error > 1%). Created P2 ML alerts on Data Drift (PSI > 0.2) and Prediction Skew.

## Q5: A new model version is deployed, and suddenly the Feature Store (Redis) crashes. What happened?
**Context:** Blue/green deploy took down Redis.
**Answer:** A Cache Stampede. V2 required a new feature not yet in Redis. Flipping 100% traffic to V2 caused 10,000 concurrent cache misses per second, overwhelming the backend DB calculating the feature. Fix: Cache Warming (run Airflow job to pre-populate Redis *before* routing traffic) and use Canary Deployments (1% traffic to slowly warm cache).

## Q6: Data scientists raise false alarms saying "Data Drift is detected!" every Monday. How do you fix it?
**Context:** Evidently AI alerting every Monday morning at 8 AM.
**Answer:** The drift monitor compared "Last 1 hour" to "Full Training Set". Monday 8 AM traffic is purely European early-birds (different behavior than global average). This is seasonality, not drift. Fix: Compare "Last 24 hours" vs "Same 24 hours last week", and enforce a minimum sample size (e.g., 10,000 predictions) before calculating drift.

---
*Next → [System Design](../system-design/README.md)*
