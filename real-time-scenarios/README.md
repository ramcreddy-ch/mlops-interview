# Real-Time Production Issues - Interview Questions

## Q1: Kafka lag is spiking, and your real-time inference SLA is failing. How do you debug and fix this?

**Real-world context:**
Our transaction scoring system read from a Kafka topic with 32 partitions. Suddenly, the consumer lag on our inference microservice started climbing by 1,000 messages per second. Transactions were being scored 5 minutes late, missing the authorization window.

**Answer:**

**Root cause analysis:**
Lag means `production rate > consumption rate`. I immediately checked Datadog to narrow it down:
1. **Did traffic spike?** (Checked Kafka producer metrics). Yes, it was Black Friday, traffic was 3x normal.
2. **Is the consumer stuck?** (Checked K8s pod CPU/Memory and application logs). CPU was pegged at 100% on the inference pods.
3. **Is there a bottleneck downstream?** (Checked latency to Redis feature store). Feature retrieval latency spiked from 5ms to 150ms because Redis was maxing out its network bandwidth.

**Debugging steps:**
1. Looked at the ELK logs for the inference pods: `ReadTimeoutError` connecting to Redis.
2. The inference pods were synchronously waiting for Redis to return features. Because Redis was slow, the inference thread was blocked, meaning it couldn't pull the next message from Kafka.
3. This is backpressure.

**The Fix (Immediate):**
1. Scaled out the Redis cluster read replicas to handle the network bandwidth spike.
2. Scaled the K8s inference deployment from 32 to 64 pods to increase concurrent consumption (since we had 64 Kafka partitions, we could have up to 64 active consumers in the group).

**Prevention strategy:**
1. **Timeouts & Circuit Breakers:** The Redis client had a default timeout of 2 seconds. I reduced it to 50ms. If feature retrieval fails within 50ms, a circuit breaker trips, returning a "safe default" feature vector so the transaction can be scored (perhaps less accurately, but meeting the SLA).
2. **Asynchronous I/O:** Switched the Python Kafka consumer to use `asyncio` for feature fetching and inference, preventing blocking on network I/O.
3. **Alerting:** Set a Datadog monitor on Kafka lag > 5,000 messages for 1 minute, paging the on-call engineer before it hits the 5-minute SLA limit.

---

## Q2: A model's inference latency suddenly spikes in production, but traffic is normal. What do you check?

**Real-world context:**
A PyTorch recommendation model's P99 latency jumped from 40ms to 800ms overnight. No new deployments had happened. Traffic volume was flat.

**Answer:**

**Root cause analysis:**
Latency in ML comes from three places: data loading, computation, or network.
1. **Network?** Datadog showed no issues with Istio routing or ingress.
2. **Computation?** K8s metrics showed CPU utilization on the inference pods dropped from 60% to 15%. Wait, if it's slow, shouldn't CPU be high?
3. **Data loading?** Yes. The model was fetching user history from a Cassandra cluster before inference.

**Debugging steps:**
1. I checked the Datadog APM trace for a slow request. The model inference itself was still taking 20ms. The database query was taking 780ms.
2. Looked at the ELK logs for the query: it was doing a full table scan.
3. Why? A data engineer had accidentally dropped an index on the `user_history` table during a midnight migration.

**Another scenario (Model issue):**
I've also seen this happen due to the model inputs themselves. If an NLP model uses dynamic padding (padding the batch to the longest sequence in the batch), and a user suddenly sends a 10,000-word essay to an endpoint that usually sees 10-word queries, the batch size expands, memory spikes, and computation takes 100x longer.

**Fix & Prevention:**
- **For the DB issue:** Restored the index. Added Datadog alerts on slow queries > 100ms.
- **For the dynamic padding issue:** Implemented strict input validation. Truncate inputs to a maximum sequence length (e.g., 512 tokens) at the API gateway level before it ever hits the model.

---

## Q3: K8s pods are crash-looping (OOMKilled) under load. How do you stabilize the system?

**Real-world context:**
During a marketing campaign, our model serving pods started continuously restarting with `OOMKilled` (Out Of Memory). We had horizontal pod autoscaling (HPA) enabled, but new pods would just crash as soon as they received traffic.

**Answer:**

**Root cause analysis:**
`OOMKilled` means the pod exceeded its `resources.limits.memory`.
In ML, memory spikes usually happen for two reasons:
1. Loading the model into RAM (if it's a large model and multiple worker processes load separate copies).
2. Processing large batches or large payload sizes.

**Debugging steps:**
1. `kubectl describe pod <pod-name>` confirmed the `OOMKilled` exit code 137.
2. Checked Datadog memory profiles. The memory grew linearly with the number of concurrent requests.
3. The serving framework (Gunicorn/Flask) was spawning a new worker process for each request up to 8 workers, and each worker was loading a 1GB Scikit-learn model into memory. 8 workers * 1GB = 8GB, but the pod limit was 4GB.

**The Fix (Immediate):**
1. Hot-patched the K8s deployment to increase the memory limit to 10GB to stabilize the system.
2. Alternatively, if nodes are full, reduce the Gunicorn worker count from 8 to 2 so memory stays under 4GB, but throughput will drop.

**The Fix (Long-term):**
1. **Shared Memory:** Switched from Flask to KServe (or Triton/FastAPI). These frameworks load the model into memory once, and multiple threads/coroutines share the model for inference, drastically reducing memory overhead.
2. **Batch Size Limits:** Configured maximum payload sizes at the ingress layer. A user shouldn't be able to send a 50MB JSON array for inference in a single request.
3. **Right-sizing:** Ran load tests using Locust to profile actual memory usage under stress, and set K8s `requests` and `limits` appropriately (e.g., Request: 3GB, Limit: 4GB).

---

## Q4: Datadog alerts are noisy, and your team is experiencing alert fatigue. How do you fix your monitoring strategy?

**Real-world context:**
Our Slack channel had 500+ alerts a day. "Pod restarted", "CPU > 80%", "Kafka lag > 100". Engineers started ignoring the channel. Then a silent failure happened where predictions were all zeros for 4 hours, and nobody noticed because it was buried in noise.

**Answer:**

**Root cause analysis:**
We were alerting on *causes* (CPU high) rather than *symptoms* (users are affected). This is an anti-pattern. High CPU is only a problem if latency goes up or requests fail.

**Debugging steps & Strategy shift:**
I completely overhauled the Datadog monitors based on Google's SRE principles (SLIs/SLOs).

1.  **Deleted all static threshold alerts** on CPU, Memory, and Network. High CPU is the HPA's job to fix, not a human's.
2.  **Created Symptom-Based Alerts (P1 - PagerDuty):**
    *   **Latency:** P99 inference latency > 100ms for 5 minutes.
    *   **Error Rate:** 5xx HTTP errors > 1% for 5 minutes.
    *   **Traffic Drop:** Incoming requests drop by > 50% compared to the same time last week (anomaly detection).
3.  **Created ML-Specific Alerts (P2 - Slack/Jira):**
    *   **Data Drift:** Feature PSI > 0.2 (checked daily).
    *   **Prediction Skew:** Ratio of "approved" vs "denied" shifts by > 10%.
    *   **Feature Missing Rate:** Null values in input features > 5%.
4.  **Used Composite Monitors:** Only alert if Kafka lag is high AND inference throughput has dropped.

**Trade-offs:**
- Anomaly detection monitors in Datadog can be tricky to tune for seasonal data. You have to ensure the baseline is accurate, otherwise, they become noisy too.

**Common mistakes:**
- Alerting when a single pod restarts. In K8s, pods are ephemeral. Restarts are expected. Only alert if the Deployment's availability drops below the target replica count.

---
*[Back to README](../README.md)*
