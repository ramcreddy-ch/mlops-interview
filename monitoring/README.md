# Monitoring & Observability for ML - Interview Questions

## Q1: Explain the difference between infra metrics, model metrics, and business metrics. How do you monitor all three?

**Real-world context:**
When I joined a previous company, the DevOps team was monitoring CPU and latency, and the data scientists were running offline scripts a week later to check model accuracy. We had a model that started predicting "0" for every request due to a bad feature update. DevOps saw healthy pods (latency was actually faster!). Data science found out 5 days later. We needed a unified strategy.

**Answer:**

A production ML monitoring strategy must cover three distinct layers. If you miss one, you have a blind spot.

**1. Infrastructure Metrics (The "Is it running?" layer)**
*   **What it is:** CPU, GPU utilization, memory, network I/O, P50/P99 latency, HTTP 5xx error rates, pod restarts.
*   **Tools:** Datadog agent, Prometheus + Grafana, Kube-state-metrics.
*   **The specific ML challenge:** GPU memory leaks. A PyTorch model might look healthy on CPU, but if GPU VRAM is exhausted due to a fragmented cache, the pod will crash. I monitor `nvidia.smi.memory_used` closely.

**2. Model Metrics (The "Is it predicting correctly?" layer)**
*   **What it is:** Data drift (input features), Concept drift (model output distribution), feature null rates, prediction latency (just the model forward pass, excluding network).
*   **Tools:** Evidently AI, custom Python scripts emitting StatsD metrics to Datadog, ELK stack for logging raw predictions.
*   **The specific ML challenge:** Ground truth delay. In fraud detection, you don't know if a prediction was correct until a customer files a chargeback 30 days later. You can't monitor accuracy in real-time. Instead, I monitor **Prediction Distribution**. If the model usually flags 2% of transactions as fraud, and suddenly it's flagging 15%, the model behavior has drifted, and I alert on that anomaly.

**3. Business Metrics (The "Is it making money?" layer)**
*   **What it is:** Conversion rate, total fraud losses prevented, click-through rate, user engagement.
*   **Tools:** Usually a BI tool (Looker/Tableau) reading from a data warehouse (Snowflake), but critical metrics should be piped back into Datadog.
*   **The specific ML challenge:** Connecting a model version to a business outcome. I ensure every prediction logged to the data lake includes a `model_version` tag so business analysts can group CTR by model version.

**My Unified Architecture:**
1. Inference service handles request.
2. Infra metrics go straight to Datadog.
3. Inference service logs the `[timestamp, model_version, input_features, prediction]` asynchronously to a Kafka topic.
4. Logstash consumes Kafka and indexes in Elasticsearch (ELK) for debugging.
5. A daily Spark job reads the Kafka data from S3, calculates Feature Drift (PSI), and pushes custom metrics to Datadog.

---

## Q2: How do you use ELK to debug a model failure in production?

**Real-world context:**
A customer complained they were receiving bizarre product recommendations. The model was a black box to the customer support team.

**Answer:**

ELK (Elasticsearch, Logstash, Kibana) is essential for forensic debugging in ML. You cannot debug a model if you don't know what features were fed into it at the exact moment of inference.

**My Logging Strategy:**
Every inference request generates a structured JSON log line. We never log unstructured text.

```json
{
  "timestamp": "2026-05-01T10:15:30Z",
  "trace_id": "req-8f7a-12bc",
  "endpoint": "/v1/recommend",
  "model_name": "reco-engine",
  "model_version": "v2.1",
  "latency_total_ms": 45,
  "latency_feature_fetch_ms": 15,
  "latency_inference_ms": 30,
  "customer_id": "user_998",
  "input_features": {
    "age_segment": 3,
    "last_purchase_days": 4,
    "cart_value": null
  },
  "prediction_output": ["item_123", "item_456"],
  "fallback_triggered": false
}
```

**The Debugging Process:**
1. Support ticket gives me a `customer_id` and rough timestamp.
2. In Kibana, I query `customer_id: "user_998" AND model_name: "reco-engine"`.
3. I find the exact log line.
4. I inspect the `input_features`. I notice `cart_value` is `null`.
5. I realize the upstream feature service failed to fetch the cart value, passing `null` to the model. The model wasn't trained to handle a null `cart_value` robustly, causing it to output default/bizarre recommendations.

**Trade-offs & Mistakes:**
*   **Mistake:** Logging synchronous in the hot path. I've seen teams add a `logger.info()` that writes to a slow network drive, adding 20ms to every inference request. Always use asynchronous logging (e.g., standard out to Fluentd/Logstash daemonset, or async Kafka producer).
*   **Trade-off (Cost):** Logging 10,000 JSON payloads per second to Elasticsearch is incredibly expensive. For high-throughput endpoints, we use a sampler (e.g., log 5% of requests to ELK, but push 100% to cold storage in S3).

---

## Q3: How do you trace an ML pipeline from data ingestion to model serving?

**Real-world context:**
Data scientists complained it was taking 3 days for a new behavioral event to reflect in the model's predictions. Nobody knew where the bottleneck was: Airflow? Spark? The Feature Store sync?

**Answer:**

Tracing ML pipelines is harder than tracing microservices because it spans batch jobs, streaming systems, and synchronous APIs.

**Implementation (Distributed Tracing):**

We use OpenTelemetry (or Datadog APM) and pass a Trace ID through the entire lifecycle.

1.  **Data Ingestion:** When a user clicks a button, the mobile app generates a `Trace ID`. This event lands in Kafka.
2.  **Feature Processing (Spark/Flink):** When the Flink job aggregates that click into the `user_clicks_24h` feature, it attaches the `Trace ID` to the metadata when writing to the Redis Feature Store.
3.  **Model Inference:** When the inference service queries Redis, it pulls the feature and the associated `Trace ID`.
4.  **Logging:** The final prediction log includes the original `Trace ID`.

**In Datadog APM, this looks like a flame graph:**
*   `Mobile_Click` (Start)
*   `Kafka_Queue` (2 mins)
*   `Flink_Processing` (500ms)
*   `Redis_Write` (2ms)
*   [... time passes ...]
*   `Inference_Request` (Start)
*   `Redis_Read` (5ms)
*   `Model_Predict` (20ms)

**Debugging the SLA miss:**
By looking at the traces, I could see the `Kafka_Queue` span was taking 2.5 days. The bottleneck was a Kafka consumer group that had died and fallen massively behind on offsets. The ML pipeline code was fine; it was a pure infrastructure bottleneck.

---
*[Back to README](../README.md)*
