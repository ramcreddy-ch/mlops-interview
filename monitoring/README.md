# Monitoring & Observability for ML - Interview Questions

## Q1: Explain the three layers of ML monitoring.
**Context:** DevOps only monitored CPU, ignoring model accuracy.
**Answer:** 1. Infra Metrics (CPU, Memory, 5xx errors) via Datadog. 2. Model Metrics (Data Drift, Prediction Skew, Feature Nulls) via custom scripts or Evidently. 3. Business Metrics (CTR, Revenue, Conversion) via BI tools or Datadog custom events. You need all three to trace a business drop to a CPU spike.

## Q2: How do you use ELK to debug a model failure?
**Context:** Customer complained about bizarre recommendations.
**Answer:** Log every request to Kafka/Logstash as structured JSON: `[timestamp, trace_id, features, prediction]`. When a customer complains, query Kibana for their `user_id` to see the exact feature vector passed to the model. Often reveals upstream data pipeline failures (e.g., passing a null value).

## Q3: How do you trace an ML pipeline end-to-end?
**Context:** Took 3 days for a user click to affect predictions.
**Answer:** Distributed Tracing (OpenTelemetry). Inject a Trace ID at the mobile app. Pass it through Kafka -> Flink -> Feature Store metadata -> Inference Request -> Final Prediction Log. Use Datadog APM to view the flame graph and identify bottlenecks (e.g., Kafka consumer lag).

## Q4: How do you fix "Alert Fatigue" in MLOps?
**Context:** 500+ Slack alerts a day, teams ignored them.
**Answer:** Delete static threshold alerts (CPU > 80%). High CPU is normal for ML. Switch to Symptom-Based SLIs (Service Level Indicators). Alert only on: Latency (P99 > 100ms), Error Rates (5xx > 1%), or Business Impact (Approval Rate drop > 10%).

## Q5: How do you monitor Data Drift efficiently?
**Context:** Calculating PSI on every request was crashing the API.
**Answer:** Drift calculation is heavy. Never do it in the hot path. Log features to S3 asynchronously. Run an hourly or daily Airflow batch job to compute PSI (Population Stability Index) against a training baseline, and push the aggregate score to Datadog as a gauge metric.

## Q6: What is Concept Drift and how do you monitor it?
**Context:** Fraud patterns changed overnight.
**Answer:** Concept drift is when the relationship between input and output changes (the definition of fraud changes). You cannot detect this with features alone. You must monitor Ground Truth. When labels arrive (chargebacks), compare Predicted vs Actual and trigger an alert if F1 score drops.

## Q7: Monitoring Model Staleness.
**Context:** A model hadn't been retrained in 2 years.
**Answer:** Track the `model_training_timestamp` as a metric. Set an alert if `time.now() - training_timestamp > 30 days`. This enforces a continuous training discipline even if drift isn't explicitly detected.

## Q8: Setting up SLIs and SLOs for an ML Endpoint.
**Context:** Defining reliability targets.
**Answer:** SLI (Indicator): Proportion of requests served under 50ms without error. SLO (Objective): 99.9% over a 30-day window. SLA (Agreement): Financial penalty if SLO is missed. For ML, also add an ML-specific SLO: "95% of predictions fall within the expected probability distribution."

## Q9: How do you monitor GPU memory fragmentation?
**Context:** CUDA OOM crashes despite low average memory use.
**Answer:** Average memory utilization hides fragmentation. Monitor `nvidia.smi.memory_allocated` vs `nvidia.smi.memory_reserved` (cached by PyTorch). If reserved is maxed but allocated is low, you have fragmentation.

## Q10: How do you handle ground-truth delay in monitoring?
**Context:** Credit defaults take 90 days to verify.
**Answer:** You can't monitor accuracy in real-time. Instead, monitor **Prediction Distribution**. If the model historically approves 80% of loans, and suddenly approves 40%, the model behavior has drifted, serving as a proxy for accuracy drops.

## Q11: How do you monitor Feature Store freshness?
**Context:** Kafka stream failed, features became stale.
**Answer:** Add a `last_updated_timestamp` metadata field to every feature vector in Redis. The inference service emits a metric: `time.now() - last_updated_timestamp`. Alert if freshness exceeds 15 minutes.

## Q12: How do you monitor fairness and bias in production?
**Context:** Ensuring the model doesn't discriminate.
**Answer:** Segment prediction distributions by protected classes (Age, Gender, Geo). Emit metrics for `approval_rate_segment_a` vs `approval_rate_segment_b`. Alert if the Disparate Impact ratio drops below 0.8.

## Q13: What is the "Golden Signals" framework applied to ML?
**Context:** Standardizing dashboards.
**Answer:** SRE Golden Signals: Latency, Traffic, Errors, Saturation. For ML, Saturation is usually GPU utilization/VRAM or Kafka Consumer Lag. Errors include 5xx AND "Fallback triggered" (when the model fails and returns a hardcoded default).

## Q14: How do you monitor embeddings/vectors for drift?
**Context:** Standard PSI doesn't work on 512-dimensional vectors.
**Answer:** Use distance metrics. Compute the Cosine Similarity between the daily average embedding vector and the training baseline average vector. Alternatively, use Maximum Mean Discrepancy (MMD) to compare the high-dimensional distributions.

## Q15: How do you implement logging sampling?
**Context:** Logging 10k JSON payloads/sec to ELK costs $100k/month.
**Answer:** Implement dynamic sampling. Log 100% of errors and anomalies (confidence score < 0.1). Log only 1% of normal requests to ELK for debugging, but push 100% to S3/Parquet for cheap batch analysis.

## Q16: Monitoring Out-of-Distribution (OOD) inputs.
**Context:** Users submitting gibberish to an NLP model.
**Answer:** Train a fast anomaly detection model (Isolation Forest) alongside the main model. If the input is OOD, the anomaly model flags it in <1ms, the main model is skipped, and a generic fallback response is returned and logged.

## Q17: Tracking Hyperparameter Tuning performance.
**Context:** 500 experiments running, hard to find the best.
**Answer:** Use MLflow or Weights & Biases (W&B). Instrument the training code to log `loss` per epoch. Use the UI parallel coordinates plot to visualize which hyperparameter combinations yield the fastest convergence.

## Q18: Monitoring for "Feature Nulling".
**Context:** Upstream API failed, sending nulls.
**Answer:** XGBoost handles nulls silently by following default tree paths. This degrades accuracy silently. Emit a metric for `percentage_of_null_features_per_request`. Alert if it spikes above a baseline.

## Q19: Using Prometheus and Grafana for ML.
**Context:** Open-source monitoring stack.
**Answer:** Expose a `/metrics` endpoint on the FastAPI/KServe pod using the Prometheus Python client. K8s Prometheus Operator scrapes it. Build Grafana dashboards visualizing P99 latency and prediction histograms.

## Q20: Monitoring Model Calibration.
**Context:** Confidence scores were wildly inaccurate.
**Answer:** Group predictions into bins (e.g., 0.8-0.9 confidence). When ground truth arrives, check if the actual positive rate in that bin was 85%. Plot a Reliability Diagram in Grafana to visualize miscalibration.

## Q21: How do you correlate ML events with infrastructure events?
**Context:** Model accuracy dropped exactly when a K8s node was replaced.
**Answer:** Use Datadog Event Overlays. Overlay K8s deployment events, feature store syncs, and Airflow job completions directly on top of the prediction latency/drift graphs to visually identify root causes.

## Q22: Monitoring Kafka Consumer Lag for ML pipelines.
**Context:** Lag spikes during peak hours.
**Answer:** Monitor `kafka_consumergroup_lag`. Set alerts based on the rate of change (derivative) rather than a static threshold, because a temporary lag spike is normal, but a continuously increasing lag indicates a dead consumer.

## Q23: How do you test your monitoring systems?
**Context:** An alert failed to fire during a real outage.
**Answer:** Chaos Engineering. Inject bad data into the staging environment. Force a Redis timeout. Verify that the circuit breakers trip, the fallback metric increments, and the PagerDuty alert fires.

## Q24: What is the "Watermelon Effect" in ML monitoring?
**Context:** Dashboard is green, but users are angry.
**Answer:** Green on the outside, red on the inside. Happens when you average metrics globally. Overall latency is 50ms, but for premium users in Europe, it's 5 seconds. Fix: Always use high-cardinality tags (Region, Tier) and monitor P99s, not averages.

## Q25: Monitoring hardware accelerators (TPUs/GPUs).
**Context:** Ensuring ROI on expensive hardware.
**Answer:** Use NVIDIA DCGM Exporter. Monitor `DCGM_FI_DEV_GPU_UTIL` (streaming multiprocessor util) and `DCGM_FI_DEV_MEM_COPY_UTIL` (PCIe bandwidth). If memory copy is high but GPU util is low, you are I/O bottlenecked.

## Q26: Monitoring Model Ensembles.
**Context:** 3 models voting on an outcome.
**Answer:** Monitor the "disagreement rate". If Model A and Model B usually agree 95% of the time, and suddenly agree only 60% of the time, one of the models is experiencing drift or feature corruption.

## Q27: How do you handle PII in logging and monitoring?
**Context:** Plaintext SSNs found in Datadog logs.
**Answer:** Implement a central logging sidecar or wrapper that scrubs strings matching regex patterns (emails, SSNs) or hashes user IDs *before* the log is emitted over the network.

## Q28: Establishing baseline metrics for new models.
**Context:** Alerting on day 1 is impossible.
**Answer:** Run the model in "Shadow Mode" for 7 days. Use this period to establish the baseline distribution of predictions and latency. After 7 days, activate the anomaly detection alerts based on this baseline.

## Q29: Monitoring third-party API dependencies.
**Context:** ML model relies on Google Maps API for features.
**Answer:** Instrument external API calls. Monitor the latency and HTTP error rates of the third-party service. If it degrades, your model degrades. Alert on `third_party_api_latency > 500ms`.

## Q30: What is continuous profiling?
**Context:** CPU spiked, but we didn't know which Python function caused it.
**Answer:** Use tools like Datadog Continuous Profiler or Pyroscope. They run in the background with low overhead and generate flame graphs of CPU and Memory usage down to the line of code, allowing post-incident forensic analysis.
