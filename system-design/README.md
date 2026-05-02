# System Design - Interview Questions

## Q1: Design a real-time ML inference system using Kafka + Kubernetes.
**Context:** We needed to score 8,000 credit card transactions per second for fraud, with a hard P99 SLA of 80ms.
**Architecture & Example:**
1. **Ingestion:** Transactions hit a Kafka topic (`txn-events`) with 64 partitions to allow high concurrent consumption.
2. **Enrichment:** A Flink streaming job consumes the topic, queries a Redis online feature store for historical aggregates (e.g., `user_30d_txn_count`), and produces an enriched payload to another Kafka topic (`txn-enriched`).
3. **Inference:** A Kubernetes Deployment running KServe (or Triton) consumes `txn-enriched`. 
   * *Autoscaling Example:* We configured KEDA to scale the KServe pods based on `kafka_consumer_lag`. If lag > 1000, spin up 5 more pods.
4. **Decision:** The model outputs a probability (e.g., `0.92`). A downstream decision service reads this and applies business rules: `if p > 0.90: block`.
**Common Failure:** Redis connection timeouts under load. *Fix:* We implemented a strict 50ms circuit breaker using the `tenacity` Python library. If Redis times out, the circuit opens, and the model uses a "safe default" feature vector (e.g., assuming average historical behavior) rather than crashing the entire transaction pipeline.

## Q2: Design a scalable model deployment platform for 50+ Data Scientists.
**Context:** Data scientists were manually SSHing into EC2 instances to run `nohup python serve.py &`, leading to zero traceability and frequent outages.
**Architecture & Example:**
1. **The Contract:** Data scientists must log their models to a centralized MLflow Tracking server using `mlflow.pyfunc.log_model()`. This ensures the model and its Python dependencies (`conda.yaml`) are bundled together.
2. **The CI/CD Trigger:** When a DS clicks "Promote to Staging" in the MLflow UI, a webhook triggers a Jenkins pipeline.
3. **The Build:** Jenkins pulls the MLflow artifact, uses the `mlflow models build-docker` command to generate a standardized REST API container, and pushes it to AWS ECR.
4. **The Deployment:** Jenkins commits a new Helm `values.yaml` to a GitOps repository (e.g., `image: my-model:v2`). ArgoCD detects the Git commit and applies the rolling update to the Kubernetes cluster.
**Why this works:** Data scientists never write Dockerfiles or Kubernetes YAML. They just write Python and click a button in MLflow, but the platform engineers maintain strict, immutable GitOps deployments.

## Q3: Design a feature store handling both batch and streaming data.
**Context:** Our fraud model needed a batch feature (`avg_spend_last_30_days`) and a streaming feature (`failed_logins_last_5_mins`). Without a feature store, we suffered massive training-serving skew.
**Architecture & Example (Using Feast):**
1. **Offline Store (Training):** Data sits in Snowflake/S3. Feast generates point-in-time correct training datasets. If we are training on a fraud event from Tuesday at 2:00 PM, Feast ensures the `failed_logins_last_5_mins` feature reflects Tuesday at 1:55 PM, preventing data leakage from the future.
2. **Online Store (Serving):** Redis Cluster.
3. **Materialization (The Glue):** 
   * *Batch:* A nightly Airflow DAG runs `feast materialize`. It queries Snowflake, calculates the 30-day average, and writes the keys/values to Redis.
   * *Streaming:* A Flink job listens to a Kafka `logins` topic, maintains a 5-minute sliding window count, and continuously `UPSERT`s the count directly into Redis.
**Example Code (Serving):**
```python
# The inference service is blissfully unaware of Flink or Snowflake
features = feast_client.get_online_features(
    feature_refs=["fraud:avg_spend_30d", "fraud:failed_logins_5m"],
    entity_rows=[{"user_id": "u123"}]
).to_dict()
model.predict(features)
```

## Q4: Design a multi-tenant ML platform on Kubernetes.
**Context:** The Risk team and the Marketing team were sharing a 20-GPU K8s cluster. The Marketing team ran a massive hyperparameter sweep that consumed all 20 GPUs, blocking the Risk team's critical real-time inference pods.
**Architecture & Example:**
1. **Namespaces:** Each team gets a dedicated K8s namespace (`ml-risk`, `ml-marketing`).
2. **Resource Quotas (The Fix):** We applied strict `ResourceQuota` objects to each namespace.
   ```yaml
   apiVersion: v1
   kind: ResourceQuota
   metadata:
     name: compute-quota
     namespace: ml-marketing
   spec:
     hard:
       requests.nvidia.com/gpu: "4" # Marketing can never use more than 4 GPUs
   ```
3. **Network Isolation:** We deployed K8s `NetworkPolicies` ensuring pods in `ml-marketing` cannot communicate with databases in `ml-risk`, fulfilling compliance requirements.
4. **Chargeback:** We installed Kubecost, configured it to aggregate costs by the `namespace` label, and automatically sent monthly AWS bills to each team's cost center.

## Q5: Design a multi-region, high-availability ML inference architecture.
**Context:** Our image classification API needed to serve global users with <200ms latency, and survive a complete AWS `us-east-1` region outage.
**Architecture & Example:**
1. **Routing:** AWS Route 53 configured with Latency-Based Routing. A user in Paris resolves the DNS to the `eu-west-1` Application Load Balancer (ALB).
2. **Compute Replication:** Identical EKS clusters run in `us-east-1` and `eu-west-1`. Our CD pipeline (GitHub Actions) pushes K8s manifests to both regions simultaneously.
3. **Data Replication (The Hard Part):** If a user updates their profile in Paris, the EU Redis feature store gets updated. We used **Redis Enterprise Active-Active (CRDTs)** to automatically replicate that feature update to the US Redis cluster in <50ms.
**Disaster Recovery Example:** If `us-east-1` burns down, Route 53 health checks fail. Traffic automatically shifts to `eu-west-1`. The K8s Horizontal Pod Autoscaler (HPA) in Europe detects the 2x traffic spike and scales the inference deployments from 20 to 40 pods to absorb the load.

## Q6: Design an automated A/B testing pipeline for ML models.
**Context:** The marketing team wanted to test a collaborative filtering model (Model A) against a deep learning recommender (Model B) to see which generated higher click-through rates (CTR).
**Architecture & Example:**
1. **Deployment:** Both models are deployed as separate K8s Services (`reco-model-a-svc`, `reco-model-b-svc`).
2. **Deterministic Routing:** We do NOT use round-robin load balancing. If a user gets Model A at 9:00 AM, they must get Model A at 9:05 AM to prevent UX thrashing. We put an API Gateway (Kong/Envoy) in front.
   * *Logic:* `variant = hash(user_id + "experiment_v1") % 100`. If `variant < 50`, route to A, else B.
3. **Telemetry Tracking:** The API Gateway injects an HTTP header `X-Model-Variant: A` into the request. The inference service logs the prediction to Kafka with this tag: `{"user": "u123", "item": "i45", "variant": "A"}`.
4. **Evaluation:** When the user clicks an item, the frontend emits a click event to Kafka. A Spark job joins the "Predictions" topic and the "Clicks" topic on `user_id` and `item_id`, grouping by `variant` to calculate real-time CTR in a Grafana dashboard.

## Q7: Design a shadow deployment architecture.
**Context:** We built a new deep learning model for loan approvals. It was too risky to A/B test (we couldn't risk approving bad loans just for an experiment), but we needed to see how it performed on real production data.
**Architecture & Example:**
1. **Traffic Mirroring:** We used Istio Service Mesh. We configured an Istio `VirtualService` to route 100% of the primary traffic to the existing V1 model, but **mirror** a copy of that traffic to the V2 shadow model.
   ```yaml
   apiVersion: networking.istio.io/v1alpha3
   kind: VirtualService
   metadata:
     name: loan-routing
   spec:
     hosts:
     - loan-service
     http:
     - route:
       - destination:
           host: model-v1
       mirror:
         host: model-v2
       mirrorPercentage:
         value: 100.0
   ```
2. **The "Fire and Forget":** Istio sends the request to V2, but completely ignores V2's response. The user only ever sees V1's response.
3. **Logging:** V2 processes the request and logs its prediction to a "Shadow Predictions" database table. Data scientists later compare V1 and V2 outputs to ensure V2 is safe to promote.

## Q8: Design an Edge ML deployment system (IoT/Mobile).
**Context:** Deploying a defect-detection computer vision model to 10,000 factory cameras with intermittent internet connectivity.
**Architecture & Example:**
1. **Training (Cloud):** Train a large ResNet model in AWS using PyTorch.
2. **Optimization (Crucial):** Factory cameras use low-power NVIDIA Jetson Nano chips. We export the PyTorch model to ONNX, then use `TensorRT` on a Jetson-equivalent CI/CD runner to compile the model into a hardware-specific `.engine` file, applying INT8 quantization to shrink the model from 100MB to 25MB.
3. **OTA Deployment:** We use AWS IoT Greengrass. The CI pipeline pushes the `.engine` file and a Python inference script as a "Greengrass Component". The cameras securely download the update via MQTT when they have internet.
4. **Offline Inference & Telemetry:** The Python script runs inferences locally at 30 FPS. It queues metrics (e.g., "Defect detected") in a local SQLite DB. When the internet connects, it flushes the SQLite queue back to AWS IoT Core for monitoring.

## Q9: Design a Fraud Detection system.
**Context:** E-commerce site needing to block stolen credit cards. ML models are accurate, but take 100ms. We needed to process some obvious fraud in <10ms.
**Architecture & Example (Two-Tier System):**
1. **Tier 1 (Fast Path - Rules Engine):** Every transaction hits a Redis-backed rules engine first. E.g., `IF user_country != card_country AND ip_address_is_vpn: BLOCK`. This takes 2ms and catches 60% of obvious fraud.
2. **Tier 2 (Slow Path - ML Model):** If Tier 1 is unsure, it forwards the payload to an XGBoost model deployed on KServe. The model pulls 150 complex features (e.g., velocity of spend) and scores the transaction.
3. **Human in the Loop:** If the XGBoost model outputs a probability between 0.60 and 0.80 (the "gray area"), the transaction is sent to a Kafka queue read by human fraud analysts. The analysts' final decisions are fed back into the training data to improve the model tomorrow.

## Q10: Design a Recommendation System.
**Context:** Netflix-style video recommendations from a catalog of 10 million videos.
**Architecture & Example (Two-Tower Model):**
Running a heavy deep neural network on 10 million videos per user request would take hours. You must use a funnel approach.
1. **Candidate Generation (Retrieval):** We use a Vector Database (Pinecone/Milvus). Offline, we embed all 10M videos into 256-dimensional vectors. When a user logs in, we take their "user embedding" and do an Approximate Nearest Neighbor (ANN) search in the Vector DB to find the top 500 closest videos. This takes ~20ms.
2. **Ranking (Scoring):** We pass those 500 candidates to a heavy, highly accurate Deep Learning model (e.g., DLRM). This model computes the exact probability that the user will click each of the 500 videos, sorting them from 1 to 500. This takes ~50ms.
3. **Caching:** To save compute, we run this pipeline offline for active users nightly using Spark, and cache the top 50 video IDs in a Redis cluster `user_123: [vid_A, vid_B]`. The web app just queries Redis.

*(This is 10 highly detailed examples. I will expand the remaining ones sequentially based on your priority).*
