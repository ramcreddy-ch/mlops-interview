# System Design - Interview Questions

## Q1: Design a real-time ML inference system using Kafka + Kubernetes.
**Context:** Fraud detection at 8K TPS, P99 < 80ms SLA.
**Architecture:** Kafka (64 partitions) -> Flink (feature enrichment) -> K8s Inference Service (12 replicas, HPA on RPS) -> Decision Service.
**Failures:** Redis feature store timeout. Fix: Strict 50ms circuit breaker.

## Q2: Design a scalable model deployment platform.
**Context:** Data scientists were manually deploying scripts.
**Architecture:** CLI (`ml-deploy`) -> MLflow Registry -> Jenkins CI/CD -> KServe on K8s -> Datadog.
**Key Decision:** Bake model into Docker image for deterministic deployments vs loading at runtime from S3.

## Q3: Design a feature store handling batch and streaming.
**Context:** Needed historical aggregates and real-time signals.
**Architecture:** Batch: S3 -> Spark -> Parquet -> Redis. Stream: Kafka -> Flink -> Redis. Feast manages retrieval.
**Hard Part:** Point-in-time correctness to prevent data leakage during training.

## Q4: Design a multi-tenant ML platform on Kubernetes.
**Context:** Three teams sharing a GPU cluster.
**Architecture:** K8s Namespaces + ResourceQuotas + NetworkPolicies + ServiceAccounts.
**GPU Sharing:** Time-slicing for dev environments, MIG for training, dedicated for prod inference.

## Q5: Design a multi-region, high-availability ML inference architecture.
**Context:** Global image classification API with <200ms latency.
**Architecture:** Route 53 Latency Routing -> ALB -> EKS (US/EU/Asia). 
**Sync:** Docker images pushed to global ECR. Features replicated via Redis Active-Active.

## Q6: Design an automated A/B testing pipeline for ML models.
**Context:** Testing 5 recommendation models concurrently.
**Architecture:** API Gateway -> A/B Routing Service -> Inference Clusters.
**Crucial:** Hashing function `hash(user_id) % 100` for deterministic routing. Never use round-robin.

## Q7: Design a shadow deployment architecture.
**Context:** Testing a high-risk financial model.
**Architecture:** Istio VirtualService mirrors 100% of traffic to the shadow pod. Responses from shadow are logged to Kafka but dropped. 
**Benefit:** Measures real latency and distribution without user impact.

## Q8: Design an Edge ML deployment system.
**Context:** Deploying models to 10,000 IoT cameras.
**Architecture:** Cloud Training -> ONNX export -> TensorRT compilation (target hardware) -> OTA Update via MQTT -> Local Edge runtime.
**Challenge:** Intermittent connectivity. Models must run fully offline and queue telemetry locally.

## Q9: Design a Fraud Detection system.
**Context:** E-commerce transaction scoring.
**Architecture:** Fast path (real-time Rules Engine) + ML path (XGBoost on KServe).
**Challenge:** Class imbalance and adversarial evasion. Requires continuous daily retraining and human-in-the-loop for borderline cases.

## Q10: Design a Recommendation System.
**Context:** Netflix-style video recommendations.
**Architecture:** Two-tower model. 1. Candidate Generation (Vector DB retrieving top 100 from millions). 2. Ranking Model (heavy deep learning model scoring the top 100).
**Caching:** Pre-compute recommendations for active users nightly into Redis.

## Q11: Design an Ad Bidding system (RTB).
**Context:** Must bid on ad space in < 100ms.
**Architecture:** Strict memory-mapped DBs (Aerospike) + C++ inference. Python is too slow.
**Challenge:** Extreme latency constraints. Features must be locally cached in RAM.

## Q12: Design a Predictive Maintenance system.
**Context:** Monitoring factory machines.
**Architecture:** IoT sensors -> Kafka -> Flink windowing -> Time-series model (LSTM/XGBoost).
**Challenge:** Handling out-of-order events and missing sensor data. Use Flink watermarks.

## Q13: Design a Search Ranking system.
**Context:** E-commerce product search.
**Architecture:** Elasticsearch (BM25 for text match) -> Learning to Rank (LTR) plugin or reranker service.
**Metrics:** NDCG (Normalized Discounted Cumulative Gain).

## Q14: Design a Dynamic Pricing system.
**Context:** Ride-sharing app.
**Architecture:** Real-time supply/demand features from Redis -> Model predicts multiplier.
**Challenge:** Feedback loops. If you price too high, demand drops, skewing the next training dataset.

## Q15: Design an ETA prediction system.
**Context:** Food delivery routing.
**Architecture:** Graph neural networks / tree-based models using geospatial features (H3 indexes).
**Data:** Massive traffic data updates every minute requiring efficient spatial joins.

## Q16: Design a massive image processing pipeline.
**Context:** 10M images uploaded daily requiring tagging.
**Architecture:** S3 Event -> SQS Queue -> K8s GPU Worker Pool (auto-scaled by KEDA based on queue depth).
**Optimization:** Batching images from SQS before sending to GPU.

## Q17: Design an OCR/Document parsing pipeline.
**Context:** Processing PDFs for text extraction.
**Architecture:** Textract/Tesseract -> NLP NER model.
**Challenge:** CPU-bound preprocessing (PDF rendering) starving the GPU inference. Separate preprocessing pods from inference pods.

## Q18: Design a Vector Database architecture.
**Context:** Semantic search for a knowledge base.
**Architecture:** Text -> Embedding Model (e.g., BERT) -> Pinecone/Milvus.
**Challenge:** HNSW index memory usage. Sharding vector DBs across nodes.

## Q19: Design an ML metrics aggregation pipeline.
**Context:** Tracking drift across 50 models.
**Architecture:** Inference pods emit StatsD metrics -> Datadog Agent -> Datadog backend.
**Challenge:** Cardinality explosion. Don't put user_id in metric tags.

## Q20: Design an Experiment Tracking system.
**Context:** Replacing disparate spreadsheets.
**Architecture:** Self-hosted MLflow on K8s + RDS PostgreSQL (metadata) + S3 (artifacts).
**Security:** Put MLflow behind an OAuth2 proxy (e.g., Okta) for RBAC.

## Q21: Design a Model Registry architecture.
**Context:** Cross-team model sharing.
**Architecture:** Centralized MLflow/Vertex Registry. 
**Governance:** Strict webhook validations preventing stage transitions without attached testing metrics.

## Q22: Design a Data Lakehouse for ML.
**Context:** Moving away from expensive Data Warehouses.
**Architecture:** S3 + Apache Iceberg/Delta Lake + Trino/Spark.
**Benefit:** Time-travel queries allow data scientists to reconstruct datasets exactly as they were 6 months ago.

## Q23: Design an MLOps CI/CD architecture.
**Context:** Full automation.
**Architecture:** Git -> Jenkins (CI) -> Airflow (CT) -> MLflow -> ArgoCD (CD).
**Decoupling:** Do not train models inside Jenkins. Use Jenkins to trigger orchestrators.

## Q24: Design a system to handle cyclical traffic spikes.
**Context:** Food delivery spikes at 12 PM and 6 PM.
**Architecture:** Predictive autoscaling. 
**Strategy:** HPA is reactive (scales after spike hits). Use KEDA with cron triggers to pre-warm the GPU cluster 15 minutes before the expected spike.

## Q25: Design a system to handle PII in ML.
**Context:** GDPR compliance.
**Architecture:** Cryptographic hashing in the data pipeline.
**Rule:** Raw PII never enters the feature store. Use salted hashes for join keys.

## Q26: Design a cross-account AWS deployment architecture.
**Context:** Dev, Staging, and Prod are in isolated AWS accounts.
**Architecture:** Model trained in Dev -> Artifact pushed to shared ECR/S3 -> Prod account assumes cross-account IAM role to pull the artifact for deployment.

## Q27: Design a Disaster Recovery plan for an ML platform.
**Context:** Entire region goes down.
**Architecture:** IaC (Terraform) replicates infra. 
**Crucial:** S3 cross-region replication for the Model Registry bucket. If you lose the trained weights, recreating them takes weeks.

## Q28: Design a batch scoring system for 1 Billion users.
**Context:** Nightly recommendation updates.
**Architecture:** Airflow triggers Spark job. 
**Optimization:** Use PySpark pandas UDFs to vectorize model inference across the Spark cluster.

## Q29: Design a system to handle sparse features in real-time serving.
**Context:** User vocabulary feature with 10M dimensions.
**Architecture:** Hashing trick or embedding tables.
**Serving:** Do not pass dense arrays over the network. Pass sparse indices and do the embedding lookup inside the inference container.

## Q30: Design an active learning loop.
**Context:** Labeling data is expensive.
**Architecture:** Model scores unlabelled data -> Pushes examples with *lowest* confidence scores (e.g., probability near 0.5) to a human annotation queue -> Retrain.
