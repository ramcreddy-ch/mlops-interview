# Cost Optimization for ML - Interview Questions

## Q1: How do you control infrastructure costs for ML training on Kubernetes?
**Context:** AWS bill jumped 40% due to hyperparameter sweeps.
**Answer:** Move all training jobs to Spot/Preemptible nodes. This provides a ~70% discount. Ensure training code checkpoints to S3 frequently so it can resume if the Spot instance is terminated.

## Q2: How do you optimize serving costs for bursty ML traffic?
**Context:** GPU nodes idling overnight.
**Answer:** Do not use Spot for real-time serving (causes 503 errors). Use Reserved Instances (1-year commit) for the baseline traffic. Use KEDA to "Scale to Zero" during off-hours, allowing Cluster Autoscaler to terminate the idle nodes.

## Q3: Batch vs Real-time inference: What are the cost implications?
**Context:** PM wants real-time predictions for a weekly email.
**Answer:** Real-time requires 24/7 Highly Available compute and a costly Redis feature store. Batch runs ephemerally (e.g., 2 hours on Spot instances) at 100% utilization. Always push for Batch or Micro-batch unless sub-second latency is a hard business requirement.

## Q4: How do you attribute ML costs back to specific teams or models?
**Context:** Finance needs ROI per data science team.
**Answer:** Enforce K8s labels (`team`, `model`). Deploy Kubecost to map AWS billing data back to specific pods based on GPU request time. Build a Grafana dashboard showing "Daily Inference Spend by Model".

## Q5: How do you prevent resource fragmentation on K8s?
**Context:** Devs requesting 1 entire A100 GPU for a 2GB model.
**Answer:** Implement NVIDIA Time-Slicing or MIG (Multi-Instance GPU) to partition the GPU. Pack multiple small inference services onto a single physical GPU, drastically increasing utilization.

## Q6: How do you reduce S3/Storage costs in ML?
**Context:** 5PB of old training data and checkpoints costing thousands.
**Answer:** Implement S3 Lifecycle Policies. Transition raw datasets to S3 Glacier/Infrequent Access after 30 days. Automatically delete intermediate training checkpoints (keeping only the final model) after 7 days.

## Q7: Optimizing Cloud Egress Costs.
**Context:** Massive egress fees moving data from AWS to GCP.
**Answer:** Keep training compute in the same cloud and region as the data gravity. If using multi-cloud, use AWS Direct Connect. Ensure training pods pull data from S3 using VPC Endpoints to avoid traversing the public NAT Gateway.

## Q8: Right-sizing EC2 instances for ML.
**Context:** Using expensive GPU instances for data prep.
**Answer:** Split pipelines. Run the Pandas/Spark data preprocessing step on cheap, memory-optimized CPU instances (R5). Only spin up the GPU instance (P4) for the actual matrix multiplication training phase.

## Q9: How do you handle Zombie ML Resources?
**Context:** Forgotten Jupyter notebooks running on A100s for weeks.
**Answer:** Implement a Reaper script. Scan K8s namespaces for Jupyter pods with 0% GPU utilization over the last 48 hours. Automatically scale their deployment to 0 and send a Slack notification to the owner.

## Q10: FinOps: How do you forecast ML infrastructure costs?
**Context:** Budgeting for a new deep learning project.
**Answer:** Estimate: (Hours to train 1 epoch) * (Total Epochs) * (Cost per GPU instance hour). Add 20% buffer for hyperparameter tuning. For inference: (Expected TPS / Max TPS per Pod) * (Cost per Pod instance).

## Q11: Using AWS Inferentia / Graviton for Cost Savings.
**Context:** Reducing reliance on NVIDIA GPUs.
**Answer:** Migrate CPU-bound ML (XGBoost) to ARM-based Graviton instances for a 20% price-performance gain. Migrate Deep Learning inference to AWS Inferentia chips for a massive cost reduction compared to standard T4/A10G GPUs.

## Q12: How do you optimize Feature Store costs?
**Context:** Redis cluster costs exceeding the ML compute costs.
**Answer:** Only put strictly necessary real-time features in Redis. Serve historical, slowly-changing features directly from an offline store or a cheaper DB like PostgreSQL. Implement aggressive TTLs (Time-to-Live) so stale features are evicted from RAM.

## Q13: Cost implications of Deep Learning vs Traditional ML.
**Context:** Deciding model architecture.
**Answer:** XGBoost/LightGBM runs on CPUs (cheap), trains fast, and is easily interpretable. Deep Learning requires GPUs (expensive). Unless dealing with unstructured data (images/text) or massive datasets, default to tree-based models to save money.

## Q14: Optimizing API Gateway costs for ML.
**Context:** Paying high fees for AWS API Gateway on high-throughput ML models.
**Answer:** API Gateway charges per million requests. For an internal microservice doing 10k TPS, API Gateway is too expensive. Use Application Load Balancers (ALB) or internal K8s ingress (NGINX/Istio) which charge by hourly usage/bandwidth, not per request.

## Q15: How do you enforce ML cost budgets?
**Context:** Junior DS spent $5k over the weekend.
**Answer:** Use AWS Budgets to trigger alerts at 50%, 80%, and 100% of the team's monthly limit. Integrate an OPA Gatekeeper policy in K8s that rejects jobs if the namespace has exceeded its predefined resource quota.

## Q16: Optimizing Distributed Training network costs.
**Context:** High costs for cross-AZ network traffic during training.
**Answer:** Distributed training requires massive node-to-node communication. Ensure all training worker nodes are deployed in the *same* Availability Zone using an EC2 Placement Group to completely eliminate cross-AZ network transfer fees.

## Q17: Cost vs Latency trade-off in batching.
**Context:** Triton Inference Server configuration.
**Answer:** Dynamic batching increases throughput (saving money on GPUs) but increases latency (waiting for the batch to fill). You must tune `max_queue_delay_microseconds` to find the exact breakeven point where SLA is met at the lowest cost.

## Q18: Optimizing Docker Image storage costs.
**Context:** ECR storage costs ballooning.
**Answer:** ML images are huge (5GB+). If CI/CD builds a new image on every commit, costs explode. Set an ECR Lifecycle Policy to keep only the last 10 tagged images, and delete untagged/orphaned images after 14 days.

## Q19: Using Spot Fleets / Mixed Instance Groups.
**Context:** Spot instances getting terminated frequently.
**Answer:** Don't rely on a single instance type (e.g., only `p3.2xlarge`). Configure an ASG Mixed Instance Group to pull from multiple pools (`p3.2xlarge`, `g4dn.xlarge`). If one pool runs out of Spot capacity, AWS automatically fulfills from another pool.

## Q20: Optimizing Data Labeling costs.
**Context:** Paying humans $100k to label images.
**Answer:** Implement Active Learning. Use a baseline model to score the unlabelled data. Only send the images where the model is highly uncertain (probability near 50%) to the expensive human labelers. Auto-label the highly confident ones.

## Q21: Cost of Model Monitoring (Datadog/ELK).
**Context:** Logging 100% of predictions doubled the Datadog bill.
**Answer:** ML logs are high volume. Do not send raw prediction JSONs to Datadog. Send them to S3 (cheap). Only send aggregated metrics (averages, error counts) to Datadog.

## Q22: Should you compress ML models?
**Context:** Model is 4GB, requires expensive instances.
**Answer:** Yes. Use Quantization (FP32 to INT8) via TensorRT or PyTorch native. This shrinks the model size by 4x, allowing you to run it on cheaper instances with less VRAM, often with negligible accuracy loss.

## Q23: Avoiding Cloud NAT Gateway charges.
**Context:** Training nodes downloading terabytes from public PyPI/HuggingFace.
**Answer:** NAT Gateways charge per GB processed. Use a private JFrog Artifactory inside your VPC to cache Python packages. Use VPC Endpoints for AWS services to avoid the NAT entirely.

## Q24: Cost of High Availability (HA) in MLOps.
**Context:** Running 3 redundant feature stores.
**Answer:** HA doubles or triples costs. Does a churn prediction model really need 99.99% uptime across 3 AZs? Tier your services. Tier 1 (Fraud/Payments) gets HA. Tier 3 (Internal analytics) runs in a single AZ with acceptable downtime.

## Q25: Optimizing CI/CD compute costs.
**Context:** GitHub Actions burning minutes on long ML builds.
**Answer:** Use Self-Hosted Runners deployed on Spot instances within your own AWS account. You only pay the raw EC2 Spot price, bypassing GitHub's per-minute premium billing.

## Q26: Managing unused K8s Persistent Volumes (PVs).
**Context:** Orphaned EBS volumes costing $1k/month.
**Answer:** When data scientists delete a Jupyter pod, the PVC sometimes remains. Run a weekly script to find and delete "Unbound" or orphaned PVs.

## Q27: Evaluating managed ML platform costs (e.g., Databricks vs EMR).
**Context:** Choosing a Spark platform for feature engineering.
**Answer:** Databricks charges a DBU (Databricks Unit) premium on top of raw AWS compute. If the team uses the collaborative notebooks and MLflow integrations heavily, it's worth it. If they just run scheduled batch jobs, raw EMR is significantly cheaper.

## Q28: How do you identify inefficient ML code?
**Context:** Python script taking 10 hours to process data.
**Answer:** Inefficient Python code burns expensive compute time. Use profilers (`cProfile`). Often, a simple Pandas `apply()` loop can be replaced with a vectorized NumPy operation, dropping runtime from hours to minutes.

## Q29: Optimizing Kafka costs for ML streams.
**Context:** Storing 7 days of ML feature streams in MSK.
**Answer:** Kafka storage is expensive. Reduce the retention period to 24 hours. Use Kafka Connect to sink older data into S3/Parquet for cheap long-term storage and historical replay.

## Q30: The hidden cost of Technical Debt in MLOps.
**Context:** Why invest in automation?
**Answer:** Manual deployments take 2 engineers 3 days. Automated CI/CD takes 5 minutes. The cost of an MLOps platform engineer is high, but the ROI is massive when you eliminate the manual toil of 50 data scientists.
