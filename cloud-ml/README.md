# Cloud MLOps - Interview Questions

## Q1: SageMaker vs Self-Managed EKS. When do you choose which?
**Context:** Deciding platform architecture for a new ML team.
**Answer:** SageMaker: Zero infra ops, great for small teams moving fast. Cons: Expensive (+30% premium), vendor lock-in. Self-Managed EKS: Complete control, aggressive spot instance usage, cheaper at scale (>10 models). Cons: Requires dedicated platform engineers.

## Q2: How do you optimize costs when training large models on AWS/GCP?
**Context:** $80k monthly AWS bill.
**Answer:** 1. Move 90% of training to Spot/Preemptible instances (requires checkpointing logic). 2. Right-size GPUs (use T4 for dev, A100 only for prod distributed training). 3. S3 Lifecycle rules (delete old checkpoints, move datasets to Infrequent Access).

## Q3: How do you handle AWS/GCP instance quota limits during massive scaling?
**Context:** Hyperparameter sweep failed requesting 200 A100s.
**Answer:** Cloud providers don't have infinite GPUs. Implement cross-AZ or cross-region fallback in the orchestrator. If `us-east-1` fails `InsufficientInstanceCapacity`, automatically retry in `us-west-2`.

## Q4: How do you prevent IP Address exhaustion during distributed training on EKS?
**Context:** EKS exhausted the /24 subnet, blocking new pods.
**Answer:** Use AWS VPC CNI "Custom Networking". Route ephemeral training pod IPs into a non-routable 100.64.0.0/10 CIDR block so they don't consume the company's internal routable IP space.

## Q5: How do you build a cloud-agnostic ML platform?
**Context:** Multi-cloud mandate (AWS + GCP).
**Answer:** Abandon managed services. Use EKS/GKE + Terraform. Use S3 interoperability API for object storage. Use Feast for Feature Store (abstracts Redis/Memorystore). Use MLflow for tracking. Deploy via KServe.

## Q6: Explain SageMaker Pipeline CI/CD via GitHub Actions.
**Context:** Automating model training.
**Answer:** GitHub Action lints code -> Authenticates via OIDC -> Triggers SageMaker Pipeline using Boto3 -> SageMaker provisions compute, trains, and registers model. CD pipeline updates CloudFormation to deploy the approved model.

## Q7: What are the trade-offs of using Vertex AI Feature Store?
**Context:** Evaluating GCP ML services.
**Answer:** Pros: Fully managed, handles point-in-time correctness, great BigQuery integration. Cons: Expensive online serving costs. If your latency requirement is < 5ms and TPS is 10k+, managing your own Redis cluster on GKE is far cheaper.

## Q8: How do you secure data in transit and at rest in Cloud ML?
**Context:** Handling PHI (healthcare) data.
**Answer:** At rest: S3 buckets encrypted with KMS Customer Managed Keys. In transit: Enforce TLS 1.2+ on all endpoints. Use VPC Endpoints (AWS PrivateLink) so training nodes pull data from S3 over the internal AWS network, never traversing the public internet.

## Q9: How do you handle Cross-Account model deployments in AWS?
**Context:** Dev, Staging, and Prod are isolated AWS accounts.
**Answer:** Train in Dev. Push artifact to a centralized shared S3 bucket. The Prod account IAM role assumes a cross-account trust policy allowing it to read from the central S3 bucket and pull the model for deployment.

## Q10: GCP TPU vs AWS Inferentia. How do you choose?
**Context:** Optimizing deep learning costs.
**Answer:** TPUs (GCP): Incredible for massive scale distributed training of Transformers (due to torus network topology). AWS Inferentia (Inf2): Incredible cost-to-performance ratio for *inference* specifically, but requires compiling models with AWS Neuron SDK.

## Q11: How do you handle persistent storage for stateful ML workloads in the cloud?
**Context:** Running a massive Redis feature store.
**Answer:** Do not use local instance store (ephemeral). Use EBS io2 Block Express for high IOPS, or transition to fully managed ElastiCache/MemoryDB to offload persistence and backup responsibilities.

## Q12: Explain the role of AWS IAM permissions boundaries for Data Scientists.
**Context:** Preventing privilege escalation.
**Answer:** Data scientists need broad permissions to experiment. A permissions boundary is a hard limit. E.g., they can create any IAM role for their SageMaker jobs, *but* the boundary dictates those roles can never access the `finance-prod` S3 bucket.

## Q13: How do you optimize S3 data loading for PyTorch?
**Context:** GPU utilization was 20% due to slow S3 reads.
**Answer:** Do not use `boto3.get_object` in a loop. Use `s3fs` or PyTorch `webdataset` for concurrent streaming. Alternatively, use Amazon FSx for Lustre linked to S3 for POSIX-compliant, sub-millisecond access.

## Q14: What is the purpose of SageMaker Model Monitor?
**Context:** Tracking drift without writing custom code.
**Answer:** It automatically intercepts prediction requests and responses (Data Capture), stores them in S3, and runs hourly Spark jobs to compare baseline statistics (from training) against live data to detect data drift.

## Q15: How do you manage infrastructure as code for ML platforms?
**Context:** Standardizing deployments across regions.
**Answer:** Use Terraform. Define EKS clusters, S3 buckets, IAM roles, and VPCs in HCL. Use Terragrunt to manage environments (Dev/Staging/Prod) to keep configurations DRY (Don't Repeat Yourself).

## Q16: Troubleshooting a SageMaker Endpoint creation failure.
**Context:** Endpoint stuck in "Creating" for 45 minutes then failed.
**Answer:** 1. Check CloudWatch logs for the inference container. 2. Usually caused by the model taking too long to load into memory (exceeding the 60-second ping health check). 3. Fix: increase the `ContainerStartupHealthCheckTimeoutInSeconds`.

## Q17: How to perform A/B testing natively in Cloud Managed Services?
**Context:** Comparing two models in GCP Vertex AI.
**Answer:** Use Vertex AI Traffic Splitting. Define multiple models under a single Endpoint. Configure rules to route 90% of traffic to Model A and 10% to Model B natively without needing to manage Istio or custom API gateways.

## Q18: What is Cloud Cost Anomaly Detection and why is it critical for ML?
**Context:** A rogue hyperparameter tuning job cost $10k over the weekend.
**Answer:** Standard billing alerts are static. Anomaly detection uses ML (ironically) to learn your normal spend patterns. It sends a Slack alert if AWS spend jumps 400% in a 2-hour window, catching rogue jobs instantly.

## Q19: Using Spot Instances with EKS Managed Node Groups.
**Context:** Automating spot interruptions.
**Answer:** Configure the EKS Node Group to use Spot. Crucially, install the AWS Node Termination Handler. It listens to EC2 metadata for the 2-minute interruption notice and gracefully drains the pods to other nodes before the instance dies.

## Q20: How do you handle secret rotation in cloud ML pipelines?
**Context:** Database password rotated, ML pipelines broke.
**Answer:** Use AWS Secrets Manager with automatic rotation. In the ML pipeline, do not fetch the secret at pipeline start. Let the K8s pods fetch the secret dynamically via External Secrets Operator so they always get the latest version.

## Q21: What are AWS Macie and GCP DLP used for in MLOps?
**Context:** Ensuring PII doesn't leak into training data.
**Answer:** They are automated data privacy services. Run Macie/DLP jobs across your S3/GCS data lakes to scan for unmasked credit cards, SSNs, or emails. If found, trigger an alert and block the dataset from being used in training.

## Q22: Designing a serverless ML architecture.
**Context:** Low-traffic, highly bursty model.
**Answer:** Use AWS Lambda + EFS (for the model weights, since Lambda has a 10GB container limit). Cold starts (loading the model into memory) will be 10-20 seconds. Use Provisioned Concurrency to keep a few Lambdas "warm" for low latency.

## Q23: How do you optimize Docker images for Cloud Container Registries (ECR/GCR)?
**Context:** 8GB Docker images taking too long to pull.
**Answer:** 1. Use multi-stage builds. 2. Put dependencies in lower layers, application code in higher layers to utilize Docker caching. 3. Use tools like `dive` to find wasted space. 4. Never put model `.pt` weights in the image; load them from S3 at runtime.

## Q24: Explain GCP's BigQuery ML and its use cases.
**Context:** Data analysts want to build models without Python.
**Answer:** BQML allows training XGBoost/Linear models directly using SQL queries inside BigQuery. It's incredibly fast for tabular data because the data never leaves the data warehouse. Great for baselines, bad for custom deep learning.

## Q25: How do you use AWS CloudTrail for ML compliance?
**Context:** Auditing who deployed a bad model.
**Answer:** CloudTrail logs every API call. We stream CloudTrail to Splunk/ELK. We can query: "Who called `UpdateEndpoint` on `FraudModel` at 2:00 AM?" providing full cryptographic non-repudiation for compliance audits.

## Q26: Managing Cloud Networking for Hybrid MLOps.
**Context:** Data is on-premise, ML compute is in AWS.
**Answer:** Set up AWS Direct Connect (dedicated fiber link). Data pipelines stream from on-prem DBs to S3 over Direct Connect to avoid public internet egress costs and ensure strict security compliance.

## Q27: How do you handle Cloud Provider outages?
**Context:** `us-east-1` goes down completely.
**Answer:** 1. Route53 health checks automatically failover traffic to `us-west-2`. 2. Inference EKS cluster in `us-west-2` autoscales to handle 2x traffic. 3. S3 Cross-Region Replication ensures the model artifacts are available in the west region.

## Q28: What is AWS Inferentia and AWS Trainium?
**Context:** Reducing reliance on NVIDIA.
**Answer:** Amazon's custom silicon chips. Trainium is for training, Inferentia for serving. They offer up to 50% cost-to-performance improvement over GPUs for specific models (like NLP Transformers), but require migrating code to use the AWS Neuron compiler.

## Q29: Using CloudWatch/Stackdriver vs Datadog for ML.
**Context:** Deciding on an observability stack.
**Answer:** CloudWatch/Stackdriver are native but basic. Datadog provides superior APM (Application Performance Monitoring) distributed tracing, better custom metrics aggregation, and easier cross-cloud visibility. We use Datadog for everything except raw infrastructure logs.

## Q30: How do you automate infrastructure provisioning for ML?
**Context:** Stood up a new team, took 3 weeks to configure AWS.
**Answer:** Use Terraform modules. Create a module `ml-environment` that stamps out: 1 S3 bucket, 1 EKS namespace, 1 IAM role, and associated policies. Provisioning a new team becomes a 5-minute PR merge.
