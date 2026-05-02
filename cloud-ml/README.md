# Cloud MLOps - Interview Questions

## Q1: Compare deploying models on managed services (SageMaker/Vertex AI) versus managing your own K8s cluster (EKS/GKE). When would you choose which?

**Real-world context:**
I've migrated a team from self-managed EKS to SageMaker to save engineering time, and later migrated a different team from Vertex AI back to GKE because managed service costs were eating our margins.

**Answer:**

**Managed Services (SageMaker / Vertex AI):**
*   **Pros:** Zero ops overhead for underlying infra. Out-of-the-box model registries, endpoint autoscaling, A/B testing, and pre-built containers for PyTorch/XGBoost.
*   **Cons:** Vendor lock-in. Very difficult to debug when things fail. Cost is typically a 20-40% premium over raw compute.
*   **When to choose:** When you have a small MLOps team (1-2 people), lots of data scientists, and you need to move fast. If your core business isn't infrastructure, pay AWS/GCP to handle it.

**Self-Managed K8s (EKS / GKE + KServe / Seldon):**
*   **Pros:** Complete control. You can run custom sidecars. You can use Spot instances aggressively for training. No vendor lock-in.
*   **Cons:** High operational burden. You have to manage the GPU drivers, Istio, cluster autoscaling, and monitoring.
*   **When to choose:** When you are operating at scale (10+ high-traffic models), require specialized hardware topologies, or need tight integration with existing enterprise microservices that already run on K8s.

**Trade-offs & Cost:**
The breakeven point is usually around $10k-$20k/month in compute spend. Below that, managed is cheaper; above that, self-managed saves money.

---

## Q2: How do you optimize costs when training large models on AWS/GCP?

**Real-world context:**
Our monthly AWS bill for ML training hit $80,000. Finance asked me to cut it in half without slowing down the data science team.

**Answer:**

Cost optimization in cloud ML comes down to three main levers: Compute type, Storage, and Job Efficiency.

**1. Spot Instances (The biggest win):**
*   **Strategy:** I switched 90% of our training jobs to use Spot/Preemptible instances.
*   **Implementation:** Spot instances can be terminated with 2 minutes' notice. To survive this, the training code *must* implement checkpointing. We save model weights to S3 every epoch. If the spot instance dies, the next run downloads the checkpoint and resumes.
*   **Result:** 60-70% savings on EC2/Compute costs immediately.

**2. Right-sizing and GPU utilization:**
*   **Implementation:** I implemented Datadog monitoring on GPU utilization. I enforced rules: use cheaper T4/V100 instances for experimentation, and only use A100s for large-scale distributed training runs where utilization is proven to be >80%.

**3. Storage Costs (The hidden killer):**
*   **Implementation:** Implemented S3 lifecycle policies. Move training datasets to infrequent access (IA) after 30 days. Delete intermediate model checkpoints after 7 days.

---

## Q3: Explain how you set up CI/CD for SageMaker or Vertex AI using GitHub Actions.

**Real-world context:**
We needed to bridge the gap between our standard software engineering practices (GitHub, PRs) and AWS SageMaker. 

**Answer:**

**Architecture:**
We treat the ML pipeline definition as code, and the model artifact as the compiled binary.

1.  **Code Commit:** Data scientist merges a PR containing updated feature engineering logic or model architecture (e.g., `train.py`).
2.  **GitHub Actions (CI):**
    *   Lints code, runs unit tests on data transformations.
    *   Authenticates to AWS via OIDC (no long-lived access keys).
    *   Builds a custom Docker image for training (if required) and pushes to ECR.
3.  **Pipeline Execution (CD - Training):**
    *   The GitHub Action uses Boto3 to trigger a SageMaker Pipeline.
    *   SageMaker provisions compute, runs `train.py`, evaluates, and registers it in the Model Registry.
4.  **Deployment (CD - Inference):**
    *   A separate GitHub Action is triggered when the model status changes to "Approved".
    *   It updates the SageMaker Endpoint via CloudFormation/Terraform.

**Common Mistakes:**
*   **Deploying directly from CI:** For mission-critical models, CI should only register the model and update IaC. The actual deployment should go through a GitOps flow or require manual approval in the CD tool.

---

## Q4: How do you handle cloud vendor quota limits and IP address exhaustion during massive ML scaling events?

**Real-world context:**
A team triggered a hyperparameter sweep requesting 200 GPU instances across 3 AWS Availability Zones. The training script failed silently for 45 minutes before crashing entirely. 

**Answer:**

When you operate ML at cloud scale, you hit physical data center limits that software engineers rarely see.

**1. Instance Quota Limits:**
*   **The Problem:** AWS/GCP do not give you unlimited GPUs. If you request 200 A100s, you will get an `InsufficientInstanceCapacity` error because that specific data center literally doesn't have them available, or you hit your account's vCPU quota.
*   **The Fix:** I implement cross-region or cross-AZ fallback in our orchestrator. If `us-east-1a` fails to provision a GPU within 5 minutes, the training job automatically falls back to requesting capacity in `us-east-1b` or even `us-west-2`. I also configure the orchestrator to automatically retry the provisioning step with exponential backoff rather than immediately failing the pipeline.

**2. IP Address Exhaustion (VPC CNI):**
*   **The Problem:** In AWS EKS, every K8s pod consumes a real IP address from your VPC subnet. When spinning up hundreds of distributed training worker pods, we completely exhausted our /24 subnet (only 254 IPs). No new pods could start.
*   **The Fix:** We attached secondary CIDR blocks to the VPC specifically for the ML subnets, expanding them to a /16. We also configured the AWS VPC CNI to use "Custom Networking," routing pod IPs into a non-routable 100.64.0.0/10 space so we don't consume the company's internal routable IP space for ephemeral training jobs.

---

## Q5: How do you build a cloud-agnostic ML platform to avoid vendor lock-in?

**Real-world context:**
The CTO mandated a multi-cloud strategy (AWS for general compute, GCP for TPUs). I was asked to ensure our MLOps platform could deploy to both seamlessly.

**Answer:**

To be truly cloud-agnostic, you have to sacrifice the convenience of managed services (SageMaker/Vertex) and build entirely on Kubernetes and open-source APIs.

**The Stack:**
1. **Compute Layer:** EKS on AWS, GKE on GCP. Both configured via Terraform.
2. **Object Storage (The tricky part):** S3 API is the de facto standard. We use the `boto3` or `s3fs` libraries natively, and on GCP, we configure GCS to use its S3-interoperability API. This allows the exact same Python training code to read/write data regardless of the cloud.
3. **Model Registry & Tracking:** Self-hosted MLflow running on K8s. The backend store is a managed PostgreSQL DB (RDS or Cloud SQL), and the artifact store is S3/GCS.
4. **Feature Store:** Feast. Feast abstracts the underlying infrastructure. A feature definition in Feast can materialize to Redis on AWS, or to Memorystore on GCP, without changing the data scientist's code.
5. **Model Serving:** KServe. Deploying a model is just applying a K8s YAML manifest. `kubectl apply -f model.yaml` works exactly the same on EKS as it does on GKE.

**The Trade-off:**
Building a cloud-agnostic platform requires a dedicated Platform Engineering team to maintain MLflow, Feast, KServe, and K8s across multiple clouds. The cognitive load is immense. I usually advise against this unless the company has a massive budget and a strict regulatory requirement for multi-cloud redundancy.

---
*[Back to README](../README.md)*
