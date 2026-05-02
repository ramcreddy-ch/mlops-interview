# Cloud MLOps - Interview Questions

## Q1: Compare deploying models on managed services (SageMaker/Vertex AI) versus managing your own K8s cluster (EKS/GKE). When would you choose which?

**Real-world context:**
I've migrated a team from self-managed EKS to SageMaker to save engineering time, and later migrated a different team from Vertex AI back to GKE because managed service costs were eating our margins.

**Answer:**

**Managed Services (SageMaker / Vertex AI):**
*   **Pros:** Zero ops overhead for underlying infra. Out-of-the-box model registries, endpoint autoscaling, A/B testing, and pre-built containers for PyTorch/XGBoost.
*   **Cons:** Vendor lock-in. Very difficult to debug when things fail (e.g., if a SageMaker endpoint fails to provision, you just get an opaque error). Cost is typically a 20-40% premium over raw compute.
*   **When to choose:** When you have a small MLOps team (1-2 people), lots of data scientists, and you need to move fast. If your core business isn't infrastructure, pay AWS/GCP to handle it.

**Self-Managed K8s (EKS / GKE + KServe / Seldon):**
*   **Pros:** Complete control. You can run custom sidecars (e.g., Datadog agent, custom feature fetchers). You can use Spot instances aggressively for training. No vendor lock-in.
*   **Cons:** High operational burden. You have to manage the GPU drivers, NVIDIA device plugins, Istio/Knative (for KServe), cluster autoscaling, and monitoring.
*   **When to choose:** When you are operating at scale (10+ high-traffic models), require specialized hardware topologies (e.g., specific NVLink setups for distributed training), or need tight integration with existing enterprise microservices that already run on K8s.

**Trade-offs & Cost:**
An ml.c5.xlarge instance on SageMaker costs about 25% more than the equivalent c5.xlarge EC2 instance. However, managing EKS requires paying a platform engineer. The breakeven point is usually around $10k-$20k/month in compute spend. Below that, managed is cheaper; above that, self-managed saves money.

---

## Q2: How do you optimize costs when training large models on AWS/GCP?

**Real-world context:**
Our monthly AWS bill for ML training hit $80,000. Finance asked me to cut it in half without slowing down the data science team.

**Answer:**

Cost optimization in cloud ML comes down to three main levers: Compute type, Storage, and Job Efficiency.

**1. Spot Instances (The biggest win):**
*   **Strategy:** Training jobs are inherently batch workloads. I switched 90% of our training jobs (SageMaker and EKS) to use Spot/Preemptible instances.
*   **Implementation:** Spot instances can be terminated with 2 minutes' notice. To survive this, the training code *must* implement checkpointing. We save model weights and optimizer states to S3 every epoch. If the spot instance dies, the next run downloads the checkpoint and resumes.
*   **Result:** 60-70% savings on EC2/Compute costs immediately.

**2. Right-sizing and GPU utilization:**
*   **Strategy:** Data scientists request A100s when they only need T4s.
*   **Implementation:** I implemented Datadog monitoring on GPU utilization (`nvidia_smi.gpu_utilization`). I found many jobs had 15% GPU utilization because they were bottlenecked on CPU data loading. I enforced rules: use cheaper T4/V100 instances for experimentation, and only use A100s for large-scale distributed training runs where utilization is proven to be >80%.

**3. Storage Costs (The hidden killer):**
*   **Strategy:** S3/GCS API costs and storage costs balloon when saving thousands of checkpoints.
*   **Implementation:** Implemented S3 lifecycle policies. Move training datasets to infrequent access (IA) after 30 days. Delete intermediate model checkpoints after 7 days, keeping only the final trained artifact.

**Trade-offs:**
Using Spot instances increases the wall-clock time of training jobs slightly due to interruptions and restarts. For hyper-critical, time-sensitive training (e.g., retraining a fraud model during an active attack), we override the default and use On-Demand instances.

---

## Q3: Explain how you set up CI/CD for SageMaker or Vertex AI using GitHub Actions.

**Real-world context:**
We needed to bridge the gap between our standard software engineering practices (GitHub, PRs) and AWS SageMaker. Data scientists were training models in SageMaker Studio and manually deploying endpoints via the AWS console.

**Answer:**

**Architecture:**
We treat the ML pipeline definition as code, and the model artifact as the compiled binary.

1.  **Code Commit:** Data scientist merges a PR containing updated feature engineering logic or model architecture (e.g., `train.py`, `pipeline.py`).
2.  **GitHub Actions (CI):**
    *   Lints code, runs unit tests on data transformations.
    *   Authenticates to AWS via OIDC (OpenID Connect - no long-lived access keys).
    *   Builds a custom Docker image for training (if required) and pushes to ECR.
3.  **Pipeline Execution (CD - Training):**
    *   The GitHub Action uses the AWS CLI/Boto3 to trigger a SageMaker Pipeline.
    *   SageMaker provisions compute, downloads data from S3, runs `train.py`, evaluates the model, and registers it in the SageMaker Model Registry.
4.  **Deployment (CD - Inference):**
    *   A separate GitHub Action is triggered when the model status changes to "Approved" in the SageMaker Model Registry.
    *   This action calls CloudFormation/Terraform to update the SageMaker Endpoint configuration, pointing it to the new model artifact on S3.
    *   It executes a blue/green deployment strategy to shift traffic safely.

**Common Mistakes:**
*   **Hardcoding data paths:** Scripts should take S3 paths as environment variables or arguments, allowing CI/CD to inject paths for dev vs. prod datasets.
*   **Deploying directly from CI:** For mission-critical models, CI should only register the model and update IaC. The actual deployment should go through a GitOps flow or require manual approval in the CD tool (like ArgoCD or Jenkins).

---
*[Back to README](../README.md)*
