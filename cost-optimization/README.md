# Cost Optimization for ML - Interview Questions

## Q1: How do you control infrastructure costs for ML training and serving on Kubernetes?

**Real-world context:**
Our AWS EC2 bill jumped 40% in one month after the data science team started hyperparameter tuning on GPU nodes. I was tasked with bringing the cost down without impacting their delivery timelines.

**Answer:**

Cost optimization on K8s for ML requires attacking the problem at three levels: Compute type, Autoscaling, and Resource Allocation.

**1. Compute Type: Spot Instances (The 70% discount)**
*   **Training:** Training jobs are perfect for Spot instances because they are asynchronous and can survive interruptions *if* checkpointing is implemented. I moved all our Kubeflow/PyTorch training jobs to a Spot node pool. If a node is preempted, the job restarts on a new node and resumes from the last S3 checkpoint.
*   **Serving:** NEVER use Spot instances for real-time, synchronous inference. You will get 503 errors during node termination. For serving, we use Reserved Instances (1-year commit) which saves ~30% compared to On-Demand.

**2. Intelligent Autoscaling (Scale to Zero)**
*   **The Problem:** GPU nodes sitting idle overnight.
*   **The Fix (KEDA + Cluster Autoscaler):** We use KEDA (Kubernetes Event-driven Autoscaling) for our async batch inference workloads. When the Kafka queue is empty, KEDA scales the Deployment to 0 replicas. The Kubernetes Cluster Autoscaler then sees empty GPU nodes and terminates them. When messages arrive in Kafka, KEDA scales the deployment to 1, the Cluster Autoscaler requests a new GPU node from AWS, and processing begins.

**3. Resource Allocation & Right-sizing**
*   **The Problem:** Data scientists requesting `nvidia.com/gpu: 1` and `memory: 64Gi` for every pod, regardless of the model size, leading to massive cluster fragmentation and wasted capacity.
*   **The Fix:** 
    *   I implemented Datadog monitoring on actual pod utilization vs. requested limits.
    *   I enforced strict `LimitRanges` and `ResourceQuotas` per namespace.
    *   For smaller models (e.g., XGBoost), we strictly enforce CPU-only inference.
    *   For GPU inference, we use NVIDIA time-slicing to pack multiple smaller models onto a single A10G GPU, sharing the hardware rather than giving each model a dedicated GPU.

---

## Q2: Batch vs Real-time inference: What are the cost implications?

**Real-world context:**
A product manager requested a real-time recommendation endpoint. I analyzed the traffic and found that 95% of users only visit the site once a week.

**Answer:**

Real-time inference is exponentially more expensive than batch inference. I always advocate for batch inference unless sub-second latency is a hard business requirement.

**Cost breakdown:**

**Real-time Inference (High Cost):**
*   Requires 24/7 compute availability. Even at 3 AM, you need at least 2 pods running across 2 availability zones for high availability.
*   Requires a low-latency, highly available online Feature Store (like Redis Cluster), which adds significant infrastructure cost.
*   Requires complex load balancing, Istio/service mesh, and HPA overhead.
*   **Efficiency:** Often runs at 20-30% utilization due to the need to over-provision for traffic spikes.

**Batch Inference (Low Cost):**
*   Compute is ephemeral. You spin up a massive Spark cluster (using Spot instances), score 10 million users in 2 hours, write the predictions to a database (PostgreSQL/DynamoDB), and terminate the cluster.
*   No online Feature Store required. You query the data lake (S3/BigQuery) directly.
*   **Efficiency:** Runs at nearly 100% CPU/GPU utilization for the duration of the job, getting maximum value out of the hardware.

**The Compromise (Near real-time):**
If they need freshness but not millisecond latency, I build a micro-batch streaming pipeline using Flink or Spark Structured Streaming running every 15 minutes. It's cheaper than a REST endpoint and fresher than a nightly batch job.

---

## Q3: How do you track and attribute ML costs to specific models or teams?

**Real-world context:**
Finance asked me, "How much does the Fraud Detection model cost to run versus the Recommendation model?" because we needed to calculate the ROI of the data science teams. I couldn't answer them because everything was running in one giant EKS cluster.

**Answer:**

You cannot optimize what you cannot measure. K8s obfuscates costs because multiple models share the same underlying EC2 instances.

**My Cost Attribution Strategy:**

1.  **Strict Labeling & Namespaces:** Every K8s deployment must have `team`, `model_name`, and `environment` labels. I enforce this using an OPA Gatekeeper policy (if you submit a pod without these labels, K8s rejects it).
2.  **Kubecost / Datadog Cloud Cost Management:** We deployed Kubecost inside the cluster. It maps the AWS/GCP billing data back to specific K8s pods based on CPU/GPU request time and labels.
3.  **MLflow Cost Tagging:** For training jobs, we calculate the cost heuristically inside the training script and log it as a metric in MLflow.
    ```python
    import time
    start = time.time()
    # ... train model ...
    duration_hours = (time.time() - start) / 3600
    cost_per_hour = 3.06 # cost of p3.2xlarge
    mlflow.log_metric("training_cost_usd", duration_hours * cost_per_hour)
    ```
4.  **Dashboards:** I built a Datadog dashboard for the Director of Data Science showing:
    *   Daily training spend per team.
    *   Daily inference spend per model.
    *   Idle GPU waste (nodes running but pods utilizing <10% GPU).

By exposing the costs directly to the data scientists via MLflow and Grafana, they naturally started optimizing their own hyperparameter sweeps because they had a budget constraint.

---
*[Back to README](../README.md)*
