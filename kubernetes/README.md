# Kubernetes & Platform Engineering for ML - Interview Questions

## Q1: How do you optimize Kubernetes deployments for ML workloads (CPU vs GPU)?

**Real-world context:**
We migrated our ML workloads to K8s, but pods were constantly getting evicted, and our cloud bill doubled because the cluster autoscaler was spinning up new nodes unnecessarily.

**Answer:**

Optimizing ML workloads on K8s requires understanding the difference between compute-bound (training/deep learning inference) and memory/IO-bound tasks (data preprocessing/XGBoost inference).

**1. Resource Requests and Limits:**
*   **The Problem:** Setting `requests` == `limits` (Guaranteed QoS) is standard best practice for web apps. But ML workloads are bursty. An inference request might spike CPU to 400% for 50ms, then drop to 5%. If you set limits too tight, K8s CPU throttling (`CPU CFS quota`) will artificially slow down inference latency.
*   **The Fix:** For CPU inference, I set a moderate `request` (e.g., 2 vCPU) for scheduling, but remove the CPU `limit` entirely (or set it very high). This allows the pod to burst. Memory limits, however, *must* be strictly set to prevent node-level OOMs.

**2. Node Affinity and Taints/Tolerations:**
*   **The Problem:** A random backend microservice scheduling onto a $30/hour GPU node.
*   **The Fix:** We use separate node pools.
    *   **GPU Node Pool:** Tainted with `nvidia.com/gpu=true:NoSchedule`. Only ML pods with the corresponding toleration can land here.
    *   **CPU Node Pool:** Labelled `workload-type=ml-inference`.

---

## Q2: Describe your strategy for autoscaling ML inference services in K8s.

**Real-world context:**
Standard Horizontal Pod Autoscaler (HPA) based on CPU utilization was failing us. An NLP model doing heavy inference showed 95% CPU usage constantly. HPA kept scaling up to the max replica count, but throughput wasn't improving.

**Answer:**

**Why CPU-based HPA fails for ML:**
ML models consume all available resources to compute matrix multiplications as fast as possible. High CPU/GPU utilization is expected, not necessarily an indicator that the pod is overwhelmed.

**My Autoscaling Strategy (Custom Metrics):**
We abandoned CPU HPA and moved to **Requests Per Second (RPS)** or **Concurrency-based scaling**.

1.  **Metrics Server:** We use Datadog to expose custom metrics to the Kubernetes API.
2.  **HPA Configuration:** Scale based on `requests_per_second`.

**Handling Cold Starts (The ML Challenge):**
Scaling up a 5GB ML model takes 1-2 minutes. If a traffic spike hits, RPS-based HPA will request new pods, but the existing pods will be crushed before the new ones are ready.

**The Fix:**
1.  **Over-provisioning slightly:** We set `targetAverageValue` lower than the pod's actual capacity to create a buffer.
2.  **KEDA:** We use KEDA to scale based on Kafka lag. KEDA can also scale deployments to zero when there's no traffic.

---

## Q3: How do you implement Canary deployments for ML models on K8s?

**Real-world context:**
A data scientist wanted to test a new XGBoost model against the production version. They asked me to spin up a whole new cluster environment to run a comparison test. I implemented canary deployments instead.

**Answer:**

**Architecture (using Istio or KServe):**

1.  We have two K8s Deployments: `model-v1` (Production) and `model-v2` (Canary).
2.  We use an Istio `VirtualService` to split traffic routing at the network edge.

**The Rollout Process:**
1.  CD pipeline deploys `model-v2` and updates Istio to 5% traffic.
2.  We wait 1 hour. Datadog monitors error rates and latency for the `model-v2` pods specifically.
3.  If metrics are stable, CD pipeline updates Istio to 25%.
4.  We wait 24 hours. Data science team reviews the prediction distribution (are the scores similar to v1?) and business metrics.
5.  If approved, CD updates to 100%. `model-v1` is scaled down.

---

## Q4: How do you manage massive datasets (TBs of images/text) for training jobs in Kubernetes?

**Real-world context:**
Data scientists were baking 50GB datasets directly into their Docker images. Docker pulls were taking 20 minutes, node disks were filling up instantly, and Kubernetes was crashing. 

**Answer:**

You absolutely cannot put datasets in Docker images or Git repositories.

**Strategy 1: Network Attached Storage (NFS/EFS) via PVCs**
*   **How:** We create a Kubernetes Persistent Volume Claim (PVC) backed by Amazon EFS or GCP Filestore. 
*   **Pros:** The data scientist mounts the PVC to their training pod at `/data`. They can read files using standard Python `open()`. Multiple pods can read the exact same data simultaneously (ReadWriteMany).
*   **Cons:** NFS is notoriously slow for Deep Learning. If your GPU can process 10,000 images a second, but NFS can only serve 500 images a second, your GPU utilization drops to 5%.

**Strategy 2: Streaming from Object Storage (S3/GCS)**
*   **How:** We use libraries like `webdataset` (for PyTorch) or TensorFlow Datasets. The training script streams the dataset as a tar archive directly from S3 into memory, completely bypassing the local Kubernetes node disk.
*   **Pros:** Infinite scale, extremely high throughput (S3 can saturate a 100Gbps network link), and zero local disk required on the K8s nodes.
*   **The Winner:** For anything over 100GB, streaming from S3 is mandatory.

---

## Q5: Standard Kubernetes scheduling is terrible for distributed ML training. Why, and what do you use instead?

**Real-world context:**
A data scientist launched a PyTorch distributed training job requiring 8 pods (each requesting 1 GPU). The K8s cluster only had 4 GPUs available. K8s scheduled 4 pods, and the other 4 sat in "Pending". The 4 running pods just sat there infinitely waiting for the others to join the NCCL communication ring, wasting thousands of dollars.

**Answer:**

**The Root Cause:**
Default Kubernetes uses a "Pod-by-Pod" scheduler. It tries to place one pod at a time. This is perfect for web servers. 
Distributed ML training (MPI, PyTorch DDP) requires "All-or-Nothing" scheduling. The job cannot start unless *all* workers are running. If you get 7 out of 8 pods, the job hangs.

**The Fix: Gang Scheduling**
We replaced standard Deployments with **Kubeflow Training Operator** (which uses Volcano or Kueue under the hood for scheduling).

*   **How it works:** When a `PyTorchJob` requests 8 GPUs, the gang scheduler checks the cluster state. If only 4 GPUs are available, it leaves *all 8 pods* in a pending state (or an internal queue) until enough resources free up. Once 8 GPUs are available, it schedules all 8 pods simultaneously.
*   This prevents deadlocks, prevents wasting expensive GPU hours on partially-scheduled jobs, and allows for fair-share queuing between different data science teams.

---

## Q6: How do you secure ML pods running on Kubernetes?

**Real-world context:**
We found a data scientist running a Jupyter notebook pod as `root`, mounting the host node's Docker socket, which effectively gave them root access to the entire underlying EC2 instance.

**Answer:**

**1. Pod Security Admission (PSA):**
We enforce the `Restricted` profile across all ML namespaces. 
*   Pods cannot run as root (`runAsNonRoot: true`).
*   Containers cannot mount host paths (prevents escaping the container).
*   Privilege escalation is explicitly denied.

**2. Network Policies:**
By default, any pod in K8s can talk to any other pod. A compromised Jupyter notebook could scan the internal network and query the production PostgreSQL database.
*   We implement default-deny network policies. 
*   The `jupyter-notebook` pods are explicitly only allowed to talk to the Internet (for downloading packages) and the specific internal S3 endpoint for data.

**3. IAM Roles for Service Accounts (IRSA/Workload Identity):**
We never inject AWS/GCP credentials as environment variables. We map a K8s ServiceAccount to an AWS IAM Role. The ML pod assumes that role via OIDC, getting temporary, short-lived tokens that only allow access to a specific S3 bucket path (e.g., `s3://ml-data/team-a/*`).

---
*[Back to README](../README.md)*
