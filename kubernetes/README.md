# Kubernetes & Platform Engineering for ML - Interview Questions

## Q1: How do you optimize Kubernetes deployments for ML workloads (CPU vs GPU)?

**Real-world context:**
We migrated our ML workloads to K8s, but pods were constantly getting evicted, and our cloud bill doubled because the cluster autoscaler was spinning up new nodes unnecessarily.

**Answer:**

Optimizing ML workloads on K8s requires understanding the difference between compute-bound (training/deep learning inference) and memory/IO-bound tasks (data preprocessing/XGBoost inference).

**1. Resource Requests and Limits:**
*   **The Problem:** Setting `requests` == `limits` (Guaranteed QoS) is standard best practice for web apps. But ML workloads are bursty. An inference request might spike CPU to 400% for 50ms, then drop to 5%. If you set limits too tight, K8s CPU throttling (`CPU CFS quota`) will artificially slow down inference latency.
*   **The Fix:** For CPU inference, I set a moderate `request` (e.g., 2 vCPU) for scheduling, but remove the CPU `limit` entirely (or set it very high). This allows the pod to burst and use available node CPU to process the request faster, decreasing latency. Memory limits, however, *must* be strictly set to prevent node-level OOMs.

**2. Node Affinity and Taints/Tolerations:**
*   **The Problem:** A random backend microservice scheduling onto a $30/hour GPU node.
*   **The Fix:** We use separate node pools.
    *   **GPU Node Pool:** Tainted with `nvidia.com/gpu=true:NoSchedule`. Only ML pods with the corresponding toleration can land here.
    *   **CPU Node Pool:** Labelled `workload-type=ml-inference`.
    *   ML deployments use `nodeSelector` or `nodeAffinity` to ensure they land on the right hardware.

**3. GPU Optimization (The tricky part):**
*   Unlike CPUs, GPUs cannot natively be shared fractionally across pods in raw Kubernetes. If a pod requests `nvidia.com/gpu: 1`, it owns the entire GPU, even if it only uses 10% of the compute.
*   **The Fix (Time-slicing):** For development environments (JupyterHub), we configured the NVIDIA GPU operator to enable time-slicing, allowing one physical GPU to appear as 4 or 8 virtual GPUs, sharing the compute and memory (though without hard isolation).
*   **The Fix (MIG):** For production A100s, we use Multi-Instance GPU (MIG) to carve the GPU into distinct hardware-isolated slices (e.g., `1g.5gb`), allowing multiple inference pods to share a node securely.

**Trade-offs:**
Removing CPU limits can lead to noisy neighbor problems if nodes are over-provisioned. You have to monitor node CPU utilization closely. Time-slicing GPUs saves money but can cause OOM errors if one user's notebook consumes all the VRAM.

---

## Q2: Describe your strategy for autoscaling ML inference services in K8s.

**Real-world context:**
Standard Horizontal Pod Autoscaler (HPA) based on CPU utilization was failing us. An NLP model doing heavy inference showed 95% CPU usage constantly. HPA kept scaling up to the max replica count, but throughput wasn't improving.

**Answer:**

**Why CPU-based HPA fails for ML:**
ML models (especially using frameworks like PyTorch or TensorFlow) are designed to consume all available resources to compute matrix multiplications as fast as possible. High CPU/GPU utilization is expected, not necessarily an indicator that the pod is overwhelmed.

**My Autoscaling Strategy (Custom Metrics):**

We abandoned CPU HPA and moved to **Requests Per Second (RPS)** or **Concurrency-based scaling**.

1.  **Metrics Server:** We use Datadog (or Prometheus Adapter) to expose custom metrics to the Kubernetes API.
2.  **The Metric:** We track `nginx.ingress.requests.rate` or custom application metrics exposing the queue depth.
3.  **HPA Configuration:**
    ```yaml
    apiVersion: autoscaling/v2
    kind: HorizontalPodAutoscaler
    spec:
      metrics:
      - type: Pods
        pods:
          metric:
            name: requests_per_second
          target:
            type: AverageValue
            averageValue: 20  # Scale up when average RPS per pod exceeds 20
    ```

**Handling Cold Starts (The ML Challenge):**
Scaling up a web app takes 2 seconds. Scaling up a 5GB ML model takes 1-2 minutes. If a traffic spike hits, RPS-based HPA will request new pods, but the existing pods will be crushed before the new ones are ready.

**The Fix:**
1.  **Over-provisioning slightly:** We set `targetAverageValue` lower than the pod's actual capacity to create a buffer.
2.  **KEDA (Kubernetes Event-driven Autoscaling):** We use KEDA to scale based on Kafka lag or SQS queue length for asynchronous ML workloads. KEDA can also scale deployments to zero when there's no traffic, saving massive costs on GPU nodes.
3.  **VPA (Vertical Pod Autoscaler):** We use VPA strictly in "Off" or "Recommend" mode to profile memory usage over time and right-size our resource requests. We never use auto-update VPA for ML pods, as restarting a heavy ML pod causes unacceptable latency spikes.

---

## Q3: How do you implement Canary deployments for ML models on K8s?

**Real-world context:**
A data scientist wanted to test a new XGBoost model against the production version. They asked me to spin up a whole new cluster environment to run a comparison test. I implemented canary deployments instead.

**Answer:**

Deploying ML models is risky because they degrade silently. You must test with real production traffic.

**Architecture (using Istio or KServe):**

1.  We have two K8s Deployments: `model-v1` (Production) and `model-v2` (Canary).
2.  We use an Istio `VirtualService` to split traffic routing at the network edge.

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: fraud-routing
spec:
  hosts:
  - fraud-service.default.svc.cluster.local
  http:
  - route:
    - destination:
        host: model-v1
      weight: 95
    - destination:
        host: model-v2
      weight: 5
```

**The Rollout Process:**
1.  CD pipeline deploys `model-v2` and updates Istio to 5% traffic.
2.  We wait 1 hour. Datadog monitors error rates and latency for the `model-v2` pods specifically.
3.  If metrics are stable, CD pipeline updates Istio to 25%.
4.  We wait 24 hours. Data science team reviews the prediction distribution (are the scores similar to v1?) and business metrics.
5.  If approved, CD updates to 100%. `model-v1` is scaled down.

**Alternative: Shadow Deployments**
For high-risk models (e.g., loan approval), even 5% canary is too risky. We use Istio's mirroring feature to send a copy of 100% of the traffic to `model-v2`. The responses from v2 are logged to Kafka but *dropped* (not returned to the user). This allows us to compare v1 vs v2 predictions on identical production traffic with zero user impact.

**Common Mistakes:**
- Doing canary rollouts based on K8s native rolling updates (modifying a single Deployment). Native K8s doesn't give you fine-grained traffic percentage control; it just replaces pods one by one. You need a service mesh (Istio/Linkerd) or an ingress controller (NGINX/Contour) that supports traffic splitting.

---
*[Back to README](../README.md)*
