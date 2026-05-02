# Kubernetes & Platform Engineering for ML - Interview Questions

## Q1: How do you optimize Kubernetes limits/requests for ML workloads?
**Context:** ML workloads were getting CPU throttled, increasing latency.
**Answer:** For CPU inference, set a moderate `request` (e.g., 2 vCPU) but remove the CPU `limit` (or set it very high) so the pod can burst during matrix multiplication. Memory limits, however, *must* be strictly set to prevent node-level OOMs.

## Q2: Why does standard CPU-based HPA fail for ML inference?
**Context:** HPA maxed out replicas but throughput didn't improve.
**Answer:** ML models (PyTorch/TF) are designed to consume 100% of available CPU to compute math as fast as possible. High CPU is expected. 
**Fix:** Use custom metrics (Requests Per Second or Concurrency) exposed via Datadog/Prometheus Adapter for HPA.

## Q3: How do you implement Canary deployments for ML models on K8s?
**Context:** Testing a new XGBoost model safely.
**Answer:** Use an Istio `VirtualService` or KServe to split traffic. Deploy `model-v2`, route 5% traffic. Monitor error rates/latency. If stable, increase to 25%, evaluate business metrics, then 100%. Native K8s rolling updates do not support traffic percentage splitting.

## Q4: How do you manage massive datasets (TBs) for K8s training jobs?
**Context:** Docker images were 50GB and crashing the nodes.
**Answer:** Never put data in Docker images. 
1. **PVCs (EFS/NFS):** Good for small data, but NFS is slow for Deep Learning.
2. **S3 Streaming:** The gold standard. Use `webdataset` to stream data directly from S3 into memory, bypassing the local node disk completely.

## Q5: Why is standard K8s scheduling terrible for distributed ML training?
**Context:** A 4-GPU distributed job requested resources, but only got 2 GPUs, deadlocking the job while holding expensive resources.
**Answer:** K8s schedules pod-by-pod. Distributed ML needs "All-or-Nothing" scheduling.
**Fix:** Use Gang Scheduling (Volcano or Kueue via Kubeflow Training Operator) so pods are only scheduled when the full quota is available.

## Q6: How do you secure ML pods running on Kubernetes?
**Context:** A Jupyter pod was running as root with host access.
**Answer:** 1. Pod Security Admission (Restricted profile: no root, no host paths). 2. NetworkPolicies (deny default ingress/egress). 3. IAM Roles for Service Accounts (IRSA/Workload Identity) to avoid hardcoded AWS/GCP credentials.

## Q7: Compare MIG, Time-slicing, and MPS for sharing GPUs in K8s.
**Context:** An A100 was wasted on a 2GB model.
**Answer:** Time-slicing: Context switching, no isolation. Good for Dev. MIG: Hardware-level strict isolation. Good for Prod multi-tenant. MPS: Process-level sharing, partial isolation. Good for multiple workers in a single container.

## Q8: How do you handle Scale-to-Zero for ML models?
**Context:** GPU nodes idling overnight.
**Answer:** Use KEDA (Kubernetes Event-driven Autoscaling) or Knative (used by KServe). It scales deployments to 0 when queue length is zero. The Cluster Autoscaler then kills the empty GPU nodes.

## Q9: How do you right-size ML pods using VPA?
**Context:** Developers were requesting 64GB RAM for every model.
**Answer:** Run Vertical Pod Autoscaler (VPA) in "Off" or "Recommend" mode. It profiles actual memory usage over weeks. Review the recommendations and update the Helm charts. Do not use VPA in "Auto" mode for ML, as restarting heavy models causes outages.

## Q10: How do you debug OOMKilled in K8s for an ML pod?
**Context:** Pods crashing under heavy traffic.
**Answer:** OOMKilled (Exit Code 137) means the pod exceeded `limits.memory`. In ML, this is usually due to large batch sizes or the serving framework spawning too many workers (e.g., 8 Gunicorn workers loading a 1GB model = 8GB). Fix: Reduce workers or increase the K8s memory limit.

## Q11: How do you use Node Affinity and Taints for ML?
**Context:** A NodeJS web app scheduled onto a $30/hr GPU node.
**Answer:** Taint GPU nodes with `nvidia.com/gpu=true:NoSchedule`. Only ML pods with the matching toleration can land there. Use `nodeAffinity` to ensure heavy CPU preprocessing lands on compute-optimized nodes.

## Q12: What is the NVIDIA Device Plugin?
**Context:** Setting up a bare-metal K8s cluster for ML.
**Answer:** It is a DaemonSet that exposes the number of GPUs on each node to the Kubelet. Without it, K8s doesn't know GPUs exist, and you cannot request `nvidia.com/gpu: 1` in your pod spec.

## Q13: Why use Istio Service Mesh for ML deployments?
**Context:** Moving from simple LoadBalancers to Istio.
**Answer:** ML requires advanced traffic management: Canary rollouts, Shadow deployments (mirroring traffic), retries with exponential backoff, and strict mTLS between the feature store and the inference pod for security.

## Q14: How do you handle cold starts of large models in K8s?
**Context:** Autoscaler requested a new pod, but traffic failed because the model took 2 minutes to load into VRAM.
**Answer:** Over-provision HPA target thresholds to scale *before* capacity is hit. Use readiness probes with a high `initialDelaySeconds` so K8s doesn't send traffic to the pod until the model is fully loaded.

## Q15: KServe vs Seldon Core vs Custom FastAPI. How do you choose?
**Context:** Platform engineering team architecture decision.
**Answer:** FastAPI: Good for simple, low-traffic models. KServe: Serverless, Knative-backed, great for scale-to-zero and standard ML frameworks. Seldon Core: Incredible for complex inference graphs (A/B testing, ensembles, multi-armed bandits) built directly into K8s CRDs.

## Q16: How do you manage ML deployments via Helm?
**Context:** Standardizing deployments.
**Answer:** Create a generic `ml-inference` Helm chart. Data scientists provide a `values.yaml` containing the `image`, `model_uri`, and `resources`. This prevents them from writing raw K8s YAML and ensures platform standards are enforced.

## Q17: How does ArgoCD (GitOps) apply to MLOps?
**Context:** Models were drifting from configuration.
**Answer:** The entire K8s state (including the specific model S3 URI) is stored in Git. Jenkins builds the image and updates Git. ArgoCD pulls from Git and applies to K8s. Rollbacks are simply `git revert`.

## Q18: When would you use StatefulSets for ML?
**Context:** Deploying a distributed vector database (Milvus/Pinecone clone).
**Answer:** Standard ML inference is stateless (Deployments). Feature stores (Redis) or Vector DBs require stable network identities and persistent storage volumes, mandating StatefulSets.

## Q19: Using DaemonSets in the ML Platform.
**Context:** Ensuring logs are captured.
**Answer:** Use DaemonSets to run Promtail/FluentBit (for log forwarding) and the NVIDIA DCGM Exporter (to scrape GPU metrics like temperature and util) on every single GPU node automatically.

## Q20: Designing NetworkPolicies for ML isolation.
**Context:** Multi-tenant K8s cluster.
**Answer:** Default deny-all. Explicitly allow the `fraud-team` namespace to talk to the `fraud-feature-store`. Explicitly block them from talking to the `reco-team` namespace to prevent cross-tenant data leakage.

## Q21: Implementing RBAC for Data Scientists in K8s.
**Context:** Security audit found devs had `cluster-admin`.
**Answer:** Data scientists get `edit` or custom Roles strictly bound to their specific namespace. They can view pods, check logs, and restart deployments, but cannot modify NetworkPolicies or access other teams' namespaces.

## Q22: PodPriority and Preemption for critical inference.
**Context:** A batch job was blocking a critical real-time inference pod from scheduling.
**Answer:** Assign a high `PriorityClass` to the real-time inference deployment. If the cluster is full, K8s will evict (preempt) the lower-priority batch job to make room for the critical inference pod.

## Q23: Handling Spot instances gracefully in K8s.
**Context:** AWS terminated a Spot node, and the training job failed ungracefully.
**Answer:** Deploy the AWS Node Termination Handler. It listens for the 2-minute interruption warning, cordons the node, and sends a SIGTERM to the pod. The ML code catches the SIGTERM and saves a final checkpoint before dying.

## Q24: Using InitContainers for downloading models.
**Context:** Baking a 10GB model into a Docker image was slowing down CI/CD.
**Answer:** The main container just has the inference code (50MB). An InitContainer runs first, downloads the 10GB model from S3 into a shared `emptyDir` volume, and exits. The main container then boots and reads from the volume.

## Q25: Ephemeral storage limits in K8s ML workloads.
**Context:** Nodes were going into `DiskPressure` state and evicting pods.
**Answer:** Training jobs often download massive ZIP files to the local container disk. Always set `requests.ephemeral-storage` and `limits` in the pod spec so K8s schedules the pod on a node with enough physical disk space.

## Q26: Implementing ML chargeback with Kube-cost.
**Context:** Finance wanted to know which ML team was burning cash.
**Answer:** Deploy Kubecost. Ensure every K8s manifest has mandatory labels (e.g., `team: risk`, `model: fraud`). Kubecost maps GPU hours back to the labels, providing dashboards for AWS cost per model.

## Q27: Liveness vs Readiness probes for ML models.
**Context:** K8s kept restarting a pod while it was downloading a model.
**Answer:** Liveness probe: "Is the API responding?" (If no, K8s restarts it). Readiness probe: "Is the model fully loaded into VRAM and ready to score?" (If no, K8s removes it from the LoadBalancer but does *not* restart it). Make the `initialDelaySeconds` high for Liveness.

## Q28: Cluster Autoscaler vs Karpenter for GPU nodes.
**Context:** AWS Cluster Autoscaler took 5 minutes to spin up a new GPU node.
**Answer:** Karpenter bypasses standard Autoscaler groups. It evaluates the pending pod's requirements (e.g., requires `nvidia.com/gpu`) and directly provisions the exact right EC2 instance in <60 seconds, saving massive scale-up time.

## Q29: Using the Sidecar pattern for Feature Fetching.
**Context:** Python inference code was bogged down making API calls to the feature store.
**Answer:** Deploy a fast Go/Rust sidecar container in the same pod. The sidecar handles the network I/O, circuit breaking, and caching to the remote feature store. The Python container just makes a fast `localhost` call to the sidecar.

## Q30: Troubleshooting slow image pulls (bottlenecks).
**Context:** K8s took 10 minutes to pull a 5GB PyTorch image.
**Answer:** 1. Use smaller base images (`slim` versions). 2. Use a DaemonSet to pre-pull the heavy base images onto all GPU nodes so they are cached locally. 3. Ensure the container registry (ECR) is in the exact same region as the EKS cluster to avoid cross-region network latency.
