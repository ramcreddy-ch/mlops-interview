# GPU Engineering for MLOps — Production Deep Dive
## CUDA, Triton, Distributed Training, GPU Sharing, TensorRT

---

## Part 1: Production GPU Architecture — What Actually Runs in Production

### The GPU Memory Hierarchy (Critical to Understand)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    A100 GPU MEMORY HIERARCHY                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  REGISTERS (~256KB per SM)  ← Fastest. Per-thread variables.        │
│       ↕ ~20 TB/s                                                    │
│  L1 CACHE / SRAM (~192KB per SM)  ← Shared within SM.              │
│       ↕ ~19 TB/s                                                    │
│  L2 CACHE (40MB)  ← Shared across all SMs.                         │
│       ↕ ~2 TB/s                                                     │
│  HBM2e (80GB VRAM)  ← "GPU memory". Where model weights live.      │
│       ↕ ~2 TB/s (HBM bandwidth, much faster than PCIe)             │
│  PCIe 4.0 (64 GB/s bidirectional)  ← To CPU RAM / NVMe            │
│       ↕ ~64 GB/s                                                    │
│  CPU RAM (System Memory)  ← Where DataLoader lives.                 │
│       ↕ ~50 GB/s (DDR5)                                            │
│  NVMe SSD  ← Where dataset files are stored.                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key insight for MLOps:** Most GPU performance problems are memory bandwidth problems, NOT compute problems. Your model is often **waiting for data** at one of these bottlenecks.

---

## Part 2: GPU Sharing Strategies — When to Use What

### MIG vs Time-Slicing vs MPS — Complete Comparison

```
A100 80GB — 3 ways to share it:

TIME-SLICING:
  ┌───────────────────────────────────────┐
  │  Thread 1 → GPU → Thread 2 → GPU ... │
  │  Context switch between processes     │
  └───────────────────────────────────────┘
  Isolation: NONE (memory shared, can read other's VRAM!)
  Performance: ~50-70% of dedicated GPU per process
  K8s config: nvidia-smi --set-compute-mode=TIME_SHARING
  Best for: Dev environments ONLY

MIG (Multi-Instance GPU) — RECOMMENDED FOR PRODUCTION:
  ┌──────────────┬──────────────┬──────────────┐
  │   Instance 1 │   Instance 2 │   Instance 3 │
  │  20GB VRAM   │  20GB VRAM   │  40GB VRAM   │
  │  26 SMs      │  26 SMs      │  52 SMs      │
  └──────────────┴──────────────┴──────────────┘
  Isolation: HARDWARE-LEVEL (fault and memory isolated)
  Performance: Exactly proportional (40GB gets exactly 50% of GPU)
  Use for: Multi-tenant production, regulated environments
  Limitation: Cannot resize dynamically (must be configured at cluster level)

MPS (Multi-Process Service):
  ┌───────────────────────────────────────────┐
  │ Process 1 ──┐                             │
  │             ├──▶ MPS Daemon ──▶ GPU SMs  │
  │ Process 2 ──┘                             │
  └───────────────────────────────────────────┘
  Isolation: PARTIAL (compute isolated, memory NOT isolated)
  Performance: BEST (no context switching, shares SM units)
  Best for: Single-tenant with multiple workers (e.g., 8 Triton workers on 1 GPU)
```

### Kubernetes Configuration for MIG:

```yaml
# MIG profile setup (run on each GPU node)
# nvidia-smi mig -cgi 3g.40gb,2g.20gb,1g.10gb -C

# Device plugin config for MIG-aware K8s
apiVersion: v1
kind: ConfigMap
metadata:
  name: time-slicing-config
  namespace: gpu-operator
data:
  config.yaml: |
    version: v1
    sharing:
      mig:
        strategy: mixed   # Allow mixed MIG profiles on one node

---
# Pod requesting a specific MIG instance
apiVersion: v1
kind: Pod
metadata:
  name: fraud-model-mig
spec:
  containers:
    - name: fraud-model
      image: fraud-model:latest
      resources:
        limits:
          nvidia.com/mig-3g.40gb: 1   # Request the 40GB MIG slice
  nodeSelector:
    nvidia.com/gpu.product: A100-SXM4-80GB-MIG-1g.10gb  # Only land on MIG nodes
```

---

## Part 3: Debugging CUDA OOM — Production Playbook

### The OOM Debugging Flowchart

```
CUDA OOM Exception
       │
       ▼
torch.cuda.memory_summary(device=0)
       │
       ├── Is reserved >> allocated?
       │     YES → Fragmentation issue
       │           Fix: torch.cuda.empty_cache() [ONLY outside hot path]
       │           Better fix: Enforce consistent input shapes
       │
       ├── Is allocated close to VRAM limit?
       │     YES → Genuine OOM
       │           Fix: Reduce batch size, use gradient checkpointing,
       │                use mixed precision (FP16), offload embeddings to CPU
       │
       └── Is reserved ≈ allocated and both high?
             YES → Memory leak
                   Fix: Check for reference cycles, detach tensors, 
                        use del + empty_cache periodically
```

### Production Memory Debugging Code:

```python
# gpu_memory_debug.py — Run this when you get CUDA OOM
import torch
import gc
import traceback

def print_gpu_memory_state(device: int = 0):
    """Comprehensive GPU memory snapshot."""
    print(f"\n{'='*60}")
    print(f"GPU {device} Memory State:")
    print(f"{'='*60}")

    # High-level summary
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    max_alloc = torch.cuda.max_memory_allocated(device) / 1024**3

    print(f"  Allocated:    {allocated:.2f} GB")
    print(f"  Reserved:     {reserved:.2f} GB  (PyTorch cache — held, not freed)")
    print(f"  Max Alloc:    {max_alloc:.2f} GB  (peak since last reset)")
    print(f"  Fragmentation:{(reserved - allocated) / (reserved + 1e-8):.1%}")

    # Find large tensors still in memory
    print("\n  Top 10 largest tensors in Python scope:")
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                tensors.append((obj.nelement() * obj.element_size() / 1024**2, obj.size(), obj.dtype))
        except Exception:
            pass

    tensors.sort(reverse=True)
    for size_mb, shape, dtype in tensors[:10]:
        print(f"    {size_mb:.1f} MB | shape={shape} | dtype={dtype}")

    # Detailed breakdown
    print(f"\n  Full summary:\n{torch.cuda.memory_summary(device=device)}")


def safe_inference(model, inputs, batch_size_fallback: int = 16):
    """Inference with automatic OOM recovery."""
    try:
        with torch.no_grad(), torch.cuda.amp.autocast():
            return model(inputs)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"OOM detected! Retrying with smaller batch...")
            torch.cuda.empty_cache()
            gc.collect()

            # Split batch in half and retry
            half = inputs.shape[0] // 2
            out1 = model(inputs[:half])
            out2 = model(inputs[half:])
            return torch.cat([out1, out2], dim=0)
        raise


# Common OOM causes and fixes:
COMMON_OOM_CAUSES = {
    "Large batch size": "Reduce batch_size or use gradient accumulation",
    "Storing tensors in lists": "Call .detach().cpu() before storing, or just store numpy",
    "Not using no_grad()": "Wrap inference in torch.no_grad() — saves 2x memory (no grad buffers)",
    "Undeleted computation graph": "Always call loss.backward() then optimizer.zero_grad(set_to_none=True)",
    "Model loaded multiple times": "Check if multiple workers each load the model (use shared memory)",
    "Input shape inconsistency": "Variable input shapes cause fragmentation — pad to fixed size",
}
```

---

## Part 4: TensorRT — Production Optimization Pipeline

```python
# tensorrt_optimizer.py — Complete PyTorch → TensorRT optimization
import torch
import tensorrt as trt
from torch2trt import torch2trt
import numpy as np
import time

def optimize_model_for_production(
    pytorch_model: torch.nn.Module,
    input_shape: tuple,
    output_path: str,
    precision: str = "fp16",    # "fp32", "fp16", "int8"
    calibration_data=None,       # Required for INT8
):
    """
    Convert PyTorch model to TensorRT engine.
    Typical speedups: FP16 = 2-3x, INT8 = 4-6x vs FP32 PyTorch.
    """
    model = pytorch_model.eval().cuda()
    dummy_input = torch.randn(*input_shape).cuda()

    print(f"Original model - running benchmark...")
    pytorch_times = []
    for _ in range(100):
        with torch.no_grad():
            start = time.time()
            _ = model(dummy_input)
            torch.cuda.synchronize()
            pytorch_times.append((time.time() - start) * 1000)
    print(f"  PyTorch FP32: {np.mean(pytorch_times):.2f}ms P99: {np.percentile(pytorch_times, 99):.2f}ms")

    # Convert to TensorRT
    print(f"\nConverting to TensorRT ({precision})...")
    trt_model = torch2trt(
        model,
        [dummy_input],
        fp16_mode=(precision in ["fp16", "int8"]),
        int8_mode=(precision == "int8"),
        int8_calib_dataset=calibration_data,
        max_batch_size=256,
        max_workspace_size=1 << 30,   # 1GB workspace
    )

    # Save engine
    torch.save(trt_model.state_dict(), output_path)
    print(f"TensorRT engine saved to {output_path}")

    # Benchmark TensorRT
    trt_times = []
    for _ in range(100):
        start = time.time()
        _ = trt_model(dummy_input)
        torch.cuda.synchronize()
        trt_times.append((time.time() - start) * 1000)
    print(f"  TensorRT {precision}: {np.mean(trt_times):.2f}ms P99: {np.percentile(trt_times, 99):.2f}ms")
    print(f"  Speedup: {np.mean(pytorch_times)/np.mean(trt_times):.2f}x")

    return trt_model


# Triton model config with TensorRT backend
TRITON_CONFIG_TENSORRT = """
name: "fraud_model_trt"
backend: "tensorrt"
max_batch_size: 256

input [
  { name: "features", data_type: TYPE_FP16, dims: [150] }
]
output [
  { name: "fraud_score", data_type: TYPE_FP16, dims: [1] }
]

dynamic_batching {
  preferred_batch_size: [32, 64, 128, 256]
  max_queue_delay_microseconds: 5000
}

instance_group [
  { kind: KIND_GPU, count: 2, gpus: [0] }
]

optimization {
  priority: PRIORITY_MAX
  cuda {
    graphs: true          # Enable CUDA graph capture (removes kernel launch overhead)
  }
}
"""
```

---

## Part 5: Distributed Training Debugging — NCCL Issues

```python
# debug_distributed.py — Diagnosing distributed training failures
import os
import torch
import torch.distributed as dist

def diagnose_nccl_environment():
    """Print everything NCCL needs to be configured correctly."""
    print("=== NCCL Environment Diagnostic ===")
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'NOT SET ← WILL FAIL!')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT', 'NOT SET ← WILL FAIL!')}")
    print(f"WORLD_SIZE:  {os.environ.get('WORLD_SIZE', 'NOT SET')}")
    print(f"RANK:        {os.environ.get('RANK', 'NOT SET')}")
    print(f"LOCAL_RANK:  {os.environ.get('LOCAL_RANK', 'NOT SET')}")
    print(f"NCCL_DEBUG:  {os.environ.get('NCCL_DEBUG', 'not set (set to INFO for debugging)')}")
    print(f"NCCL_IB_DISABLE: {os.environ.get('NCCL_IB_DISABLE', 'not set')}")

    # Check if NCCL can see GPUs
    print(f"\nGPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    VRAM: {props.total_memory / 1024**3:.1f}GB")
        print(f"    NVLink: {'Yes' if props.multi_gpu_board else 'PCIe only'}")


# Common NCCL failures and fixes:
NCCL_FAILURES = {
    "Hang at init_process_group": [
        "MASTER_ADDR not reachable from all nodes",
        "Firewall blocking MASTER_PORT",
        "Fix: kubectl exec into both pods, test nc -zv MASTER_ADDR MASTER_PORT",
    ],
    "NCCL WARN Timeout": [
        "One process is slower (data imbalance between GPU workers)",
        "Fix: Ensure DistributedSampler(drop_last=True) to equalize batch sizes",
    ],
    "Falling back to Socket (slow)": [
        "InfiniBand not detected — using slower Ethernet",
        "Fix: Set NCCL_IB_DISABLE=0, verify IB devices: ibv_devinfo",
        "Perf impact: 10x slower gradient sync on Ethernet vs IB",
    ],
    "CUDA error: device-side assert": [
        "One GPU has an error, but DDP hangs waiting for it",
        "Fix: Set CUDA_LAUNCH_BLOCKING=1 to get real error location",
    ],
}
```

---

## Part 6: GPU Monitoring in Production — Complete Stack

```yaml
# dcgm-exporter-daemonset.yaml — GPU metrics for every node
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: dcgm-exporter
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: dcgm-exporter
  template:
    metadata:
      labels:
        app: dcgm-exporter
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9400"
    spec:
      containers:
        - name: dcgm-exporter
          image: nvcr.io/nvidia/k8s/dcgm-exporter:3.3.0-3.2.0-ubuntu22.04
          securityContext:
            runAsNonRoot: false      # DCGM requires root
            privileged: true
          ports:
            - containerPort: 9400
          env:
            - name: DCGM_EXPORTER_INTERVAL
              value: "5000"          # Scrape every 5 seconds
          volumeMounts:
            - name: pod-gpu-resources
              mountPath: /var/lib/kubelet/pod-resources
      volumes:
        - name: pod-gpu-resources
          hostPath:
            path: /var/lib/kubelet/pod-resources
      nodeSelector:
        nvidia.com/gpu: "true"
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
```

```yaml
# Key Prometheus alerts for GPU production health
groups:
  - name: gpu_alerts
    rules:
      - alert: GPUMemoryHigh
        expr: DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_TOTAL > 0.95
        for: 5m
        annotations:
          summary: "GPU {{ $labels.gpu }} VRAM > 95% for 5 minutes"
          action: "Check for memory leaks. Consider scaling horizontally."

      - alert: GPUUtilizationLow
        expr: DCGM_FI_DEV_GPU_UTIL < 20
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "GPU {{ $labels.gpu }} utilization < 20% for 30 min"
          action: "Possible I/O bottleneck. Check DataLoader, S3 throughput."

      - alert: GPUTemperatureCritical
        expr: DCGM_FI_DEV_GPU_TEMP > 85
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "GPU temperature critical: {{ $value }}°C"
          action: "Thermal throttling imminent. Check cooling / reduce load."

      - alert: NVLinkErrors
        expr: increase(DCGM_FI_DEV_NVLINK_BANDWIDTH_TX[5m]) == 0
          and DCGM_FI_DEV_GPU_UTIL > 50
        annotations:
          summary: "NVLink may be down — training using PCIe (10x slower)"
          action: "Run nvidia-smi nvlink --status on affected nodes."

      - alert: ECCMemoryErrors
        expr: DCGM_FI_DEV_ECC_SBE_VOL_TOTAL > 0
        annotations:
          summary: "GPU ECC memory errors detected"
          action: "Schedule GPU health check. Persistent errors = hardware failure."
```

---

## Interview Questions (30 Q&A) — See Original Below

---

## Q1: How do you debug `CUDA out of memory` under load?
**Context:** PyTorch model crashing despite math showing enough VRAM.
**Answer:** Dynamic batching spikes memory usage. Fragmentation causes OOM even if total free VRAM is sufficient. Fix: Enforce `max_batch_size` in Triton, normalize input resolutions to prevent fragmentation, and use `torch.no_grad()` for serving.

## Q2: MIG vs Time-Slicing vs MPS for K8s.
**Context:** Sharing A100s across teams.
**Answer:** Time-Slicing: Context switching, no isolation. MIG: Hardware-level strict fault and memory isolation. MPS: Daemon funneling multiple processes simultaneously, partial compute isolation but no strict fault isolation.

## Q3: Why choose Triton Inference Server over FastAPI?
**Context:** FastAPI CPU bottlenecking the GPU.
**Answer:** Python's GIL prevents concurrent request processing. Triton is written in C++, handles dynamic batching asynchronously, and bypasses Python entirely for ONNX/TensorRT backends, maximizing GPU throughput.

## Q4: How do you identify distributed training bottlenecks?
**Context:** A100s sitting at 20% utilization.
**Answer:** CPU pegged? DataLoader bottleneck. Disk I/O pegged? Storage bottleneck. GPUs idling periodically? NCCL All-Reduce communication bottleneck. Fix: Use pinned memory, NVMe drives, and ensure NVLink/InfiniBand is active.

## Q5: How does TensorRT optimize PyTorch models?
**Context:** Meeting a 20ms latency SLA.
**Answer:** AOT compiler. 1. Fuses layers (Conv + ReLU) into single kernels. 2. Calibrates weights to INT8. 3. Auto-tunes kernels for the target GPU. Typical result: 2-6x speedup over PyTorch.

## Q6: What is GPUDirect Storage (GDS)?
**Context:** CPU bottlenecking data loading from NVMe.
**Answer:** Bypasses the CPU bounce buffer. Data flows directly from NVMe → PCIe → GPU memory. Critical for massive dataset training where CPU cannot parse data fast enough.

## Q7: How do you monitor GPU health in production?
**Context:** Silent hardware failures causing bad predictions.
**Answer:** Run DCGM Exporter on K8s DaemonSets. Alert on DCGM_FI_DEV_XID_ERRORS (hardware faults), thermal throttling, and ECC memory errors.

## Q8: Explain NVLink vs PCIe.
**Context:** Designing a multi-GPU training rig.
**Answer:** PCIe Gen4 maxes out at ~64 GB/s. NVLink provides up to 900 GB/s direct GPU-to-GPU bandwidth. Essential for distributed DL where massive gradient tensors must be synced constantly.

## Q9: How do you handle heterogeneous GPU clusters in K8s?
**Context:** Cluster has T4s, V100s, and A100s.
**Answer:** Label nodes via the NVIDIA Device Plugin. Deployments must specify nodeAffinity because a TensorRT engine compiled for T4 will crash on A100.

## Q10: What is CUDA Pinned Memory?
**Context:** Speeding up PyTorch DataLoaders.
**Answer:** Page-locked CPU memory. Allows GPU to use DMA to pull data across PCIe without CPU involvement. Use pin_memory=True in DataLoader.

## Q11: How do you profile GPU code?
**Answer:** Use Nsight Systems or PyTorch Profiler. Generate trace file and view in Chrome tracing. Shows exact microsecond duration of every CUDA kernel launch and memory transfer.

## Q12: Explain FP16/BF16 vs FP32.
**Answer:** FP32: 4 bytes/weight. FP16: 2 bytes. Halves VRAM, doubles Tensor Core throughput. BF16 has same range as FP32, preventing gradient underflow during training.

## Q13: What are Tensor Cores?
**Answer:** Specialized units performing matrix multiply-accumulate in single clock cycle. Primary reason modern GPUs are so fast for ML. Ampere: 312 TFLOPS FP16 vs 19.5 TFLOPS FP32.

## Q14: Handling GPU driver updates in K8s.
**Answer:** Use the NVIDIA GPU Operator. Containerizes driver, container toolkit, device plugin. Updates become K8s DaemonSet rolling updates — no manual apt-get on nodes.

## Q15: What is NCCL?
**Answer:** NVIDIA Collective Communications Library. Provides optimized multi-GPU primitives (All-Reduce, Broadcast) over NVLink or PCIe. If falling back to Ethernet instead of InfiniBand, training slows 10x.

## Q16: Troubleshooting PyTorch DDP hangs.
**Answer:** Ensure MASTER_ADDR and MASTER_PORT are set correctly across all pods. Set NCCL_DEBUG=INFO to verify communication ring formation.

## Q17: Dynamic Batching internals in Triton.
**Answer:** Triton holds requests in a C++ queue for max_queue_delay_microseconds. Concatenates inputs along batch dim, runs single forward pass, slices output back to individual clients.

## Q18: What is ONNX Runtime?
**Answer:** Highly optimized C++ inference engine. Exporting PyTorch to ONNX removes Python/PyTorch overhead — significantly reduces CPU usage and latency (2-3x speedup on CPU inference).

## Q19: Managing GPU power states.
**Answer:** Use nvidia-smi -pl to set Power Limit. Capping A100 from 400W to 300W prevents breaker issues with ~5% performance loss.

## Q20: Gradient Accumulation.
**Answer:** Run N forward passes, accumulate gradients without optimizer.step(). On Nth pass, step. Simulates batch_size * N without OOM.

## Q21: What is FlashAttention?
**Answer:** Fuses attention into single GPU kernel, uses SRAM instead of HBM for intermediate matrices. Reduces memory from O(N²) to O(N). Critical for long-context transformers.

## Q22: GPU Memory Fragmentation vs PyTorch Caching.
**Answer:** PyTorch's caching allocator holds memory to avoid expensive cudaMalloc calls. nvidia-smi shows full, but torch.cuda.memory_allocated() shows actual use. Use torch.cuda.memory_summary() for truth.

## Q23: When to use torch.cuda.empty_cache()?
**Answer:** Almost never in hot path. Forces GPU sync (slow). Only in except RuntimeError OOM blocks for emergency recovery.

## Q24: CPU-GPU synchronization overhead.
**Answer:** GPU ops are async. loss.item() forces CPU to wait for GPU. Do this sparingly (every 100 steps), not every step.

## Q25: Optimizing embedding tables for GPUs.
**Answer:** 10M embeddings won't fit in VRAM. Use CPU memory for full table, transfer active batch embeddings to GPU. Or use NVIDIA HugeCTR to partition across GPUs.

## Q26: CUDA Execution Model — Warp Divergence.
**Answer:** 32 threads = 1 Warp. If 16 threads branch one way and 16 the other in an if statement, Warp executes BOTH paths sequentially. Avoid branching in inner loops of custom kernels.

## Q27: Continuous Profiling of GPUs.
**Answer:** Deploy Pyroscope or Datadog Continuous Profiler. Occasionally samples GPU kernels, builds flame graph. Finds if new code commit accidentally moved ops from GPU back to CPU.

## Q28: Data Parallelism vs Model Parallelism.
**Answer:** Data Parallel (DDP): Copy full model to each GPU, different data batch per GPU. Model Parallel: Model too large for 1 GPU, split layers across GPUs. Add pipeline parallelism to prevent GPU stalls.

## Q29: Kubernetes Ephemeral Storage with GPUs.
**Answer:** Mount NVMe drives as emptyDir volumes. Training scripts download data from S3 to this emptyDir. Prevents GPU starvation from slow network reads during training.

## Q30: Security of GPUs in K8s (Time-Slicing risks).
**Answer:** Time-slicing has NO memory isolation. Malicious Pod A could read Pod B's VRAM residue. For zero-trust multi-tenancy, use MIG or dedicated GPU nodes per tenant.

---
*Next → [Python Internals for MLOps](../python-mlops/README.md)*
