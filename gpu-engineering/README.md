# GPU Engineering for MLOps - Interview Questions

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
**Answer:** CPU pegged? DataLoader bottleneck. Disk I/O pegged? Storage bottleneck. GPUs idling periodically? NCCL "All-Reduce" communication bottleneck. Fix: Use pinned memory, NVMe drives, and ensure NVLink/InfiniBand is active.

## Q5: How does TensorRT optimize PyTorch models?
**Context:** Meeting a 20ms latency SLA.
**Answer:** AOT compiler. 1. Fuses layers (e.g., Conv + ReLU) into single kernels to reduce memory reads. 2. Calibrates weights to INT8 precision (4x smaller). 3. Auto-tunes kernels specifically for the target GPU architecture.

## Q6: What is GPUDirect Storage (GDS)?
**Context:** CPU bottlenecking data loading from NVMe.
**Answer:** Bypasses the CPU bounce buffer. Data flows directly from the NVMe drive over the PCIe bus into GPU memory. Critical for massive dataset training where the CPU cannot parse data fast enough.

## Q7: How do you monitor GPU health in production?
**Context:** Silent hardware failures causing bad predictions.
**Answer:** Run DCGM Exporter on K8s DaemonSets. Alert on `DCGM_FI_DEV_XID_ERRORS` (hardware faults), excessive thermal throttling (`DCGM_FI_DEV_GPU_TEMP`), and ECC memory errors.

## Q8: Explain NVLink vs PCIe.
**Context:** Designing a multi-GPU training rig.
**Answer:** PCIe Gen4 maxes out around 64 GB/s. NVLink provides up to 900 GB/s direct GPU-to-GPU bandwidth. Essential for distributed deep learning where massive gradient tensors must be synced constantly.

## Q9: How do you handle heterogeneous GPU clusters in K8s?
**Context:** Cluster has T4s, V100s, and A100s.
**Answer:** Label nodes via the NVIDIA Device Plugin (e.g., `nvidia.com/gpu.product=A100`). Deployments must specify `nodeSelector` or `nodeAffinity` because a TensorRT engine compiled for a T4 will crash on an A100.

## Q10: What is CUDA Pinned Memory?
**Context:** Speeding up PyTorch DataLoaders.
**Answer:** Page-locked CPU memory. It allows the GPU to use Direct Memory Access (DMA) to pull data across the PCIe bus without involving the CPU, drastically speeding up host-to-device transfers. Use `pin_memory=True`.

## Q11: How do you profile GPU code?
**Context:** Identifying the slowest layer in a model.
**Answer:** Use Nsight Systems or PyTorch Profiler. Generate a trace file and view it in Chrome Tracing (`chrome://tracing`). It visually shows the exact microsecond duration of every CUDA kernel launch and memory transfer.

## Q12: Explain Half Precision (FP16/BF16) vs Single Precision (FP32).
**Context:** Reducing VRAM usage.
**Answer:** FP32 uses 4 bytes per weight. FP16 uses 2 bytes. Moving to FP16/BF16 halves VRAM usage and doubles Tensor Core throughput. BF16 (Brain Float) has the same range as FP32, preventing gradient underflow during training.

## Q13: What are Tensor Cores?
**Context:** Understanding Ampere architecture.
**Answer:** Specialized execution units on NVIDIA GPUs designed specifically to perform matrix multiply-and-accumulate (MAC) operations in a single clock cycle. They are the primary reason modern GPUs are so fast for ML.

## Q14: Handling GPU driver updates in K8s.
**Context:** Rolling out new CUDA versions.
**Answer:** Use the NVIDIA GPU Operator. It containerizes the driver, container toolkit, and device plugin. Updating drivers becomes a standard K8s rolling update of a DaemonSet, avoiding manual `apt-get` on bare nodes.

## Q15: What is NCCL (Nickel)?
**Context:** Debugging distributed training hangs.
**Answer:** NVIDIA Collective Communications Library. It provides highly optimized multi-GPU primitives (All-Reduce, Broadcast) over NVLink or PCIe. If NCCL falls back to standard Ethernet instead of InfiniBand, training slows to a crawl.

## Q16: Troubleshooting PyTorch DDP (DistributedDataParallel).
**Context:** Processes getting stuck waiting for each other.
**Answer:** Ensure the `MASTER_ADDR` and `MASTER_PORT` are correctly set across all pods. Set `NCCL_DEBUG=INFO` to verify the GPUs are successfully forming a communication ring.

## Q17: Dynamic Batching internals in Triton.
**Context:** Optimizing throughput.
**Answer:** Triton holds requests in a C++ queue for `max_queue_delay_microseconds`. It concatenates the input tensors along the 0th dimension, runs a single forward pass, and slices the output tensor back to the individual clients.

## Q18: What is ONNX Runtime?
**Context:** Running models without PyTorch.
**Answer:** A highly optimized C++ inference engine. Exporting PyTorch to ONNX and running via ONNX Runtime removes the massive Python/PyTorch overhead, significantly reducing CPU usage and latency.

## Q19: Managing GPU power states.
**Context:** Bare metal data center limits.
**Answer:** Use `nvidia-smi -pl` to set a Power Limit. If your data center lacks cooling/power capacity, capping an A100 from 400W to 300W prevents tripped breakers with only a ~5% loss in ML performance.

## Q20: Deep Dive: Gradient Accumulation.
**Context:** Trying to train with a batch size of 256 on a GPU that only fits 64.
**Answer:** Run 4 forward passes of batch size 64. Accumulate the gradients without calling `optimizer.step()`. On the 4th pass, step the optimizer. Simulates a larger batch size without OOMing.

## Q21: What is FlashAttention?
**Context:** Optimizing Transformer memory.
**Answer:** Standard self-attention memory complexity is quadratic to sequence length ($O(N^2)$). FlashAttention fuses the attention mechanism into a single GPU kernel, utilizing SRAM (fast, small memory on the GPU) to avoid reading/writing intermediate matrices to HBM (slow, large VRAM).

## Q22: GPU Memory Fragmentation vs PyTorch Caching.
**Context:** Why `nvidia-smi` shows 100% full, but PyTorch claims to have free memory.
**Answer:** PyTorch's caching allocator holds onto memory to avoid expensive `cudaMalloc` calls. If `nvidia-smi` is full, it's just PyTorch holding the cache. Use `torch.cuda.memory_summary()` to see the truth.

## Q23: When to use `torch.cuda.empty_cache()`?
**Context:** Avoiding OOMs.
**Answer:** Almost never in production loops. It forces a sync and slows down the GPU drastically. Only use it in an `except RuntimeError:` block to attempt recovery after a massive, unexpected batch spike.

## Q24: What is CPU-GPU synchronization overhead?
**Context:** `loss.item()` slowing down training.
**Answer:** GPU operations are asynchronous. Calling `loss.item()` forces the CPU to halt and wait for the GPU to finish computing the loss so it can print the float value. Do this sparingly (e.g., every 100 batches).

## Q25: Optimizing embedding tables for GPUs.
**Context:** Recommender system with 10M users.
**Answer:** 10M embeddings won't fit in VRAM. Use CPU memory for the full table, and only transfer the active batch's embeddings to the GPU, or partition the embedding table across multiple GPUs using libraries like NVIDIA HugeCTR.

## Q26: Understanding the CUDA Execution Model.
**Context:** Writing custom CUDA kernels.
**Answer:** Grids, Blocks, and Threads. A GPU executes Threads in groups of 32 called Warps. If 16 threads branch one way in an `if` statement, and 16 branch the other, the Warp executes both sequentially (Warp Divergence), killing performance.

## Q27: Continuous Profiling of GPUs.
**Context:** Noticing slow degradation over time.
**Answer:** Deploy Datadog or Pyroscope. It occasionally samples GPU kernels and builds a flame graph. Helps identify if a new code commit accidentally moved an operation from GPU back to CPU.

## Q28: How to use Multiple GPUs on a single node efficiently.
**Context:** 8x GPU node. Data parallelism vs Model parallelism.
**Answer:** Data Parallelism (DDP): Copy the whole model to all 8 GPUs, split the data batch. Model Parallelism: The model is too big for 1 GPU. Put layers 1-10 on GPU 1, layers 11-20 on GPU 2. (Requires Pipeline Parallelism to prevent GPUs from idling).

## Q29: Using Kubernetes Ephemeral Storage with GPUs.
**Context:** Training pod needs scratch space.
**Answer:** High-speed NVMe drives attached to the GPU node should be mounted as `emptyDir` volumes. Training scripts download data from S3 to this `emptyDir` to ensure the GPU isn't starved waiting for network reads.

## Q30: Security of GPUs in K8s (Time-Slicing risks).
**Context:** Multi-tenant cluster.
**Answer:** Time-slicing provides no memory isolation. A malicious user in Pod A could theoretically read the VRAM left behind by Pod B. For zero-trust multi-tenancy, you MUST use MIG or dedicated GPUs.
