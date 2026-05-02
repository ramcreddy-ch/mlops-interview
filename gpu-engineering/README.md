# GPU Engineering for MLOps - Interview Questions

## Q1: Your PyTorch inference pods are crashing with `RuntimeError: CUDA out of memory`. The model size is 2GB, but the pod has a 16GB T4 GPU. How do you debug and fix this?

**Real-world context:**
We deployed a vision model. It ran fine for hours, but randomly crashed with CUDA OOM during traffic spikes. The data scientists insisted the GPU was too small, but math showed 2GB model + 1GB intermediate tensors = 3GB max. Where was the other 13GB going?

**Answer:**

**Root Cause Analysis:**
In PyTorch (and Python in general), memory isn't immediately released back to the OS/GPU when an object goes out of scope due to the memory allocator. 
1. **Batch Size Spikes:** If the endpoint uses dynamic batching, a sudden request to batch 64 images instead of 8 will exponentially increase the intermediate activation memory required during the forward pass.
2. **Memory Fragmentation:** PyTorch caches CUDA memory to speed up future allocations. If tensor sizes vary wildly (e.g., dynamic input resolutions in computer vision), the cache fragments. You might have 5GB "free" in the cache, but no contiguous block large enough for a new 1GB tensor, triggering OOM.
3. **Memory Leaks:** Storing historical tensors (like keeping a list of predictions that accidentally still have the computational graph attached via `.grad_fn`).

**Debugging Steps:**
1. Check Datadog for `nvidia.smi.memory_used`. Is it a slow leak over hours, or a sudden spike?
2. SSH into the pod/node and run `watch -n 1 nvidia-smi` to catch the behavior during load testing.
3. Use `torch.cuda.memory_summary()` inside the `except RuntimeError:` block before the pod dies to see the exact breakdown of allocated vs. cached memory.

**The Fix:**
1. **Strict Batch Limits:** Enforce `max_batch_size` in the serving framework (like Triton or FastAPI). 
2. **Input Normalization:** Force all images to a fixed resolution *before* they hit the GPU to prevent varying activation sizes and cache fragmentation.
3. **Empty Cache (Last Resort):** `torch.cuda.empty_cache()` releases the allocator's cached memory. We placed this inside a `try/except` block to recover from unexpected spikes without crashing the pod, though it incurs a performance hit.
4. **Use `torch.no_grad()`:** Ensure the inference code is wrapped in `@torch.no_grad()` or `with torch.inference_mode():`. Without this, PyTorch stores activations for backward passes, immediately causing OOM in serving.

---

## Q2: Compare Multi-Instance GPU (MIG), Time-Slicing, and MPS. When would you use each in a Kubernetes cluster?

**Real-world context:**
We had a cluster of 8x A100 (80GB) GPUs. We had 20 data scientists wanting Jupyter notebooks, 5 heavy training jobs, and 10 small inference services. Dedicating one A100 to a 2GB inference model was a $10,000/month waste. We needed to share them.

**Answer:**

**1. Time-Slicing (Kubernetes Device Plugin parameter):**
*   **How it works:** Context switching. The GPU rapidly switches between workloads, just like a CPU. 
*   **Isolation:** None. Memory and compute are shared. If Pod A allocates 79GB of VRAM, Pod B crashes with OOM. If Pod A runs a heavy matrix multiplication, Pod B's latency spikes.
*   **When to use:** Jupyter notebooks and Dev environments where users are mostly idle (writing code, thinking) and occasional latency spikes are acceptable. 

**2. MIG (Multi-Instance GPU):**
*   **How it works:** Hardware-level partitioning (only available on Ampere/Hopper like A100/H100). Carves one physical GPU into up to 7 distinct virtual GPUs (e.g., 1g.10gb profiles).
*   **Isolation:** Strict hardware isolation. Fault isolation (if one slice crashes, others survive), Memory bandwidth isolation, and Compute isolation.
*   **When to use:** Production multi-tenant training and high-SLA inference. If team A needs guaranteed latency, they get a dedicated MIG slice. 

**3. MPS (Multi-Process Service):**
*   **How it works:** A background daemon that intercepts CUDA API calls from multiple processes and funnels them onto the GPU simultaneously, overlapping compute and memory transfers.
*   **Isolation:** Partial. Better compute sharing than time-slicing, but no strict memory limits or fault isolation. If the MPS daemon crashes, all connected processes die.
*   **When to use:** When you have multiple lightweight inference workers *inside the same pod/container* (e.g., 4 Gunicorn workers sharing 1 GPU for inference). Not recommended for multi-tenant K8s pod sharing due to the lack of fault isolation.

---

## Q3: You are tasked with serving a large Deep Learning model (e.g., ResNet-50 or a small LLM). Why would you choose NVIDIA Triton Inference Server over a custom FastAPI application?

**Real-world context:**
Our team built a beautiful FastAPI wrapper around a PyTorch model. At 50 requests per second, CPU utilization hit 100% on the GIL (Global Interpreter Lock), while the GPU sat at 15% utilization. We couldn't feed the GPU fast enough.

**Answer:**

Writing a Flask/FastAPI wrapper is fine for prototypes, but it fails in high-throughput production for several reasons that Triton solves natively.

**1. Dynamic Batching:**
*   **FastAPI:** Processes requests sequentially (or via thread pools, which hit the GIL). If 10 requests come in at exactly the same time, the GPU processes them 1 by 1.
*   **Triton:** You configure a `max_queue_delay_microseconds` (e.g., 5000µs). Triton holds incoming requests in a C++ queue for 5ms, combines them into a single batched tensor, ships the batch to the GPU, gets the batched result, splits it back up, and responds to all 10 clients. This increases GPU throughput by 5x-10x.

**2. The Python GIL:**
*   **FastAPI:** Python's GIL means only one thread executes Python bytecode at a time. Data serialization (JSON to NumPy to PyTorch tensor) is extremely CPU-heavy. The GPU starves waiting for Python to prepare the next tensor.
*   **Triton:** Written in C++. It handles network I/O, queuing, and memory movement outside of Python. It can use ONNX Runtime or TensorRT directly (C++ backends) bypassing Python entirely for the forward pass.

**3. Model Ensembles / Pipelines:**
*   If your pipeline is: `Tokenizer -> Model -> Post-processor`. In FastAPI, you serialize/deserialize JSON between microservices or run it all in one script. Triton allows defining "Ensembles" where tensors are passed between models in shared GPU memory without ever copying data back to the CPU.

**Trade-offs:**
Triton has a steep learning curve. Configuring `config.pbtxt` and converting models to ONNX/TensorRT takes engineering effort. For a model getting 1 request per minute, FastAPI is better. For 100+ RPS, Triton is mandatory.

---

## Q4: How do you identify and fix GPU utilization bottlenecks during distributed training? 

**Real-world context:**
A team was training a ResNet model on 4x A100s. The training was taking 3 days. I checked Datadog, and `nvidia.smi.gpu_utilization` was hovering around 25%. We were burning money.

**Answer:**

Low GPU utilization in training almost always means the GPU is starved for data. The GPU computes so fast that it finishes its batch and sits idle waiting for the CPU/Network to send the next batch.

**Debugging framework:**

1.  **Is it a CPU Data Loading bottleneck?**
    *   **Symptom:** `htop` shows CPU cores pegged at 100% while GPU is low.
    *   **Fix:** Increase `num_workers` in the PyTorch `DataLoader`. Move image resizing/augmentation from the CPU to the GPU using libraries like NVIDIA DALI or `torchvision.transforms.v2`.
2.  **Is it a Storage I/O bottleneck?**
    *   **Symptom:** CPU has free cycles, GPU is low, but `iostat` shows 100% disk utilization. 
    *   **Fix:** Ensure datasets are on local NVMe SSDs, not EBS volumes or network shares. If using K8s, use local ephemeral storage. If data must be read from S3, use streaming webdataset formats so you don't download millions of tiny files.
3.  **Is it a Network/Communication bottleneck (Distributed Training)?**
    *   **Symptom:** Using DistributedDataParallel (DDP). GPUs are at 100% during the forward/backward pass, then drop to 0% for a long time.
    *   **Fix:** This is the "All-Reduce" synchronization step. Ensure NCCL is using the fastest path. Are the GPUs connected via NVLink, or are they going over the PCIe bus? Is NCCL falling back to Ethernet instead of InfiniBand/EFA? Check NCCL logs (`NCCL_DEBUG=INFO`).

---

## Q5: What is TensorRT, and how does it optimize a PyTorch model for production?

**Real-world context:**
We had a hard 20ms latency SLA for a fraud model. PyTorch native inference was hitting 35ms. By compiling to TensorRT, we dropped latency to 8ms and saved half our compute costs.

**Answer:**

TensorRT is NVIDIA's C++ SDK for high-performance deep learning inference. You cannot train with TensorRT; it is strictly an AOT (Ahead-of-Time) compiler for deploying models.

**How it optimizes the model:**

1.  **Layer & Tensor Fusion:** PyTorch executes layers sequentially. If you have a Convolution layer followed by a ReLU activation, PyTorch launches two separate CUDA kernels. This means reading/writing to global GPU memory twice. TensorRT "fuses" these into a single kernel, reading memory once, applying both operations in registers, and writing once. This massively reduces memory bandwidth limits.
2.  **Precision Calibration (Quantization):** TensorRT can automatically convert FP32 weights to FP16 or INT8. INT8 uses 4x less memory and executes much faster on Tensor Cores, while TensorRT's calibration process ensures minimal accuracy loss.
3.  **Kernel Auto-Tuning:** The optimal way to multiply two matrices depends on the specific architecture of the GPU (T4 vs A100) and the exact batch size. TensorRT profiles multiple kernel implementations on the target hardware during compilation and selects the fastest one.

**The Workflow:**
1. Train in PyTorch (.pt).
2. Export to ONNX (`torch.onnx.export`).
3. Run `trtexec` on the *target production GPU* to compile the ONNX graph into a `.plan` or `.engine` file. 

**Common Mistake:**
Compiling the TensorRT engine on a CI/CD server with a CPU or a T4 GPU, and then trying to deploy that artifact to a production server running an A100. TensorRT engines are highly hardware-specific and will crash if the architectures don't match exactly.
