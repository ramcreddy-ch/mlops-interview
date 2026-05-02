# Python Internals for MLOps - Interview Questions

## Q1: How does the GIL impact ML model serving, and how do you bypass it?
**Context:** FastAPI CPU bottlenecking GPUs.
**Answer:** The Global Interpreter Lock allows only one thread to execute Python bytecodes at once. Math libraries release it, but preprocessing (JSON/Pandas) holds it. Bypass by using multi-processing (Gunicorn workers) or tools like Ray/Triton that manage processes outside the GIL.

## Q2: How do you profile a memory leak in a long-running ML microservice?
**Context:** OOMKilled every 48 hours.
**Answer:** Python uses reference counting. Leaks happen when references (like appending predictions to a global list) aren't cleared. Use the `tracemalloc` module to take snapshots before and after 10k requests, compare them, and find the exact line creating the un-garbage-collected objects.

## Q3: Why is `pickle` dangerous for ML model serialization?
**Context:** Security audit failure.
**Answer:** `pickle` deserializes arbitrary Python objects and executes the `__reduce__` method. Attackers can embed `os.system("rm -rf /")` inside a `model.pkl`. Always use ONNX, Safetensors, or strict validation pipelines.

## Q4: How do you guarantee reproducible Python dependencies?
**Context:** Autoscaler crashed because `pip install` pulled a new breaking NumPy version.
**Answer:** Never use unpinned `requirements.txt`. Use Poetry or Pip-tools to generate a lockfile (`poetry.lock`) containing exact versions and cryptographic hashes of every transitive dependency. Bake this into the Docker image.

## Q5: How do you write an efficient data loader to prevent GPU starvation?
**Context:** GPU at 15% util during PyTorch training.
**Answer:** 1. Never load data on the main thread. Set PyTorch `num_workers=8` (multiprocessing). 2. Set `pin_memory=True` to use DMA across the PCIe bus. 3. Set `prefetch_factor=2` so batches are ready before the GPU finishes the current one.

## Q6: Explain Python Garbage Collection (Reference Counting vs Generational GC).
**Context:** Intermittent latency spikes during inference.
**Answer:** Ref counting deletes objects immediately when count hits 0. Generational GC runs periodically to find cyclic references (A points to B, B points to A). The periodic GC pause can cause 100ms latency spikes. 

## Q7: How do you optimize JSON serialization in Python APIs?
**Context:** High latency when returning large feature vectors.
**Answer:** The standard `json` library is slow and written in C but blocks the GIL heavily. Swap it out for `orjson` or `ujson` which are heavily optimized Rust/C++ implementations that parse massive arrays 10x faster.

## Q8: Threading vs Asyncio vs Multiprocessing in ML.
**Context:** Deciding how to parallelize feature fetching.
**Answer:** Threading/Asyncio: Good for I/O bound tasks (fetching from Redis). Asyncio is lighter weight than threads. Multiprocessing: Good for CPU bound tasks (Pandas math) because it bypasses the GIL by spawning new OS processes.

## Q9: Managing Memory Fragmentation in Pandas.
**Context:** Feature engineering script OOMing on a 32GB RAM node with a 5GB CSV.
**Answer:** Pandas allocates contiguous memory blocks. Dropping/adding columns fragments memory. Fix: Use `df.copy()` after heavy filtering to consolidate memory, use categorical datatypes for strings, or switch to Polars for Zero-Copy memory models (Apache Arrow).

## Q10: Python C-Extensions and the GIL.
**Context:** Why PyTorch is fast despite Python.
**Answer:** Python is just a wrapper. When you call `torch.matmul()`, execution drops into a C++ compiled binary. The C++ code explicitly calls `Py_BEGIN_ALLOW_THREADS`, releasing the GIL so other Python threads can run while the matrix math computes.

## Q11: Managing CUDA Contexts in Multiprocessing.
**Context:** PyTorch crashing when using `multiprocessing.Pool`.
**Answer:** You cannot initialize CUDA in the parent process and then fork it. The child processes will inherit an invalid CUDA context and crash. Fix: Set the multiprocessing start method to `spawn` instead of `fork` before importing PyTorch.

## Q12: Vectorization vs List Comprehensions vs For Loops.
**Context:** Data engineer wrote a loop to apply a formula to 10M rows.
**Answer:** For loops in Python are interpreted and incredibly slow. List comprehensions are slightly faster. Vectorization (NumPy/Pandas) pushes the loop down to C, applying the SIMD instruction set, making it 100x faster.

## Q13: Debugging Segmentation Faults in Python.
**Context:** Script just prints `Segmentation fault (core dumped)` and dies.
**Answer:** Python code cannot natively segfault. This means a C-extension (NumPy, PyTorch) crashed. Use `gdb --args python script.py` or `faulthandler.enable()` to dump the C-level stack trace and find the offending library.

## Q14: Memory Mapping (`mmap`) for large datasets.
**Context:** Need to load a 100GB dataset on a 16GB machine.
**Answer:** Instead of loading into RAM, use `numpy.memmap`. It treats the file on disk like an array in memory, loading only the requested chunks into RAM via the OS page cache.

## Q15: Dependency Hell: Handling conflicting C-libraries (glibc/libstdc++).
**Context:** `ImportError: /lib64/libstdc++.so.6: version GLIBCXX_3.4.29 not found`.
**Answer:** The Python wheel was compiled on a newer OS than the Docker container running it. Fix: Ensure the build environment matches the runtime, use `manylinux` wheels, or use Conda which packages its own C-libraries.

## Q16: Profiling CPU usage in Python.
**Context:** Finding the slow function in an ML pipeline.
**Answer:** Do not use `time.time()` prints. Use `cProfile` for deterministic profiling, or `py-spy` / `Austin` for sampling profiling. A sampling profiler takes snapshots of the call stack at 100Hz without slowing down the application significantly.

## Q17: Using `__slots__` to save memory.
**Context:** Creating millions of small custom objects representing features.
**Answer:** By default, every Python object has a `__dict__` to store dynamic attributes, which is memory-heavy. Defining `__slots__ = ['feat_1', 'feat_2']` prevents the creation of the dict, saving ~40% memory per object.

## Q18: What is PyPy and why isn't it used for Deep Learning?
**Context:** Trying to speed up Python with a JIT compiler.
**Answer:** PyPy has a Just-In-Time compiler that makes pure Python code much faster. However, it is highly incompatible with C-extensions (like PyTorch and NumPy) because it implements the C-API differently, making it useless for modern MLOps.

## Q19: Using `dataclasses` and `Pydantic` for ML data validation.
**Context:** Guaranteeing inference input schemas.
**Answer:** Raw dictionaries are dangerous. Use Pydantic to define the strict schema (types, min/max values) for the ML endpoint. It automatically handles type coercion, validation, and returns clear 422 errors for bad payloads.

## Q20: Python Virtual Environments vs Docker.
**Context:** Data scientist "works on my machine" issues.
**Answer:** Virtual environments (`venv`, `conda`) only isolate Python packages. They do not isolate system dependencies (CUDA, apt packages, OS version). Docker isolates everything down to the OS kernel, making it the only acceptable artifact for production MLOps.

## Q21: Avoiding the "Thundering Herd" in Python connection pools.
**Context:** 100 Gunicorn workers all reconnecting to Redis at once on boot.
**Answer:** Add random jitter to the initialization logic. Instead of connecting instantly, wait `random.uniform(0, 5)` seconds. This staggers the connection requests and prevents overwhelming the Redis server.

## Q22: Understanding Python's `import` mechanics.
**Context:** Booting an ML app takes 30 seconds.
**Answer:** Importing heavy libraries (TensorFlow) executes hundreds of files. Use lazy loading (importing inside the function that needs it) if a library is only used rarely. This speeds up CLI tools and worker boot times.

## Q23: Building custom Python Wheels for internal MLOps tools.
**Context:** Sharing code between training and serving repos.
**Answer:** Package shared logic (feature engineering, custom metrics) into a standard Python wheel (`.whl`). Publish it to a private PyPI server (Artifactory). This versions the logic and prevents copy-pasting code between repositories.

## Q24: What is `cython` and when should MLOps engineers use it?
**Context:** Custom C++ layer needed for an ML model.
**Answer:** Cython allows you to write Python-like code that compiles to pure C/C++. Use it when you have a custom math operation (e.g., a complex graph traversal) that cannot be vectorized in NumPy and needs C-level speed.

## Q25: Handling timezone data correctly in ML.
**Context:** Model trained on UTC, but receiving local time from mobile app.
**Answer:** Timezone bugs cause massive data leakage/drift. Rule: All timestamps must be converted to UTC ISO-8601 *before* they enter the feature store. The ML model should only ever see UTC.

## Q26: Using Apache Arrow vs Pandas.
**Context:** Passing data between Python and a JVM system.
**Answer:** Pandas serializes data. Arrow is a cross-language, zero-copy, in-memory columnar format. You can pass an Arrow table from Java to Python without any serialization overhead, crucial for high-speed feature serving.

## Q27: Context Managers (`with` statements) in ML.
**Context:** Preventing DB connection leaks.
**Answer:** Always use context managers when opening files, connecting to Redis, or allocating GPU memory. They guarantee the `__exit__` method is called (closing the connection or freeing memory) even if an exception is thrown.

## Q28: How do you mock external ML dependencies for testing?
**Context:** CI pipeline failing because it can't reach the real Feature Store.
**Answer:** Use Python's `unittest.mock`. Patch the `redis.get` call to return a hardcoded JSON string. This isolates the inference logic from the infrastructure, ensuring unit tests run fast and without network access.

## Q29: Optimizing the Python Logger for high throughput.
**Context:** `logging.info()` slowing down inference by 10ms.
**Answer:** String formatting is evaluated even if the log level is disabled (`logger.debug(f"Data: {huge_tensor}")`). Pass arguments dynamically (`logger.debug("Data: %s", huge_tensor)`) so formatting is skipped if DEBUG is off.

## Q30: Why Python 3.11/3.12 upgrades matter for MLOps.
**Context:** Staying on ancient Python 3.7.
**Answer:** Python 3.11 introduced the "Specializing Adaptive Interpreter", providing a free 10-60% speedup on pure Python code. Upgrading Python versions is often the cheapest way to drop latency without rewriting architecture.
