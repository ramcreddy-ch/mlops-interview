# Section 6: Model Serving — Production Inference at Scale
## Tools: KServe, Triton Inference Server, vLLM, FastAPI

---

## 6.1 The Three Inference Patterns — Choose Based on Business Need

```
┌────────────────────┬──────────────────────┬────────────────────────┐
│  REAL-TIME         │  BATCH               │  STREAMING             │
│  INFERENCE         │  INFERENCE           │  INFERENCE             │
├────────────────────┼──────────────────────┼────────────────────────┤
│ User waits for     │ Pre-compute for all  │ Continuous scoring on  │
│ immediate response │ users offline        │ event stream           │
├────────────────────┼──────────────────────┼────────────────────────┤
│ Latency: <100ms    │ Latency: Hours       │ Latency: <1s           │
│ Cost: High         │ Cost: Low            │ Cost: Medium           │
│ Compute: 24/7      │ Compute: On-demand   │ Compute: 24/7          │
├────────────────────┼──────────────────────┼────────────────────────┤
│ Use: Fraud detect  │ Use: Churn predict   │ Use: Ad scoring        │
│ Use: Search rank   │ Use: Reco (cached)   │ Use: Anomaly detect    │
│ Use: Chatbot       │ Use: Risk scoring    │ Use: Click prediction  │
├────────────────────┼──────────────────────┼────────────────────────┤
│ Stack: KServe,     │ Stack: Spark,        │ Stack: Flink +         │
│ Triton, FastAPI    │ Airflow, SageMaker   │ Redis + FastAPI        │
│ + Redis features   │ Batch Transform      │                        │
└────────────────────┴──────────────────────┴────────────────────────┘
```

---

## 6.2 FastAPI Inference Server — Production-Grade Implementation

Most teams start here. FastAPI is not just for prototyping — with the right design it handles millions of requests/day.

```python
# serve.py — Production FastAPI inference server
import os
import time
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
import redis
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.responses import Response
import uvicorn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Prometheus Metrics ─────────────────────────────────────────────────────────
INFERENCE_REQUESTS = Counter(
    "ml_inference_requests_total",
    "Total inference requests",
    ["model_version", "status"]
)
INFERENCE_LATENCY = Histogram(
    "ml_inference_latency_seconds",
    "Inference latency in seconds",
    ["model_version"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0]
)
FEATURE_FETCH_LATENCY = Histogram(
    "feature_store_latency_seconds",
    "Feature store fetch latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
)
PREDICTION_SCORE = Histogram(
    "ml_prediction_score",
    "Distribution of prediction scores",
    ["model_version"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
MODEL_VERSION_GAUGE = Gauge("ml_model_version", "Current model version")
FALLBACK_COUNTER = Counter("ml_fallback_total", "Times fallback was triggered")
NULL_FEATURE_COUNTER = Counter(
    "ml_null_features_total",
    "Number of null features in requests",
    ["feature_name"]
)

# ── Request / Response Models ──────────────────────────────────────────────────
class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., min_length=1, max_length=64)
    user_id: str = Field(..., min_length=1, max_length=64)
    amount: float = Field(..., gt=0, lt=1_000_000)
    merchant_id: str
    currency: str = Field(..., regex="^[A-Z]{3}$")
    timestamp: int  # Unix epoch milliseconds

    @validator("amount")
    def amount_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Amount must be positive")
        return round(v, 2)


class BatchTransactionRequest(BaseModel):
    transactions: List[TransactionRequest] = Field(..., max_items=100)


class FraudPrediction(BaseModel):
    transaction_id: str
    fraud_probability: float
    fraud_flag: bool
    risk_tier: str  # LOW / MEDIUM / HIGH / CRITICAL
    model_version: str
    latency_ms: float
    fallback_used: bool = False


class BatchFraudResponse(BaseModel):
    predictions: List[FraudPrediction]
    total_latency_ms: float
    model_version: str


# ── Global state (loaded once at startup) ─────────────────────────────────────
class ModelServer:
    model: Optional[xgb.XGBClassifier] = None
    model_version: str = "unknown"
    redis_client: Optional[redis.Redis] = None
    feature_columns: List[str] = []
    fallback_features: Optional[np.ndarray] = None


server = ModelServer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and connections on startup. Clean up on shutdown."""
    logger.info("🚀 Starting Fraud Detection Inference Server...")

    # ── Load model from MLflow Registry ───────────────────────────────────
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    model_uri = "models:/fraud-detection-model/Production"

    start = time.time()
    server.model = mlflow.xgboost.load_model(model_uri)
    load_time = time.time() - start
    logger.info(f"✅ Model loaded in {load_time:.2f}s from {model_uri}")

    # Get model version from registry
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions("fraud-detection-model", stages=["Production"])
    server.model_version = versions[0].version if versions else "unknown"
    MODEL_VERSION_GAUGE.set(float(server.model_version) if server.model_version.isdigit() else 0)

    # ── Connect to Redis Feature Store ────────────────────────────────────
    server.redis_client = redis.Redis(
        host=os.environ.get("REDIS_HOST", "redis-feature-store"),
        port=int(os.environ.get("REDIS_PORT", 6379)),
        db=0,
        socket_timeout=0.05,          # 50ms timeout — fail fast
        socket_connect_timeout=0.05,
        decode_responses=True,
        max_connections=100,
    )
    server.redis_client.ping()
    logger.info("✅ Redis feature store connected")

    # ── Pre-compute fallback feature vector (safe defaults) ───────────────
    # Used when Redis is down — prevents hard failure
    server.fallback_features = np.array([[
        100.0,   # avg_spend_30d = $100 (average)
        15,      # txn_count_30d = 15 (average)
        500.0,   # max_spend_30d = $500
        0.1,     # failed_login_5m = 0.1
        1.0,     # is_returning_merchant = True (safe assumption)
        0,       # cross_border_flag = False
        # ... 144 more features
    ]])
    logger.info("✅ Fallback features computed")

    logger.info(f"✅ Server ready | Model v{server.model_version}")
    yield

    # ── Cleanup on shutdown ───────────────────────────────────────────────
    if server.redis_client:
        server.redis_client.close()
    logger.info("👋 Shutdown complete")


app = FastAPI(
    title="Fraud Detection Inference API",
    version="3.2.0",
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ── Feature fetching with circuit breaker ─────────────────────────────────────
async def fetch_features_from_store(user_id: str, txn_amount: float) -> tuple:
    """Fetch user features from Redis. Returns (features, fallback_used)."""
    try:
        start = time.time()

        # Pipeline multiple Redis GET calls into one network round-trip
        pipe = server.redis_client.pipeline(transaction=False)
        pipe.get(f"user:{user_id}:avg_spend_30d")
        pipe.get(f"user:{user_id}:txn_count_30d")
        pipe.get(f"user:{user_id}:max_spend_30d")
        pipe.get(f"user:{user_id}:failed_login_5m")
        pipe.get(f"user:{user_id}:last_country")
        # ... more feature keys
        values = pipe.execute()

        fetch_time = time.time() - start
        FEATURE_FETCH_LATENCY.observe(fetch_time)

        # Track null features (upstream data quality monitoring)
        feature_names = [
            "avg_spend_30d", "txn_count_30d", "max_spend_30d",
            "failed_login_5m", "last_country"
        ]
        features = {}
        for name, val in zip(feature_names, values):
            if val is None:
                NULL_FEATURE_COUNTER.labels(feature_name=name).inc()
                features[name] = server.fallback_features[0][feature_names.index(name)]
            else:
                features[name] = float(val)

        # Add request-time features
        features["txn_amount"] = txn_amount
        features["amount_vs_avg_ratio"] = txn_amount / max(features["avg_spend_30d"], 1.0)

        feature_vector = np.array([[features[k] for k in sorted(features.keys())]])
        return feature_vector, False

    except redis.exceptions.TimeoutError:
        logger.warning(f"Redis timeout for user {user_id}. Using fallback features.")
        FALLBACK_COUNTER.inc()
        return server.fallback_features, True

    except redis.exceptions.ConnectionError:
        logger.error("Redis connection lost! Using fallback features.")
        FALLBACK_COUNTER.inc()
        return server.fallback_features, True


def get_risk_tier(probability: float) -> str:
    if probability < 0.3:
        return "LOW"
    elif probability < 0.6:
        return "MEDIUM"
    elif probability < 0.85:
        return "HIGH"
    else:
        return "CRITICAL"


# ── Main inference endpoint ───────────────────────────────────────────────────
@app.post("/v1/predict", response_model=FraudPrediction)
async def predict_single(
    request: TransactionRequest,
    background_tasks: BackgroundTasks,
):
    start_time = time.time()

    if server.model is None:
        raise HTTPException(503, "Model not loaded yet. Try again in a moment.")

    # ── Fetch features ────────────────────────────────────────────────────
    features, fallback_used = await fetch_features_from_store(
        user_id=request.user_id,
        txn_amount=request.amount,
    )

    # ── Run inference ─────────────────────────────────────────────────────
    with INFERENCE_LATENCY.labels(model_version=server.model_version).time():
        fraud_prob = float(server.model.predict_proba(features)[0, 1])

    PREDICTION_SCORE.labels(model_version=server.model_version).observe(fraud_prob)
    INFERENCE_REQUESTS.labels(
        model_version=server.model_version, status="success"
    ).inc()

    latency_ms = (time.time() - start_time) * 1000

    # ── Async logging (don't block response) ─────────────────────────────
    background_tasks.add_task(
        log_prediction_async,
        transaction_id=request.transaction_id,
        user_id=request.user_id,
        features=features.tolist(),
        fraud_probability=fraud_prob,
        model_version=server.model_version,
        latency_ms=latency_ms,
        fallback_used=fallback_used,
    )

    return FraudPrediction(
        transaction_id=request.transaction_id,
        fraud_probability=round(fraud_prob, 6),
        fraud_flag=fraud_prob > 0.85,
        risk_tier=get_risk_tier(fraud_prob),
        model_version=server.model_version,
        latency_ms=round(latency_ms, 2),
        fallback_used=fallback_used,
    )


@app.post("/v1/predict/batch", response_model=BatchFraudResponse)
async def predict_batch(request: BatchTransactionRequest):
    """Batch inference endpoint — processes up to 100 transactions at once."""
    start = time.time()

    # Fetch features for all users in parallel
    feature_tasks = [
        fetch_features_from_store(txn.user_id, txn.amount)
        for txn in request.transactions
    ]
    results = await asyncio.gather(*feature_tasks)

    # Stack all features into a single matrix for batch inference
    feature_matrix = np.vstack([r[0] for r in results])
    fallback_flags = [r[1] for r in results]

    # Single batch prediction call (much faster than N individual calls)
    all_probs = server.model.predict_proba(feature_matrix)[:, 1]

    predictions = []
    for txn, prob, fallback in zip(request.transactions, all_probs, fallback_flags):
        predictions.append(FraudPrediction(
            transaction_id=txn.transaction_id,
            fraud_probability=round(float(prob), 6),
            fraud_flag=float(prob) > 0.85,
            risk_tier=get_risk_tier(float(prob)),
            model_version=server.model_version,
            latency_ms=0,
            fallback_used=fallback,
        ))

    total_latency = (time.time() - start) * 1000
    return BatchFraudResponse(
        predictions=predictions,
        total_latency_ms=round(total_latency, 2),
        model_version=server.model_version,
    )


@app.get("/health/live")
async def liveness():
    """Kubernetes liveness probe — is the process alive?"""
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness():
    """Kubernetes readiness probe — is the model loaded and Redis connected?"""
    if server.model is None:
        raise HTTPException(503, "Model not loaded")
    try:
        server.redis_client.ping()
    except Exception as e:
        # If Redis is down, we can still serve with fallback features
        # So we stay 'ready' but log the degradation
        logger.warning(f"Redis unhealthy: {e}")
    return {"status": "ready", "model_version": server.model_version}


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type="text/plain")


async def log_prediction_async(
    transaction_id: str,
    user_id: str,
    features: list,
    fraud_probability: float,
    model_version: str,
    latency_ms: float,
    fallback_used: bool,
):
    """Fire-and-forget logging to Kafka for monitoring pipeline."""
    import json
    # In production, this writes to Kafka for the monitoring pipeline
    log_entry = {
        "timestamp": int(time.time() * 1000),
        "transaction_id": transaction_id,
        "user_id": user_id,  # In GDPR environments, hash this!
        "fraud_probability": fraud_probability,
        "model_version": model_version,
        "latency_ms": latency_ms,
        "fallback_used": fallback_used,
        # Note: We log features separately for drift monitoring (async Airflow job)
    }
    logger.debug(f"Prediction logged: {json.dumps(log_entry)}")


if __name__ == "__main__":
    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=8080,
        workers=4,           # One worker per CPU core (not for GPU models!)
        loop="uvloop",       # Faster event loop
        access_log=False,    # Disable for high throughput
    )
```

---

## 6.3 KServe — Serverless Model Serving on Kubernetes

KServe (formerly KFServing) provides a standardized, serverless interface for model serving.

```yaml
# kserve-fraud-model.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: fraud-detection-model
  namespace: ml-serving
  annotations:
    serving.kserve.io/deploymentMode: Serverless   # Scale to zero when idle
spec:
  predictor:
    # KServe natively supports MLflow models
    model:
      modelFormat:
        name: mlflow
      storageUri: "s3://fraud-models/production/fraud-model-v3/"
      resources:
        requests:
          cpu: "2"
          memory: "4Gi"
        limits:
          cpu: "4"
          memory: "8Gi"
      # Autoscaling config
      minReplicas: 2        # Never scale below 2 (for availability)
      maxReplicas: 20       # Maximum scale-out
      scaleTarget: 100      # Scale up when concurrency > 100 req/replica
      scaleMetric: concurrency

  # Optional: Add a transformer for pre/post processing
  transformer:
    containers:
      - name: transformer
        image: 123456789.dkr.ecr.us-east-1.amazonaws.com/fraud-transformer:v2
        resources:
          requests:
            cpu: "1"
            memory: "1Gi"

  # Optional: Add an explainer for SHAP explanations
  explainer:
    alibi:
      type: AnchorTabular
      storageUri: "s3://fraud-models/explainer/anchor-tabular-v1/"
      resources:
        requests:
          cpu: "1"
          memory: "2Gi"
```

```bash
# Check inference service status
kubectl get inferenceservice fraud-detection-model -n ml-serving

# Send a test request to KServe
curl -X POST http://fraud-detection-model.ml-serving.svc.cluster.local/v2/models/fraud-detection-model/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{
      "name": "features",
      "shape": [1, 150],
      "datatype": "FP32",
      "data": [1.0, 2.5, 100.0, ...]
    }]
  }'
```

---

## 6.4 NVIDIA Triton Inference Server — For GPU-Heavy Models

Triton is the production standard for GPU model serving. It supports batching, concurrency, and multiple model frameworks simultaneously.

```python
# model_repository/fraud_transformer/config.pbtxt
name: "fraud_transformer"
backend: "pytorch_libtorch"

input [
  {
    name: "INPUT__0"
    data_type: TYPE_FP32
    dims: [-1, 150]    # -1 = dynamic batch size
  }
]

output [
  {
    name: "OUTPUT__0"
    data_type: TYPE_FP32
    dims: [-1, 1]
  }
]

# Dynamic batching: Triton accumulates requests and batches them
dynamic_batching {
  preferred_batch_size: [32, 64, 128]  # Target batch sizes
  max_queue_delay_microseconds: 10000  # Wait up to 10ms to fill batch
}

# Run 2 model instances in parallel on 2 different GPU streams
instance_group [
  {
    kind: KIND_GPU
    count: 2           # 2 concurrent model instances
    gpus: [0]          # Both on GPU 0 (different CUDA streams)
  }
]
```

```python
# triton_client.py — Python client for Triton
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient(
    url="triton.ml-serving.svc.cluster.local:8000"
)

# Check model is ready
assert client.is_model_ready("fraud_transformer")

# Prepare input
features = np.random.randn(1, 150).astype(np.float32)
inputs = [
    httpclient.InferInput("INPUT__0", features.shape, "FP32")
]
inputs[0].set_data_from_numpy(features)

outputs = [httpclient.InferRequestedOutput("OUTPUT__0")]

# Send inference request
response = client.infer(
    model_name="fraud_transformer",
    inputs=inputs,
    outputs=outputs,
)

fraud_prob = response.as_numpy("OUTPUT__0")[0, 0]
print(f"Fraud probability: {fraud_prob:.4f}")
```

---

## 6.5 REST vs gRPC — When to Use What

| Dimension | REST/HTTP | gRPC |
|---|---|---|
| Protocol | HTTP/1.1 or HTTP/2 | HTTP/2 only |
| Payload | JSON (verbose) | Protocol Buffers (compact) |
| Latency overhead | Higher | Lower (3-5x less) |
| Payload size | Larger | Smaller (2-3x) |
| Streaming support | Limited | Native bidirectional |
| Browser support | Native | Requires gRPC-Web proxy |
| Debugging | Easy (curl, Postman) | Harder (need proto tools) |
| Best for | External-facing APIs, mobile | Internal service-to-service |

**Real production pattern:**
```
Mobile App ──REST/JSON──▶ API Gateway ──gRPC──▶ Inference Service
                                                      │
                                           gRPC──▶ Feature Store
                                           gRPC──▶ Decision Service
```

External-facing: REST. Internal: gRPC. This gives you debuggability at the edge and performance internally.

---

## 6.6 Latency Optimization — Production Techniques

```python
# Technique 1: Model quantization (INT8) — 2-4x speedup
import torch
import torch.quantization

model = FraudTransformer().eval()
model.load_state_dict(torch.load("fraud_transformer.pt"))

# Dynamic quantization (easiest, good for CPU inference)
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},   # Only quantize Linear layers
    dtype=torch.qint8,
)

# Measure speedup
import time
x = torch.randn(1, 150)

start = time.time()
for _ in range(1000):
    model(x)
fp32_time = (time.time() - start) / 1000

start = time.time()
for _ in range(1000):
    quantized_model(x)
int8_time = (time.time() - start) / 1000

print(f"FP32: {fp32_time*1000:.2f}ms | INT8: {int8_time*1000:.2f}ms | Speedup: {fp32_time/int8_time:.2f}x")
```

```python
# Technique 2: ONNX export for cross-platform optimization
import torch.onnx

model.eval()
dummy_input = torch.randn(1, 150)

torch.onnx.export(
    model,
    dummy_input,
    "fraud_model.onnx",
    input_names=["features"],
    output_names=["fraud_score"],
    dynamic_axes={"features": {0: "batch_size"}},  # Dynamic batch
    opset_version=17,
)

# Run with ONNX Runtime (2-5x faster than PyTorch for inference)
import onnxruntime as ort
session = ort.InferenceSession(
    "fraud_model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

features_np = dummy_input.numpy()
result = session.run(
    ["fraud_score"],
    {"features": features_np}
)
```

---

*Next → [Section 7: Kubernetes for MLOps](../kubernetes/README.md)*
