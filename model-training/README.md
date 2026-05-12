# Section 5: Model Training at Scale
## Distributed Training, Ray, Spark, GPU/TPU Optimization

---

## 5.1 Why Distributed Training is Required in Production

Single-GPU training limitations:
- A single A100 (80GB VRAM) can train a ~7B parameter model but takes weeks
- Large recommendation models with 100M+ user/item embeddings exceed single GPU memory
- Hyperparameter sweeps need parallelism to be economically viable

**Production reality:** Most companies never need distributed training for inference models (XGBoost, small NNs). They need it for:
1. Large deep learning models (transformers, recommendation systems)
2. Very large datasets that don't fit in memory
3. Hyperparameter tuning at scale

---

## 5.2 Distributed Training Strategies — Know the Difference

```
┌──────────────────────────────────────────────────────────────────┐
│               DISTRIBUTED TRAINING STRATEGIES                     │
├─────────────────────────┬────────────────────────────────────────┤
│   DATA PARALLELISM      │      MODEL PARALLELISM                 │
│                         │                                        │
│ Same model on each GPU  │  Different model layers on each GPU    │
│ Different data batches  │  Same data flows through all GPUs      │
│                         │                                        │
│ GPU1: [data_batch_A]    │  GPU1: [Layer 1-10]                    │
│       → model copy 1    │  GPU2: [Layer 11-20]                   │
│ GPU2: [data_batch_B]    │  GPU3: [Layer 21-30]                   │
│       → model copy 2    │  GPU4: [Layer 31-40]                   │
│                         │                                        │
│ Gradients averaged      │  Pipeline parallelism: GPU1 sends      │
│ across all GPUs         │  activations to GPU2 sequentially      │
│                         │                                        │
│ Best for: Most models   │  Best for: Models > GPU memory         │
│ (XGBoost, ResNet, BERT) │  (GPT-3, LLaMA-70B)                  │
└─────────────────────────┴────────────────────────────────────────┘
```

---

## 5.3 PyTorch Distributed Data Parallel (DDP) — Production Implementation

DDP is the gold standard for multi-GPU training on a single machine or across multiple machines.

```python
# train_ddp.py — Production-grade DDP training script
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import mlflow
import argparse

# ── NCCL environment setup (critical for performance) ─────────────────────────
os.environ["NCCL_DEBUG"] = "WARN"           # Set to INFO for debugging
os.environ["NCCL_IB_DISABLE"] = "0"         # Enable InfiniBand if available
os.environ["NCCL_NET_GDR_LEVEL"] = "5"      # GPU Direct RDMA


def setup_distributed():
    """Initialize distributed training from environment variables."""
    # These are set by torchrun automatically
    rank = int(os.environ["RANK"])           # Global rank (0 to world_size-1)
    local_rank = int(os.environ["LOCAL_RANK"])  # GPU index on this machine
    world_size = int(os.environ["WORLD_SIZE"])  # Total number of processes

    dist.init_process_group(
        backend="nccl",        # Use NCCL for GPU-GPU communication
        init_method="env://",  # Reads MASTER_ADDR, MASTER_PORT from env
    )

    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup():
    dist.destroy_process_group()


class FraudTransformer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=6,
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x.unsqueeze(1)).squeeze(1)
        return self.classifier(x).squeeze(-1)


def main():
    rank, local_rank, world_size = setup_distributed()

    # ── Only log from rank 0 (master process) ─────────────────────────────
    is_master = rank == 0

    if is_master:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment("fraud-distributed-training")
        run = mlflow.start_run(
            run_name=f"ddp-{world_size}gpu-fraud-transformer",
            tags={"world_size": str(world_size), "backend": "nccl"}
        )

    # ── Load dataset with DistributedSampler ──────────────────────────────
    # DistributedSampler ensures each GPU sees different data
    dataset = FraudDataset("s3://fraud-features/train/")
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=256,              # Per-GPU batch size
        sampler=sampler,
        num_workers=4,
        pin_memory=True,             # Speeds up CPU→GPU transfer
        prefetch_factor=2,           # Pre-fetch 2 batches ahead
    )

    # ── Create model and wrap with DDP ────────────────────────────────────
    model = FraudTransformer(input_dim=150, hidden_dim=256, num_heads=8)
    model = model.cuda(local_rank)

    # DDP synchronizes gradients across all GPUs after backward()
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,  # Set True only if needed (slower)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4 * world_size,    # Scale LR linearly with # of GPUs (linear scaling rule)
        weight_decay=0.01,
    )

    # Warmup + cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4 * world_size,
        steps_per_epoch=len(loader),
        epochs=50,
        pct_start=0.05,   # 5% warmup
    )

    pos_weight = torch.tensor([99.0]).cuda(local_rank)  # Class imbalance
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = 0.0

    for epoch in range(50):
        model.train()
        sampler.set_epoch(epoch)   # Ensures different shuffle each epoch

        epoch_loss = 0.0
        for step, (features, labels) in enumerate(loader):
            features = features.cuda(local_rank, non_blocking=True)
            labels = labels.float().cuda(local_rank, non_blocking=True)

            # Mixed precision training (automatic loss scaling)
            with torch.cuda.amp.autocast():
                logits = model(features)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()

            # Gradient clipping (important for transformer stability)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            scheduler.step()

            epoch_loss += loss.item()

        # ── Only master logs metrics ───────────────────────────────────
        if is_master:
            avg_loss = epoch_loss / len(loader)
            mlflow.log_metrics({"train/loss": avg_loss}, step=epoch)

            # Save checkpoint
            if epoch % 5 == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),  # .module for DDP
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                }, f"s3://fraud-models/checkpoints/epoch_{epoch}.pt")

    if is_master:
        mlflow.end_run()

    cleanup()


if __name__ == "__main__":
    # GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()
    main()
```

### Launch Command — Multi-Node Multi-GPU:

```bash
# Single machine, 4 GPUs (most common):
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    train_ddp.py

# Multi-node (2 machines, 4 GPUs each = 8 total):
# On NODE 0 (master):
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="10.0.0.1" \
    --master_port=29500 \
    train_ddp.py

# On NODE 1 (worker):
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="10.0.0.1" \
    --master_port=29500 \
    train_ddp.py

# On Kubernetes using Training Operator (PyTorchJob):
# (automatically sets MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE)
kubectl apply -f pytorch-training-job.yaml
```

### Kubernetes PyTorchJob Manifest:

```yaml
# pytorch-training-job.yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: fraud-transformer-training
  namespace: ml-training
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: 123456789.dkr.ecr.us-east-1.amazonaws.com/fraud-trainer:v3.2
              command:
                - torchrun
                - --nproc_per_node=4
                - train_ddp.py
              resources:
                requests:
                  nvidia.com/gpu: 4
                  memory: "64Gi"
                  cpu: "16"
                limits:
                  nvidia.com/gpu: 4
                  memory: "64Gi"
              env:
                - name: MLFLOW_TRACKING_URI
                  value: "http://mlflow.ml-platform.svc.cluster.local:5000"
                - name: NCCL_DEBUG
                  value: "WARN"
              volumeMounts:
                - name: shared-storage
                  mountPath: /shared
          nodeSelector:
            nvidia.com/gpu: "true"
          tolerations:
            - key: nvidia.com/gpu
              operator: Exists
              effect: NoSchedule
          volumes:
            - name: shared-storage
              persistentVolumeClaim:
                claimName: training-pvc
    Worker:
      replicas: 1    # 2 total machines (Master + 1 Worker) = 8 GPUs
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: 123456789.dkr.ecr.us-east-1.amazonaws.com/fraud-trainer:v3.2
              command:
                - torchrun
                - --nproc_per_node=4
                - train_ddp.py
              resources:
                requests:
                  nvidia.com/gpu: 4
                  memory: "64Gi"
```

---

## 5.4 Ray — Scalable Hyperparameter Tuning + Distributed Training

Ray is the Swiss Army knife of distributed ML: hyperparameter tuning, data processing, training, and serving all in one framework.

```python
# ray_tune_fraud.py — Production HPO with Ray Tune
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import xgboost as xgb
import pandas as pd
from sklearn.metrics import average_precision_score
import mlflow

ray.init(address="ray://ray-cluster.internal:10001")   # Connect to Ray cluster

def train_xgboost(config: dict):
    """Training function called by each Ray worker."""
    # Each worker gets different hyperparameters from config
    df = pd.read_parquet("s3://fraud-features/train/")
    X = df.drop(["is_fraud", "transaction_id"], axis=1)
    y = df["is_fraud"]

    split = int(0.8 * len(df))
    X_tr, X_val = X.iloc[:split], X.iloc[split:]
    y_tr, y_val = y.iloc[:split], y.iloc[split:]

    model = xgb.XGBClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        scale_pos_weight=99,
        random_state=42,
        eval_metric="aucpr",
        early_stopping_rounds=30,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    preds = model.predict_proba(X_val)[:, 1]
    auc_pr = average_precision_score(y_val, preds)

    # Report metric back to Ray Tune scheduler
    tune.report(auc_pr=auc_pr, done=True)


# ── Configure search space ────────────────────────────────────────────────────
search_space = {
    "n_estimators": tune.randint(100, 600),
    "max_depth": tune.randint(3, 10),
    "learning_rate": tune.loguniform(1e-4, 0.3),
    "subsample": tune.uniform(0.6, 1.0),
    "colsample_bytree": tune.uniform(0.6, 1.0),
}

# ── ASHA Scheduler — kills bad trials early ───────────────────────────────────
scheduler = ASHAScheduler(
    metric="auc_pr",
    mode="max",
    max_t=100,              # Max iterations
    grace_period=10,        # Minimum iterations before killing
    reduction_factor=3,     # Kill bottom 2/3 of trials
)

# ── Optuna for Bayesian optimization (smarter than random search) ─────────────
search_alg = OptunaSearch(metric="auc_pr", mode="max")

# ── Run 200 trials in parallel ────────────────────────────────────────────────
analysis = tune.run(
    train_xgboost,
    config=search_space,
    num_samples=200,          # Total trials
    scheduler=scheduler,
    search_alg=search_alg,
    resources_per_trial={"cpu": 4, "gpu": 0},  # Each trial gets 4 CPUs
    max_concurrent_trials=50,  # 50 trials running simultaneously
    storage_path="s3://fraud-ray-results/",
    name="fraud-hpo-nov2024",
    verbose=1,
)

# Get best config
best_config = analysis.get_best_config(metric="auc_pr", mode="max")
best_auc = analysis.get_best_trial(metric="auc_pr", mode="max").last_result["auc_pr"]

print(f"Best config: {best_config}")
print(f"Best AUC-PR: {best_auc:.4f}")

# Log best run to MLflow
with mlflow.start_run(run_name="best-hpo-config"):
    mlflow.log_params(best_config)
    mlflow.log_metric("best_auc_pr", best_auc)
```

---

## 5.5 GPU/TPU Selection — Cost vs Performance Guide

```
Training Workload Decision Tree:
                     │
                     ▼
         Is model > 80GB parameters?
         ┌──────┤
         Yes    No
         │      │
         ▼      ▼
    Model       Is it a Transformer?
    Parallelism ┌──────┤
    (A100 80GB) Yes    No
    or TPU      │      │
                ▼      ▼
           TPU v4    XGBoost/
           (GCP)     LightGBM/
           or        Sklearn?
           A100      │
           cluster   ▼
                  CPU cluster
                  (no GPU needed!)
```

### GPU Cost Comparison (AWS, May 2024):

| Instance | GPU | VRAM | On-Demand/hr | Spot/hr | Best For |
|---|---|---|---|---|---|
| g4dn.xlarge | 1x T4 | 16GB | $0.526 | ~$0.16 | Dev, small models |
| g4dn.12xlarge | 4x T4 | 64GB | $3.912 | ~$1.17 | Medium DL models |
| p3.2xlarge | 1x V100 | 16GB | $3.06 | ~$0.92 | Training |
| p3.8xlarge | 4x V100 | 64GB | $12.24 | ~$3.67 | Distributed training |
| p4d.24xlarge | 8x A100 | 320GB | $32.77 | ~$9.83 | Large model training |
| p4de.24xlarge | 8x A100 | 640GB | $40.97 | ~$12.29 | Very large models |

**Production Strategy:** Always use Spot for training (80% cost reduction). Implement checkpointing to handle interruptions.

---

## 5.6 Mixed Precision Training — Critical for Production GPU Utilization

Without mixed precision, A100 GPU is used at 30-40% efficiency. With it, 80-90%.

```python
# mixed_precision.py — Automatic Mixed Precision (AMP)
import torch
from torch.cuda.amp import GradScaler, autocast

model = FraudTransformer().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# GradScaler prevents underflow when gradients become very small in FP16
scaler = GradScaler()

for batch in train_loader:
    features, labels = batch
    features = features.cuda()
    labels = labels.float().cuda()

    optimizer.zero_grad(set_to_none=True)

    # Forward pass in FP16 (uses Tensor Cores on Ampere GPUs)
    with autocast(dtype=torch.float16):
        logits = model(features)
        loss = criterion(logits, labels)

    # Backward pass with scaled gradients
    scaler.scale(loss).backward()

    # Unscale, clip, and step
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

**Memory savings from AMP:**
- FP32 model (baseline): 4 bytes/param → 300M params = 1.2GB just for weights
- FP16 with AMP: 2 bytes/param → 300M params = 600MB + overhead
- Allows 2x larger batch sizes or 2x larger models on same GPU

---

*Next → [Section 6: Model Serving](../model-serving/README.md)*
