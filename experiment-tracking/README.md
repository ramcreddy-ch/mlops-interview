# Section 4: Experiment Tracking
## Tools: MLflow, Weights & Biases | Model Registry Workflows

---

## 4.1 Why Experiment Tracking is Non-Negotiable in Production

Without tracking, this happens:
- DS runs 200 experiments in Jupyter notebooks over 3 weeks
- The best model was in `notebook_v7_FINAL_really_final.ipynb`, run #47
- Nobody can reproduce it because the random seed wasn't set
- The second-best model gets deployed instead
- 3 months later, the DS leaves the company

**Experiment tracking makes every run reproducible, comparable, and auditable.**

---

## 4.2 MLflow — Complete Production Setup

### Architecture of MLflow in Production:

```
┌──────────────────────────────────────────────────────────┐
│                     MLflow Components                     │
├─────────────────┬────────────────┬───────────────────────┤
│  Tracking Server│  Model Registry│     Artifact Store    │
│  (FastAPI app)  │  (PostgreSQL)  │     (S3 Bucket)       │
│                 │                │                       │
│  - Experiments  │  - Model names │  - Model weights      │
│  - Runs         │  - Versions    │  - Plots              │
│  - Metrics      │  - Stages      │  - Datasets           │
│  - Parameters   │  (None→Staging │  - Custom artifacts   │
│  - Tags         │   →Production) │                       │
└─────────────────┴────────────────┴───────────────────────┘
         │                │                    │
         └────────────────┴────────────────────┘
                          │
              PostgreSQL (metadata) + S3 (artifacts)
```

### Production MLflow Server Setup (Docker Compose):

```yaml
# docker-compose-mlflow.yml
version: "3.8"

services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.10.0
    ports:
      - "5000:5000"
    environment:
      MLFLOW_BACKEND_STORE_URI: postgresql://mlflow:${DB_PASS}@postgres:5432/mlflow
      MLFLOW_DEFAULT_ARTIFACT_ROOT: s3://company-mlflow-artifacts/
      AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID}
      AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY}
    command: >
      mlflow server
        --backend-store-uri postgresql://mlflow:${DB_PASS}@postgres:5432/mlflow
        --default-artifact-root s3://company-mlflow-artifacts/
        --host 0.0.0.0
        --port 5000
        --workers 4
        --gunicorn-opts "--timeout 300"
    depends_on:
      - postgres

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: mlflow
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: ${DB_PASS}
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### Full Production Training Script with MLflow:

```python
# train.py — Production-grade training with complete MLflow logging
import mlflow
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import shap
import argparse
import json
import os

def train_fraud_model(
    features_path: str,
    model_name: str = "fraud-detection-model",
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    scale_pos_weight: float = 99.0,
    training_date: str = None,
):
    # ── Connect to MLflow ──────────────────────────────────────────────────
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run(
        run_name=f"xgboost-v3-{training_date}",
        tags={
            "model_family": "xgboost",
            "data_date": training_date,
            "team": "fraud-platform",
            "git_commit": os.environ.get("GIT_SHA", "unknown"),
            "environment": os.environ.get("ENV", "development"),
        }
    ) as run:

        # ── Log all parameters upfront ─────────────────────────────────────
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "scale_pos_weight": scale_pos_weight,
            "features_path": features_path,
            "training_date": training_date,
            "cv_folds": 5,
            "random_seed": 42,
        })

        # ── Load data ──────────────────────────────────────────────────────
        df = pd.read_parquet(features_path)
        feature_cols = [c for c in df.columns
                        if c not in ["is_fraud", "transaction_id", "date"]]
        X, y = df[feature_cols], df["is_fraud"]

        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("fraud_rate", float(y.mean()))

        # ── Cross-validation (5-fold stratified) ──────────────────────────
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_auc_prs, cv_aucs = [], []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                scale_pos_weight=scale_pos_weight,
                eval_metric=["logloss", "aucpr"],
                early_stopping_rounds=50,
                random_state=42,
            )
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            preds = model.predict_proba(X_val)[:, 1]
            fold_auc_pr = average_precision_score(y_val, preds)
            fold_auc = roc_auc_score(y_val, preds)

            cv_auc_prs.append(fold_auc_pr)
            cv_aucs.append(fold_auc)

            # Log per-fold metrics
            mlflow.log_metric("cv_auc_pr", fold_auc_pr, step=fold)
            mlflow.log_metric("cv_roc_auc", fold_auc, step=fold)

        # ── Log aggregate CV metrics ───────────────────────────────────────
        mlflow.log_metrics({
            "cv_auc_pr_mean": np.mean(cv_auc_prs),
            "cv_auc_pr_std": np.std(cv_auc_prs),
            "cv_roc_auc_mean": np.mean(cv_aucs),
            "cv_roc_auc_std": np.std(cv_aucs),
        })

        # ── Train final model on full training set ─────────────────────────
        X_train = X.iloc[:int(0.8*len(X))]
        X_test = X.iloc[int(0.8*len(X)):]
        y_train = y.iloc[:int(0.8*len(y))]
        y_test = y.iloc[int(0.8*len(y)):]

        final_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
        )
        final_model.fit(X_train, y_train)

        # ── Evaluate on holdout test set ───────────────────────────────────
        test_preds_proba = final_model.predict_proba(X_test)[:, 1]
        test_preds_binary = (test_preds_proba > 0.5).astype(int)

        test_metrics = {
            "test_roc_auc": roc_auc_score(y_test, test_preds_proba),
            "test_auc_pr": average_precision_score(y_test, test_preds_proba),
            "test_f1": f1_score(y_test, test_preds_binary),
            "test_precision": precision_score(y_test, test_preds_binary),
            "test_recall": recall_score(y_test, test_preds_binary),
        }
        mlflow.log_metrics(test_metrics)

        # ── Generate and log SHAP feature importance plot ──────────────────
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_test.head(1000))

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test.head(1000), show=False)
        plt.tight_layout()
        plt.savefig("/tmp/shap_summary.png", dpi=150)
        plt.close()
        mlflow.log_artifact("/tmp/shap_summary.png", "plots")

        # ── Log feature importance ─────────────────────────────────────────
        importance = dict(zip(
            feature_cols,
            final_model.feature_importances_
        ))
        importance_sorted = sorted(importance.items(), key=lambda x: -x[1])
        with open("/tmp/feature_importance.json", "w") as f:
            json.dump(importance_sorted[:20], f, indent=2)
        mlflow.log_artifact("/tmp/feature_importance.json", "metrics")

        # ── Log the model with input/output signature ──────────────────────
        from mlflow.models.signature import infer_signature
        signature = infer_signature(
            X_train,
            final_model.predict_proba(X_train)
        )

        # Log with conda environment for reproducibility
        mlflow.xgboost.log_model(
            final_model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(3),
            registered_model_name=model_name,
            conda_env={
                "channels": ["defaults"],
                "dependencies": [
                    f"python={os.sys.version_info.major}.{os.sys.version_info.minor}",
                    {"pip": [
                        f"xgboost=={xgb.__version__}",
                        f"mlflow=={mlflow.__version__}",
                        "pandas>=1.5.0",
                        "scikit-learn>=1.2.0",
                    ]}
                ]
            }
        )

        print(f"✅ Run ID: {run.info.run_id}")
        print(f"✅ Test AUC-PR: {test_metrics['test_auc_pr']:.4f}")
        print(f"✅ Test ROC-AUC: {test_metrics['test_roc_auc']:.4f}")

        return run.info.run_id, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", required=True)
    parser.add_argument("--training-date", required=True)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    args = parser.parse_args()

    run_id, metrics = train_fraud_model(
        features_path=args.features_path,
        training_date=args.training_date,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
```

---

## 4.3 MLflow Model Registry — Lifecycle State Machine

```
                 ┌─────────────────────────────────────┐
                 │          MODEL REGISTRY              │
                 │                                      │
  Training ───▶  │  None ──▶ Staging ──▶ Production    │
  (auto-log)     │                  ↘                  │
                 │                   Archived           │
                 └─────────────────────────────────────┘
```

### Automating Registry Transitions:

```python
# registry_manager.py — Automates model promotion workflow
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="http://mlflow.internal:5000")

MODEL_NAME = "fraud-detection-model"

def promote_to_staging_if_better(run_id: str, threshold_auc_pr: float = 0.75):
    """Automatically promote model to Staging if it beats the threshold."""

    run = client.get_run(run_id)
    auc_pr = run.data.metrics.get("test_auc_pr", 0)

    if auc_pr < threshold_auc_pr:
        print(f"❌ Model AUC-PR {auc_pr:.4f} < threshold {threshold_auc_pr}. Not promoting.")
        return None

    # Find the model version created in this run
    versions = client.search_model_versions(f"run_id='{run_id}'")
    if not versions:
        raise ValueError(f"No model version found for run {run_id}")

    version = versions[0].version

    # Transition to Staging
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Staging",
        archive_existing_versions=False,  # Keep old staging for comparison
    )

    # Add description
    client.update_model_version(
        name=MODEL_NAME,
        version=version,
        description=f"Auto-promoted | AUC-PR={auc_pr:.4f} | Run={run_id[:8]}"
    )

    print(f"✅ Model v{version} promoted to Staging (AUC-PR={auc_pr:.4f})")
    return version


def champion_challenger_promote(challenger_run_id: str):
    """Compare challenger to current Production champion."""

    # Get current production model metrics
    prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])

    if prod_versions:
        champion = prod_versions[0]
        champion_run = client.get_run(champion.run_id)
        champion_auc = champion_run.data.metrics.get("test_auc_pr", 0)
    else:
        champion_auc = 0  # No champion yet
        print("No production model exists yet. Any model can be promoted.")

    # Get challenger metrics
    challenger_run = client.get_run(challenger_run_id)
    challenger_auc = challenger_run.data.metrics.get("test_auc_pr", 0)

    print(f"Champion AUC-PR: {champion_auc:.4f}")
    print(f"Challenger AUC-PR: {challenger_auc:.4f}")

    improvement = challenger_auc - champion_auc
    MIN_IMPROVEMENT = 0.005  # Must improve by at least 0.5%

    if improvement >= MIN_IMPROVEMENT:
        # Promote challenger to production
        challenger_versions = client.search_model_versions(
            f"run_id='{challenger_run_id}'"
        )
        challenger_version = challenger_versions[0].version

        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=challenger_version,
            stage="Production",
            archive_existing_versions=True,  # Archive old production model
        )
        print(f"✅ Challenger v{challenger_version} promoted to Production! (+{improvement:.4f})")
    else:
        print(f"❌ Challenger not better enough (+{improvement:.4f} < {MIN_IMPROVEMENT}). Keeping champion.")


def get_production_model():
    """Get the current production model for serving."""
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not versions:
        raise ValueError("No production model available!")

    version = versions[0]
    model_uri = f"models:/{MODEL_NAME}/Production"
    model = mlflow.xgboost.load_model(model_uri)

    return model, version.version, version.run_id
```

---

## 4.4 Weights & Biases (W&B) — When to Use It Over MLflow

W&B is the tool of choice when:
- You have **many concurrent experiments** and need visual comparison
- You need **real-time training curves** (loss per step, not just per epoch)
- You need **collaborative experiment sharing** across teams or externally
- You train **deep learning models** (image/text/audio) and need media logging

```python
# train_wandb.py — W&B integration for deep learning
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Initialize W&B run
run = wandb.init(
    project="fraud-deep-learning",
    name=f"transformer-v2-{training_date}",
    config={
        "architecture": "FraudTransformer",
        "learning_rate": 1e-4,
        "batch_size": 512,
        "epochs": 50,
        "hidden_dim": 256,
        "num_heads": 8,
        "dropout": 0.1,
        "warmup_steps": 1000,
    },
    tags=["transformer", "production-candidate", "nov-2024"],
    notes="Testing attention mechanism on transaction sequences",
)

# Watch model gradients automatically
wandb.watch(model, log="all", log_freq=100)

for epoch in range(config.epochs):
    model.train()
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        loss, logits = model(**batch)
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        # Log every 50 steps for real-time visibility
        if step % 50 == 0:
            wandb.log({
                "train/loss": loss.item(),
                "train/grad_norm": grad_norm,
                "train/learning_rate": scheduler.get_last_lr()[0],
                "epoch": epoch,
                "step": step,
            })

    # ── Validation ──────────────────────────────────────────────────────
    model.eval()
    val_metrics = evaluate(model, val_loader)
    wandb.log({
        "val/auc_pr": val_metrics["auc_pr"],
        "val/roc_auc": val_metrics["roc_auc"],
        "val/loss": val_metrics["loss"],
        "epoch": epoch,
    })

    # Log sample predictions (visual inspection)
    sample_preds = predict_samples(model, val_loader, n=50)
    wandb.log({
        "predictions": wandb.Table(
            columns=["transaction_id", "true_label", "pred_score"],
            data=sample_preds
        )
    })

    # Save checkpoint to W&B artifacts
    torch.save(model.state_dict(), f"/tmp/model_epoch_{epoch}.pt")
    artifact = wandb.Artifact(
        name=f"fraud-transformer-epoch-{epoch}",
        type="model",
        metadata=val_metrics
    )
    artifact.add_file(f"/tmp/model_epoch_{epoch}.pt")
    run.log_artifact(artifact)

# Mark best model
run.summary["best_val_auc_pr"] = max(all_val_auc_prs)
run.finish()
```

### MLflow vs W&B Side-by-Side Comparison:

| Feature | MLflow | W&B |
|---|---|---|
| Self-hosted | ✅ Free | ✅ (W&B Server, paid) |
| Cloud hosted | ✅ (Databricks managed) | ✅ (wandb.ai, free tier) |
| Real-time curves | Basic | Excellent |
| Media logging | Basic | Excellent (images, audio, video) |
| Model registry | ✅ Production-grade | ✅ Good |
| Sweep/HPO | Basic | Excellent (W&B Sweeps) |
| Team collab | Basic | Excellent |
| Framework support | All | All |
| HIPAA compliance | ✅ (self-hosted) | ✅ (enterprise) |
| Price | Free (self-hosted) | Free tier limited |
| Best for | Traditional ML, platform teams | DL teams, research |

---

*Next → [Section 5: Model Training at Scale](../model-training/README.md)*
