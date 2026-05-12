# Section 2: Versioning & Data Management
## Tools: DVC, Git, LakeFS | Real Project: Fraud Detection

---

## 2.1 Why ML Versioning is 10x Harder Than Software

In software, version `code`. In ML, version ALL FIVE:

```
1. CODE          → Git (training scripts, feature logic, serving code)
2. DATA          → DVC / LakeFS (training datasets, feature tables)
3. MODEL         → MLflow Registry (weights, hyperparameters, metrics)
4. ENVIRONMENT   → Docker + requirements.txt (Python deps, CUDA version)
5. CONFIG        → Git + params.yaml (hyperparameters, thresholds)
```

**Real Production Failure — Why This Matters:**
> A fraud model performed at AUC 0.92 in Q1. In Q2, a data engineer deleted the training S3 bucket to save storage. Six months later, compliance audited a disputed transaction. The company could not reproduce the model that made that decision. They paid $2.3M in regulatory fines.

**The fix:** DVC + Git guarantees any model is reproducible for life.

---

## 2.2 DVC — Complete Production Workflow

### How DVC Works Internally:

```
Git Repo (tiny pointer files)          S3 Remote (actual large data)
─────────────────────────────          ──────────────────────────────────
data/train.csv.dvc ─────hash──────▶   s3://ml-data/cache/ab/cd123456...
  md5: abcd1234                        (actual 52GB CSV lives here)
  size: 52428800
  path: data/train.csv
```

### Project Setup — Step by Step:

```bash
# 1. Initialize DVC in existing Git repo
cd fraud-detection-ml
git init && dvc init

# 2. Configure S3 remote storage
dvc remote add -d prod-remote s3://fraud-ml-data-store/dvc-cache
dvc remote modify prod-remote region us-east-1
# For GCS: dvc remote add -d prod-remote gs://fraud-ml-data-store/dvc

# 3. Track a large dataset file
dvc add data/raw/transactions_nov2024.csv
# Creates: data/raw/transactions_nov2024.csv.dvc
# Adds:    data/raw/transactions_nov2024.csv  to .gitignore

# 4. Commit the pointer (NOT the data) to Git
git add data/raw/transactions_nov2024.csv.dvc .gitignore
git commit -m "feat: add Nov 2024 transactions dataset (52GB)"

# 5. Push actual data to S3
dvc push
# Uploads 52GB → s3://fraud-ml-data-store/dvc-cache/

# On any new machine / CI server:
git pull            # gets the .dvc pointer file
dvc pull            # downloads exact 52GB dataset from S3
```

### DVC Pipeline (dvc.yaml) — Full Example:

```yaml
stages:
  prepare:
    cmd: python src/prepare.py --input data/raw/transactions_nov2024.csv
    deps:
      - src/prepare.py
      - data/raw/transactions_nov2024.csv
    outs:
      - data/processed/train.parquet
      - data/processed/test.parquet

  featurize:
    cmd: python src/featurize.py
    deps:
      - src/featurize.py
      - data/processed/train.parquet
    params:
      - params.yaml:
        - featurize.window_days
        - featurize.min_txn_count
    outs:
      - data/features/train_features.parquet

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/features/train_features.parquet
    params:
      - params.yaml:
        - train.n_estimators
        - train.max_depth
        - train.learning_rate
    outs:
      - models/fraud_model.pkl
    metrics:
      - metrics/eval_metrics.json:
          cache: false

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - models/fraud_model.pkl
      - data/features/test_features.parquet
    metrics:
      - metrics/eval_metrics.json:
          cache: false
```

```yaml
# params.yaml
featurize:
  window_days: 30
  min_txn_count: 5
train:
  n_estimators: 300
  max_depth: 6
  learning_rate: 0.05
  scale_pos_weight: 99
```

```bash
# Key DVC commands
dvc repro                          # Re-run changed stages only
dvc status                         # What has changed?
dvc metrics diff HEAD~1            # Compare vs previous commit
dvc dag                            # Visualize pipeline DAG
dvc push && dvc pull               # Share data with teammates
```

---

## 2.3 Git Branching for ML Projects

```
main (production-only, requires 2 reviewers + CI pass)
├── staging (validated models awaiting human approval)
│   ├── experiment/xgboost-v3-hyperparams
│   ├── experiment/add-merchant-category-feature
│   └── experiment/90d-window-test
└── data/refresh-nov-2024
```

**Commit convention:**
```
feat: add merchant_category_code feature (+4.3% AUC)
fix: correct rolling window bug (was calendar days, now 30d rolling)
data: refresh training data to Nov 2024 cutoff
model: promote xgboost-v3 to staging (AUC-PR=0.923)
```

---

## 2.4 Handling Large Datasets in Production

**Problem:** 500GB training dataset. CI takes 45 minutes just reading data.

```python
# Solution 1: Stratified sampling for CI/CD
if os.environ.get("CI") == "true":
    df = df.sample(frac=0.01, random_state=42, stratify=df["is_fraud"])

# Solution 2: Parquet partitioning for selective reads
df.to_parquet(
    "s3://fraud-data/features/",
    partition_cols=["year", "month"],
    engine="pyarrow"
)
# Read only last 3 months:
dataset = pq.ParquetDataset(
    "s3://fraud-data/features/",
    filters=[("year", ">=", 2024), ("month", ">=", 9)]
)

# Solution 3: WebDataset streaming (PyTorch, never load full data into RAM)
import webdataset as wds
dataset = (
    wds.WebDataset("s3://fraud/shards/train-{000000..004999}.tar")
    .shuffle(1000).decode().to_tuple("features.npy", "label.cls").batched(256)
)
```

---

## 2.5 LakeFS — Git for Your Entire Data Lake

**Use when:** Entire data lake needs branching (not just individual files).

```bash
# Create experiment branch (isolated from main)
lakectl branch create \
    lakefs://fraud-data/experiment/add-chargeback-data \
    --source lakefs://fraud-data/main

# Upload to branch (main is untouched)
lakectl fs upload -s chargebacks_2024.parquet \
    lakefs://fraud-data/experiment/add-chargeback-data/chargebacks/

# Validate, then merge to main
lakectl merge \
    lakefs://fraud-data/experiment/add-chargeback-data \
    lakefs://fraud-data/main

# Tag snapshot for reproducibility
lakectl tag create lakefs://fraud-data/training-nov-2024-v3 \
    lakefs://fraud-data/main
```

---

## 2.6 Data Contracts — Enforcing Schema Between Teams

```yaml
# data_contract_transactions.yaml
version: "1.3"
owner: "data-engineering-team"
schema:
  transaction_id: {type: string, nullable: false, unique: true}
  user_id:        {type: string, nullable: false}
  amount:         {type: float64, nullable: false, minimum: 0.01, maximum: 1000000}
  currency:       {type: string, nullable: false, pattern: "^[A-Z]{3}$"}
  timestamp:      {type: timestamp, nullable: false, timezone: UTC}
sla:
  freshness: "< 5 minutes"
  completeness: "> 99.5%"
```

```python
# validate_contract.py — fails CI if contract is violated
import pandas as pd, yaml
from great_expectations.dataset import PandasDataset

contract = yaml.safe_load(open("data_contract_transactions.yaml"))
df = pd.read_parquet("data/processed/train.parquet")
dataset = PandasDataset(df)

for col, spec in contract["schema"].items():
    if not spec.get("nullable", True):
        result = dataset.expect_column_values_to_not_be_null(col)
        assert result.success, f"CONTRACT VIOLATION: {col} has nulls!"

print("All data contracts satisfied")
```

---

*Next → [Section 3: Orchestration](../orchestration/README.md)*
