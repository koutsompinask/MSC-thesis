# IEEE-CIS Fraud Detection — End‑to‑End IPython Notebook Development Guide
**Target dataset:** [Kaggle: IEEE‑CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)  
**Primary objective:** Build a rigorous, thesis‑quality fraud detection workflow in a single `ipynb` using Python.  
**Primary metric (Kaggle):** ROC‑AUC on probabilistic predictions for `isFraud`.  
**Notebook style:** Clear sectioning, reproducible, memory‑aware, time‑respecting evaluation, and explainable results suitable for an academic thesis.

> This document is a **precise build plan** you can feed to a coding agent (e.g., Codex CLI) to generate a production‑quality notebook. It specifies **file I/O**, **function contracts**, **cell order**, **checkpoints**, and **review artifacts**.

---

## 0) Prerequisites

### 0.1 Python & OS
- Python ≥ 3.10
- 16–32 GB RAM recommended (dataset ~1.5–2.5 GB uncompressed). If limited, use **chunked loading** and **dtype down‑casting** below.

### 0.2 Packages to install
```
pip install -U \
  numpy pandas pyarrow polars \
  scikit-learn imbalanced-learn \
  xgboost lightgbm catboost \
  optuna \
  matplotlib plotly kaleido \
  shap \
  tqdm \
  ipywidgets \
  pyjanitor \
  pyyaml \
  fastparquet
```

> If using Kaggle Notebooks, most packages are present. Install missing ones with `pip -q install ...` at the top cell.

### 0.3 Project structure (local or repo)
```
ieee-cis-fraud/
├─ data/                      # put CSVs here (or symlink from Kaggle input)
│  ├─ train_transaction.csv
│  ├─ train_identity.csv
│  ├─ test_transaction.csv
│  ├─ test_identity.csv
│  └─ sample_submission.csv
├─ models/                    # saved models, encoders, and CV artifacts
├─ oof/                       # out-of-fold predictions
├─ figs/                      # plots for thesis
├─ logs/                      # experiment + Optuna logs
└─ ieee_cis_fraud.ipynb       # main notebook to produce
```

### 0.4 Reproducibility policy
- Set global seeds; use `numpy.random.seed`, `random.seed`, and model seeds.
- Record versions with `pip freeze | tee requirements.freeze.txt` (or in a cell).
- Save a `config.yaml` capturing paths, CV strategy, and model choices.


---

## 1) Notebook Header & Configuration

**Cell: Imports & Settings**
```python
# Core
import os, sys, gc, json, math, random, warnings, string, re
from pathlib import Path
import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 200)
warnings.filterwarnings("ignore")

# Viz
import matplotlib.pyplot as plt
import plotly.express as px

# ML
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

# Imbalance
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE

# Optimization
import optuna

# Explainability
import shap

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED); random.seed(RANDOM_SEED)
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
OUT_DIR  = BASE_DIR / "models"; OUT_DIR.mkdir(exist_ok=True)
OOF_DIR  = BASE_DIR / "oof"; OOF_DIR.mkdir(exist_ok=True)
FIG_DIR  = BASE_DIR / "figs"; FIG_DIR.mkdir(exist_ok=True)
LOG_DIR  = BASE_DIR / "logs"; LOG_DIR.mkdir(exist_ok=True)
```

**Cell: Config YAML (editable)**
```python
import yaml

CONFIG = {
  "data_dir": "data",
  "target": "isFraud",
  "id_col": "TransactionID",
  "time_col": "TransactionDT",   # seconds from reference
  "primary_metric": "roc_auc",   # Kaggle uses ROC-AUC
  "cv": {
    "strategy": "time_based_holdout",   # or "stratified_kfold"
    "n_splits": 5,
    "group_by": ["uid"]                 # to avoid leakage across same card/device
  },
  "model": {
    "type": "lightgbm",
    "params": {
      "objective": "binary",
      "metric": "auc",
      "num_leaves": 256,
      "learning_rate": 0.03,
      "feature_fraction": 0.8,
      "bagging_fraction": 0.8,
      "bagging_freq": 1,
      "max_depth": -1,
      "min_data_in_leaf": 50,
      "lambda_l1": 0.0,
      "lambda_l2": 0.0,
      "n_estimators": 10000,
      "verbose": -1
    },
    "early_stopping_rounds": 500
  }
}

with open("config.yaml", "w") as f:
    yaml.safe_dump(CONFIG, f, sort_keys=False)
print(yaml.safe_dump(CONFIG, sort_keys=False))
```

**Checkpoint:** The notebook prints config content and creates folders.


---

## 2) Data Access & Loading

### 2.1 Data placement
- Download from Kaggle and place the five files into `data/` (see structure above).
- (Optional) Use Kaggle API locally: `kaggle competitions download -c ieee-fraud-detection` and unzip.

### 2.2 Memory‑aware readers with dtype map
**Cell: Define dtypes and fast readers**
```python
# Minimal dtype map (extend as needed for memory pressure)
DTYPES_TRANSACTION = {
    "TransactionID": "int32",
    "TransactionDT": "int32",
    "TransactionAmt": "float32",
    "isFraud": "int8",
    # common categoricals (examples):
    "ProductCD": "category",
    "card1": "int32", "card2": "float32", "card3": "float32", "card4": "category", "card5": "float32", "card6": "category",
    "addr1": "float32", "addr2": "float32",
    "dist1": "float32", "dist2": "float32",
}

DTYPES_IDENTITY = {
    "TransactionID": "int32",
    # many are strings/categoricals; keep as 'object' then cast as needed
}

USECOLS_TEST_TARGETLESS = None  # read all; Kaggle test has no 'isFraud'

def read_data(data_dir=DATA_DIR):
    train_tr = pd.read_csv(data_dir / "train_transaction.csv", dtype=DTYPES_TRANSACTION)
    train_id = pd.read_csv(data_dir / "train_identity.csv", dtype=DTYPES_IDENTITY)
    test_tr  = pd.read_csv(data_dir / "test_transaction.csv", dtype={k: v for k, v in DTYPES_TRANSACTION.items() if k != "isFraud"})
    test_id  = pd.read_csv(data_dir / "test_identity.csv", dtype=DTYPES_IDENTITY)
    sample   = pd.read_csv(data_dir / "sample_submission.csv")

    # Merge on TransactionID (not all rows have identity information)
    train = train_tr.merge(train_id, on="TransactionID", how="left")
    test  = test_tr.merge(test_id, on="TransactionID", how="left")
    del train_tr, train_id, test_tr, test_id; gc.collect()
    return train, test, sample

train, test, sample = read_data()
train.shape, test.shape, train["isFraud"].mean()
```

**Checkpoint:** Display shapes; check target prevalence (`isFraud` mean).


---

## 3) Data Hygiene & EDA (time‑respecting)

### 3.1 Sanity checks
- Missingness profile per column.
- Unique counts for key IDs (e.g., `card1..card6`, `addr1/2`, emails, `DeviceInfo`, `ProductCD`).
- Target leakage scan: ensure no future info used in training split.

**Cells:**
```python
def quick_report(df, name):
    print(f"== {name} ==")
    print("shape:", df.shape)
    print("columns:", len(df.columns))
    print(df.dtypes.value_counts())
    print("missing ratio (top 20):")
    miss = df.isna().mean().sort_values(ascending=False).head(20)
    display(miss.to_frame("missing_ratio"))

quick_report(train, "train")
quick_report(test, "test")

# Class imbalance
pos_ratio = train["isFraud"].mean()
print("Fraud rate:", round(float(pos_ratio), 4))
```

### 3.2 Time features and plots
- `TransactionDT` is seconds from a reference. Create: `DT_day`, `DT_week`, `DT_month`, `DT_hour`, `DT_dayofweek`.
- Plot **counts and fraud rate vs. time** to reveal drift.

```python
def add_time_features(df):
    dt = df["TransactionDT"]
    df["DT_day"]   = (dt // (24*60*60)).astype("int32")
    df["DT_week"]  = (dt // (7*24*60*60)).astype("int32")
    df["DT_month"] = (dt // (30*24*60*60)).astype("int32")
    df["DT_hour"]  = (dt // 3600 % 24).astype("int8")
    df["DT_wday"]  = (df["DT_day"] % 7).astype("int8")
    return df

train = add_time_features(train)
test  = add_time_features(test)

# Plot volume and fraud rate by DT_day
tmp = train.groupby("DT_day").agg(n=("isFraud","size"), r=("isFraud","mean")).reset_index()
fig = px.line(tmp, x="DT_day", y=["n","r"], title="Volume (n) and Fraud Rate (r) by DT_day")
fig.write_image(str(FIG_DIR / "time_volume_fraud_rate.png"))
fig
```

**Checkpoint:** Save plots into `figs/` for thesis.


---

## 4) Robust Identifiers & Group Keys (no leakage)

Construct **stable pseudo‑IDs** used in literature and top solutions to reduce sparsity while avoiding test leakage:

```python
def build_keys(df):
    # Card ID combinations (common trick for denoising)
    df["uid"]  = df["card1"].astype(str)
    for col in ["card2","card3","card4","card5","card6"]:
        if col in df.columns:
            df["uid"] = df["uid"] + "_" + df[col].astype(str)
    # Address + distance
    for col in ["addr1","addr2","dist1","dist2"]:
        if col in df.columns:
            df[col] = df[col].fillna(-9999)
    return df

train = build_keys(train)
test  = build_keys(test)
```

> **Note:** Never aggregate using labels; avoid creating keys that directly encode target.


---

## 5) Feature Engineering (FE)

Focus on **behavioral**, **risk**, and **denoised categorical** features. Implement FE in functions to reuse at inference.

```python
CATEGORICAL_BASE = [
  "ProductCD", "card4", "card6",
  "id_12","id_13","id_14","id_15","id_16",
  "id_23","id_27","id_28","id_29","id_30","id_31","id_33",
  "DeviceType","DeviceInfo","P_emaildomain","R_emaildomain"
]

def make_features(df):
    # Amount features
    df["TransactionAmt_log1p"] = np.log1p(df["TransactionAmt"].clip(lower=0))
    # Relative amount to personal baseline
    grp = df.groupby("uid")["TransactionAmt"]
    df["Amt_to_uid_mean"] = df["TransactionAmt"] / (grp.transform("mean") + 1e-3)
    df["Amt_to_uid_std"]  = (df["TransactionAmt"] - grp.transform("mean")) / (grp.transform("std") + 1e-3)

    # Frequency encodings (safe, use OOF on train)
    for col in ["ProductCD","card4","card6","P_emaildomain","R_emaildomain","DeviceInfo"]:
        if col in df.columns:
            vc = df[col].value_counts(dropna=False)
            df[f"{col}_freq"] = df[col].map(vc).astype("float32")

    # Time × Amount interactions
    df["hour_x_amt"] = df["DT_hour"] * df["TransactionAmt_log1p"]

    # Email domain simplification
    for col in ["P_emaildomain","R_emaildomain"]:
        if col in df.columns:
            df[col+"_suffix"] = df[col].str.extract(r"\.([a-z]+)$", expand=False).fillna("NA").astype("category")

    return df

train = make_features(train)
test  = make_features(test)
```

### 5.1 Out‑of‑Fold (OOF) target‑aware encodings (optional, **leak‑safe**)
If you use target encoding (TE), do it **OOF** on train, then fit on full train to transform test.

```python
def oof_target_encode(train, test, col, target="isFraud", n_splits=5, noise=1e-3):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    tr_enc = pd.Series(index=train.index, dtype="float32")
    for tr_idx, va_idx in skf.split(train, train[target]):
        tr_mean = train.iloc[tr_idx].groupby(col)[target].mean()
        tr_enc.iloc[va_idx] = train.iloc[va_idx][col].map(tr_mean).astype("float32")
    # Fit on full train for test transform
    full_mean = train.groupby(col)[target].mean()
    te_test = test[col].map(full_mean).astype("float32")
    # Regularization
    tr_enc = tr_enc.fillna(train[target].mean()) + noise * np.random.randn(len(tr_enc))
    te_test = te_test.fillna(train[target].mean())
    return tr_enc, te_test

for col in ["ProductCD","card4","card6"]:
    te_tr, te_te = oof_target_encode(train, test, col)
    train[f"{col}_te"] = te_tr; test[f"{col}_te"] = te_te
```

**Checkpoint:** Persist FE column list for reproducibility.


---

## 6) Train/Validation Protocol (Time‑based)

To reflect real‑world deployment and the competition’s temporal nature, use a **chronological split**.

```python
def time_based_split(df, time_col="DT_day", valid_days=20):
    cut = df[time_col].quantile(1.0 - valid_days / df[time_col].nunique())
    trn = df[df[time_col] <= cut].copy()
    val = df[df[time_col] >  cut].copy()
    return trn, val, cut

trn, val, cut = time_based_split(train, time_col="DT_day", valid_days=20)
print(trn.shape, val.shape, "cut@DT_day:", cut)
```

> Alternative: **Grouped K‑Fold** by `uid` with time stratification to minimize leakage across the same card/device.


---

## 7) Baselines

### 7.1 Logistic Regression (strong sanity check)
```python
FEATURES_BASE = [c for c in train.columns if c not in ["isFraud","TransactionID"]]
X_tr, y_tr = trn[FEATURES_BASE], trn["isFraud"]
X_va, y_va = val[FEATURES_BASE], val["isFraud"]

num_cols = X_tr.select_dtypes(include=["int16","int32","int64","float16","float32","float64"]).columns.tolist()
cat_cols = X_tr.select_dtypes(include=["object","category"]).columns.tolist()

preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler(with_mean=False))]), num_cols),
    ("cat",  SimpleImputer(strategy="most_frequent"), cat_cols),
], remainder="drop")

logit = Pipeline([("prep", preprocess),
                  ("clf", LogisticRegression(max_iter=200, n_jobs=-1, class_weight="balanced"))])

logit.fit(X_tr, y_tr)
val_pred = logit.predict_proba(X_va)[:,1]
print("ROC-AUC:", roc_auc_score(y_va, val_pred))
print("PR-AUC :", average_precision_score(y_va, val_pred))
```

### 7.2 Gradient Boosting (LightGBM – tabular SOTA)
```python
LGB_PARAMS = CONFIG["model"]["params"].copy()

lgb_train = lgb.Dataset(trn[FEATURES_BASE], label=y_tr, free_raw_data=False)
lgb_valid = lgb.Dataset(val[FEATURES_BASE], label=y_va, reference=lgb_train, free_raw_data=False)

model_lgb = lgb.train(
    LGB_PARAMS, lgb_train,
    num_boost_round=LGB_PARAMS.pop("n_estimators", 10000),
    valid_sets=[lgb_train, lgb_valid],
    valid_names=["train","valid"],
    early_stopping_rounds=CONFIG["model"]["early_stopping_rounds"],
    verbose_eval=200
)
va_pred = model_lgb.predict(val[FEATURES_BASE], num_iteration=model_lgb.best_iteration)
print("LGB ROC-AUC:", roc_auc_score(y_va, va_pred))
print("LGB PR-AUC :", average_precision_score(y_va, va_pred))
```

**Checkpoint:** Save model and OOF predictions.
```python
import joblib
joblib.dump(model_lgb, OUT_DIR / "lgb_baseline.pkl")
oof = pd.DataFrame({"TransactionID": val["TransactionID"], "isFraud": y_va, "pred": va_pred})
oof.to_parquet(OOF_DIR / "oof_lgb_baseline.parquet", index=False)
```


---

## 8) Hyperparameter Optimization (Optuna)

Optimize **ROC‑AUC** with time‑based validation. Limit search for speed, then expand.

```python
def objective(trial):
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 64, 1024, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 300),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
        "is_unbalance": True
    }
    dtrain = lgb.Dataset(trn[FEATURES_BASE], label=y_tr)
    dvalid = lgb.Dataset(val[FEATURES_BASE], label=y_va, reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=10000,
                      valid_sets=[dtrain, dvalid],
                      early_stopping_rounds=300, verbose_eval=False)
    preds = model.predict(val[FEATURES_BASE], num_iteration=model.best_iteration)
    return roc_auc_score(y_va, preds)

study = optuna.create_study(direction="maximize", study_name="lgb_opt")
study.optimize(objective, n_trials=40)
best_params = study.best_params; best_params.update({"objective":"binary","metric":"auc","verbosity":-1})
json.dump(best_params, open(OUT_DIR / "lgb_best_params.json","w"), indent=2)
best_params
```

**Checkpoint:** Save `logs/study.db` or `study.pkl` for thesis reproducibility.


---

## 9) Advanced Modeling (Optional but Recommended)

- **CatBoost** for high‑cardinality categoricals without heavy encoding.
- **XGBoost** with `scale_pos_weight` tuned to imbalance.
- **Stacking/Ensembling**: average calibrated predictions from LGBM + CatBoost + XGB across seeds.

```python
# Example: Calibrated stacking (simple average)
preds = np.column_stack([
    va_pred,                          # LGBM
    # add other model validation preds here
])
val_blend = preds.mean(axis=1)
print("Blend ROC-AUC:", roc_auc_score(y_va, val_blend))
```


---

## 10) Thresholding, Budgeting, and Business Metrics

Even though Kaggle uses ROC‑AUC, a thesis should report:
- **PR‑AUC** and **precision@k** (e.g., top 0.5% alerts).
- **Cost curves**: expected profit with review capacity constraint.

```python
def precision_at_k(y_true, y_score, k=0.005):
    n = max(1, int(len(y_true)*k))
    idx = np.argsort(-y_score)[:n]
    return (y_true.iloc[idx].sum() / n)

print("Precision@0.5%:", precision_at_k(y_va.reset_index(drop=True), pd.Series(val_blend).reset_index(drop=True), k=0.005))
```

Save plots (PR curve, threshold vs. precision/recall) to `figs/`.


---

## 11) Explainability (Compliance‑ready)

Use **SHAP** on tree models:
```python
explainer = shap.TreeExplainer(model_lgb)
shap_values = explainer.shap_values(val[FEATURES_BASE], check_additivity=False)
shap.summary_plot(shap_values, val[FEATURES_BASE], show=False)
plt.tight_layout(); plt.savefig(FIG_DIR / "shap_summary.png", dpi=200)
```

Also include:
- **Permutation importance** for robustness.
- **Slice analysis**: compare performance by `DT_month`, `ProductCD`, and `uid` frequency buckets.


---

## 12) Full‑Train & Inference on Test

**Cell: Retrain with best params on full train (chronological policy)**  
- Option A (competition style): train on **all training data** and predict test.  
- Option B (realistic): train until `cut` and reserve last window for a **pseudo‑production** check.

```python
best = json.load(open(OUT_DIR / "lgb_best_params.json"))
dtrain_full = lgb.Dataset(train[FEATURES_BASE], label=train["isFraud"])
final = lgb.train(best, dtrain_full, num_boost_round=12000)
test_pred = final.predict(test[FEATURES_BASE])
sub = pd.DataFrame({"TransactionID": test["TransactionID"], "isFraud": test_pred})
sub.to_csv("submission.csv", index=False)
sub.head()
```

**Checkpoint:** Write `submission.csv`.


---

## 13) Robustness & Drift Studies (Thesis‑critical)

- **Temporal holdouts**: rolling origin evaluation (walk‑forward).  
- **Delay simulation**: hide last `k` days labels during training to mimic investigation lag.  
- **Monitoring**: stability of feature distributions (PSI) and model scores across time.

```python
def population_stability_index(expected, actual, bins=20):
    qs = np.linspace(0, 1, bins+1)
    e, a = np.quantile(expected, qs), np.quantile(actual, qs)
    # compute PSI on binned frequencies (left as exercise); save figure to figs/
```

Report in thesis: drift plots, PSI table, AUC per window.


---

## 14) Documentation & Artifacts to Save

- `config.yaml`, notebook, `requirements.freeze.txt`  
- Best params JSON, trained models (`.pkl`), OOF predictions (`.parquet`), figures (`.png`), submission (`.csv`)  
- A short `README.md` explaining how to reproduce all results.


---

## 15) Quality Checklist (use before submission / thesis freeze)

- [ ] All random seeds set; environment versions recorded.  
- [ ] No feature leakage (time‑based split validated).  
- [ ] Baselines (LogReg, LGBM) + at least one alternative model compared.  
- [ ] Metrics: ROC‑AUC, PR‑AUC, precision@k; error analysis included.  
- [ ] Explainability: SHAP summary and top features documented.  
- [ ] Drift study: AUC over time & PSI/feature shift plots.  
- [ ] Re‑run from clean kernel produces identical results (within stochastic tolerance).  


---

## Appendix A — Suggested Feature Catalog (dataset‑specific hints)

- **Amounts:** `TransactionAmt`, `log1p(TransactionAmt)`, z‑scores per `uid`, per `ProductCD`.  
- **Time:** `DT_day`, `DT_hour`, `DT_wday`; rolling counts per `uid` (transactions in last 1/6/24 hours).  
- **Cards & Addresses:** frequency encodings, co‑occurrence counts (`uid` × `P_emaildomain`, `addr1`).  
- **Identity:** parsed `DeviceInfo` (vendor, version), `id_30` (OS), `id_31` (browser) → split to base/version.  
- **Distances:** `dist1`, `dist2` missingness flags and quantized buckets.  
- **Emails:** domain groupings (e.g., free vs. corporate), suffix (`.com`, `.edu`, etc.).  
- **High‑dim features V1–V339, C1–C14, D1–D15, M1–M9:** treat as numeric; explore missing patterns; add missing flags.  
- **Risk ratios:** profile deviation per `uid` (amount, hour) and per `ProductCD`.  

> Always compute **train statistics only** (OOF or pre‑cut) when normalizing/encoding to prevent leakage.


---

## Appendix B — Minimal Submission Contract

- CSV with columns: `TransactionID`, `isFraud` (probability).  
- No index column; header row required.  
- Use `float32` to reduce size.  

```python
sub["isFraud"] = sub["isFraud"].astype("float32")
sub.to_csv("submission.csv", index=False)
```


---

## References & Pointers

- Kaggle IEEE‑CIS Fraud Detection — competition & data description (files `train_transaction`, `train_identity`, `test_*`, `sample_submission`; **join on** `TransactionID`; **official metric**: ROC‑AUC).  
- Classic baseline approaches and community insights emphasize **time‑based validation**, **frequency/target encodings (OOF)**, **boosted trees**, and **memory‑aware loading**.