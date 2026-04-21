"""
Run this script once to generate fastapi/examples.json with 3 pre-selected test transactions.
Requires: X_test, y_test, and the LightGBM model loaded in memory (run training.ipynb first),
OR run this standalone if the processed test CSV exists.

Usage (standalone):
  cd /home/kkout/Workspaces/MSC-thesis
  source .venv/bin/activate
  python fastapi/generate_examples.py
"""
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from model_input import json_safe_value, prepare_lightgbm_input

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "mlruns/161229706116606837/models/m-85ab0e322208453d847c06d654120b9e/artifacts/model.pkl"
OUTPUT = Path(__file__).parent / "examples.json"

# API field names (from model.py)
API_FIELDS = [
    "D8","uid_M3_ct","uid_V_PCA_5_rel","uid_day_of_week_TransactionAmt_std","uid_C14_mean",
    "uid_V_PCA_11_mean","uid_C11_rel","D10","uid_V_PCA_13_std","uid_DeviceType_freq","card6",
    "D15","uid_C13_mean","uid_V_PCA_5_std","uid_V_PCA_7_rel","uid_C13_std",
    "ProductCD_TransactionAmt_rel","id_19","uid_V_PCA_10_mean","card1","C1","uid_V_PCA_9_rel",
    "uid_V_PCA_4_mean","id_18","uid_C12_rel","uid_id_30_ct","id_30","uid_V_PCA_3_std",
    "P_emaildomain","D4","uid_id_31_ct","uid_DeviceInfo_ct","addr2","D14","uid_V_PCA_1_mean",
    "V_PCA_13","C9","uid_TransactionAmt_rel","day_of_week","uid_M4_ct","C5","uid_V_PCA_13_mean",
    "uid_V_PCA_7_mean","id_21","uid_C5_std","uid_V_PCA_5_mean","uid_V_PCA_3_rel","M1",
    "uid_C7_rel","D13","uid_TransactionAmt_mean","M4","uid_C9_std","uid_C14_rel","id_38",
    "V_PCA_10","ProductCD_TransactionAmt_mean","uid_V_PCA_12_rel","V_PCA_2","M6","D7",
    "uid_V_PCA_8_mean","uid_C3_mean","uid_M9_ct","DecimalPlaces","uid_V_PCA_8_std","V_PCA_12",
    "uid_V_PCA_2_rel","uid_V_PCA_6_mean","card4","id_05","id_15","D9","uid_TransactionAmt_std",
    "id_02","R_emaildomain","addr1","V_PCA_8","DeviceType","uid_C7_std","id_07",
]

DEMO_ANCHORS = {
    "clear_legit": ("Low probability", 0.0),
    "borderline": ("Mid probability", 0.5),
    "clear_fraud": ("High probability", 1.0),
}

print("Loading model...")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

def csv_columns(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()

def needed_columns(path: Path) -> list[str]:
    columns = set(csv_columns(path))
    wanted = set(model.feature_name_) | set(API_FIELDS)
    return [column for column in csv_columns(path) if column in wanted and column in columns]

def count_csv_rows(path: Path) -> int:
    with open(path) as f:
        return sum(1 for _ in f) - 1

# Try to load processed test data
test_csv = ROOT / "ieee-fraud-detection-data/processed/test_processed.csv"
if not test_csv.exists():
    # Fall back to training CSV and use the last 19% as test proxy
    train_csv = ROOT / "ieee-fraud-detection-data/processed/train_processed.csv"
    if not train_csv.exists():
        print(f"ERROR: Could not find processed data at {test_csv} or {train_csv}")
        print("Run EDA_and_preprocessing.ipynb first, then re-run this script.")
        exit(1)
    print(f"Loading from {train_csv}...")
    n = count_csv_rows(train_csv)
    start = int(n * 0.81)
    df = pd.read_csv(train_csv, usecols=needed_columns(train_csv), skiprows=range(1, start + 1))
else:
    print(f"Loading from {test_csv}...")
    df = pd.read_csv(test_csv, usecols=needed_columns(test_csv))

print(f"Test set: {len(df)} rows")

available_fields = [f for f in API_FIELDS if f in df.columns]
X = df[available_fields].copy()

print(f"Running inference on {len(X)} rows ({len(available_fields)}/{len(API_FIELDS)} API fields available)...")
X_full = prepare_lightgbm_input(model, X)

probs = model.predict_proba(X_full)[:, 1]

def row_to_dict(idx: int) -> dict:
    row = X.iloc[idx]
    return {k: json_safe_value(v) for k, v in row.items()}

cases: dict = {}
used_indices: set[int] = set()

def keep_case(name: str, idx: int, label: str) -> None:
    cases[name] = row_to_dict(idx)
    used_indices.add(idx)
    print(f"{label}: idx={idx}, prob={probs[idx]:.4f}")

def closest_to_anchor(anchor: float, exclude: set[int]) -> int:
    order = np.argsort(np.abs(probs - anchor))
    for candidate in order:
        idx = int(candidate)
        if idx not in exclude:
            return idx
    return int(order[0])

for name, (label, anchor) in DEMO_ANCHORS.items():
    idx = closest_to_anchor(anchor, used_indices)
    keep_case(name, idx, label)

with open(OUTPUT, "w") as f:
    json.dump(cases, f, indent=2)

print(f"\n✅ examples.json written to {OUTPUT}")
