from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from fastapi import Security, HTTPException
from pathlib import Path
from typing import Any
import json
import pickle
import numpy as np
import pandas as pd
import shap
import mlflow.pyfunc

from config import API_KEY, API_KEY_NAME
from model_input import json_safe_value, prepare_lightgbm_input

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def authenticate(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Invalid API Key")

# ------------------------------------
# Model loading
# ------------------------------------
ARTIFACTS_DIR = Path(__file__).parent.parent / "mlruns/161229706116606837/models/m-85ab0e322208453d847c06d654120b9e/artifacts"

print("Loading MLflow pyfunc model...")
ml_model = mlflow.pyfunc.load_model(str(ARTIFACTS_DIR))
print("Loading native LightGBM model for predict_proba + SHAP...")
with open(ARTIFACTS_DIR / "model.pkl", "rb") as f:
    lgbm_native = pickle.load(f)

FEATURE_COLUMNS: list[str] = lgbm_native.feature_name_

print("Initializing SHAP TreeExplainer (cached)...")
shap_explainer = shap.TreeExplainer(lgbm_native)
print("All models ready!")

# Metrics from the best run
MODEL_ROC_AUC = 0.9191
MODEL_PR_AUC  = 0.5737

EXAMPLES_PATH = Path(__file__).parent / "examples.json"

# ------------------------------------
# FastAPI app
# ------------------------------------
app = FastAPI(
    title="Fraud Detection API",
    description="Live inference with LightGBM — fraud probability + SHAP explainability.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------
# Endpoints
# ------------------------------------

@app.post("/predict")
def predict(data: dict[str, Any], api_key: str = Depends(authenticate)):
    full_df = prepare_lightgbm_input(lgbm_native, pd.DataFrame([data]))
    prediction = ml_model.predict(full_df)
    return {"prediction": int(prediction[0])}


@app.post("/predict_explain")
def predict_explain(data: dict[str, Any], api_key: str = Depends(authenticate)):
    full_df = prepare_lightgbm_input(lgbm_native, pd.DataFrame([data]))

    # Fraud probability
    prob = float(lgbm_native.predict_proba(full_df)[:, 1][0])

    # SHAP values (log-odds space)
    shap_vals = shap_explainer.shap_values(full_df)
    # For binary LGBMClassifier, shap_values returns array shape (n_samples, n_features)
    if isinstance(shap_vals, list):
        sv = shap_vals[1][0]  # class 1 (fraud)
    else:
        sv = shap_vals[0]

    base_value = float(shap_explainer.expected_value)
    if isinstance(shap_explainer.expected_value, (list, np.ndarray)):
        base_value = float(shap_explainer.expected_value[1])

    # Build sorted top-15 feature contributions
    feature_values = full_df.values[0]
    pairs = sorted(
        zip(FEATURE_COLUMNS, feature_values, sv),
        key=lambda x: abs(x[2]),
        reverse=True
    )[:15]

    shap_out = [
        {
            "feature": feat,
            "value": json_safe_value(val),
            "shap": float(s),
            "direction": "positive" if s > 0 else "negative",
        }
        for feat, val, s in pairs
    ]

    return {
        "probability": prob,
        "shap_values": shap_out,
        "shap_base_value": base_value,
        "model_name": "LightGBM",
        "config": "Reduced Features (81 API fields / 215 trained)",
        "roc_auc": MODEL_ROC_AUC,
        "pr_auc": MODEL_PR_AUC,
    }


@app.get("/examples")
def get_examples():
    if not EXAMPLES_PATH.exists():
        raise HTTPException(status_code=404, detail="examples.json not found — run the extraction script first")
    with open(EXAMPLES_PATH) as f:
        data = json.load(f)
    return data
