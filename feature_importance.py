"""
feature_selector.py
--------------------
Feature selection helper for fraud detection and thesis experiments.

Author: konstantinos koutsompinas
Date: 2025
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from typing import Optional
import pandas as pd
import numpy as np

import shap
import numpy as np
import pandas as pd

def get_top_features_shap(model, X, y, max_fraud=5000):
    """
    Computes SHAP feature importance for a trained tree-based model,
    using all (or up to max_fraud) fraud cases and an equal-sized 
    sample of non-fraud cases.

    Parameters
    ----------
    model : trained XGBoost / LightGBM / CatBoost model
    X : pandas DataFrame 
        Input data for SHAP evaluation
    y : pandas Series or array-like
        Ground-truth labels (0 = non-fraud, 1 = fraud)
    max_fraud : int
        Maximum number of fraud examples to include

    Returns
    -------
    DataFrame with columns: feature, importance (sorted)
    """

    # ---------- 1. Build SHAP sample ----------
    fraud_idx = np.where(y == 1)[0]
    nonfraud_idx = np.where(y == 0)[0]

    # limit frauds if too many
    selected_fraud_idx = (
        np.random.choice(fraud_idx, size=min(len(fraud_idx), max_fraud), replace=False)
    )

    # equal-sized non-fraud sample
    selected_nonfraud_idx = (
        np.random.choice(nonfraud_idx, size=len(selected_fraud_idx), replace=False)
    )

    # final sample
    sample_idx = np.concatenate([selected_fraud_idx, selected_nonfraud_idx])
    X_shap = X.iloc[sample_idx]

    # ---------- 2. Build SHAP explainer ----------
    try:
        explainer = shap.TreeExplainer(model)
    except Exception as e:
        raise RuntimeError(
            "Error initializing SHAP TreeExplainer. "
            "Model must be a tree-based model.\n"
            f"Original error: {e}"
        )

    # ---------- 3. Compute SHAP values ----------
    try:
        shap_values = explainer.shap_values(X_shap)
    except Exception as e:
        raise RuntimeError(
            "Error computing SHAP values.\n"
            f"Original error: {e}"
        )

    # multi-class output (list) → use class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.abs(shap_values)

    # ---------- 4. Compute global importance ----------
    importance = shap_values.mean(axis=0)

    df = pd.DataFrame({
        "feature": X.columns,
        "importance": importance
    }).sort_values("importance", ascending=False)

    return df