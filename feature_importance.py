"""
feature_selector.py
--------------------
Feature selection helper for fraud detection and thesis experiments.

This module:
  • Computes feature importance using LightGBM (robust, fast)
  • Optionally refines using SHAP-based importance
  • Removes redundant correlated features
  • Returns reduced dataset + ranked importance table

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


def select_important_features(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: Optional[int] = None,
    importance_threshold: float = 0.95,
    remove_correlated: bool = True,
    correlation_threshold: float = 0.95,
    use_shap: bool = False,
    verbose: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Selects top-N or cumulative-importance-based features using LightGBM (optionally SHAP).

    Parameters
    ----------
    X : pd.DataFrame
        Input features
    y : pd.Series
        Target labels
    top_n : int, optional
        Number of top features to keep. If None, uses cumulative threshold.
    importance_threshold : float
        If top_n is None, keeps features explaining this fraction of total importance.
    remove_correlated : bool
        Whether to drop highly correlated features after importance ranking.
    correlation_threshold : float
        Absolute correlation above which a feature will be dropped.
    use_shap : bool
        If True, refines feature importance using mean absolute SHAP values.
    verbose : bool
        Print progress updates.

    Returns
    -------
    X_reduced : pd.DataFrame
        Reduced dataset with selected features
    feature_ranking : pd.DataFrame
        DataFrame with ranked feature importances and SHAP values (if computed)
    """
    if verbose:
        print("[INFO] Training LightGBM for feature importance estimation...")

    # --- Train baseline LightGBM model ---
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        categorical_feature=X.select_dtypes(include=['category', 'object']).columns.tolist()
    )
    model.fit(X, y)

    # --- Compute importance ---
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importance_df = pd.DataFrame({
        "feature": importances.index,
        "importance_gain": importances.values
    }).sort_values("importance_gain", ascending=False)
    importance_df["importance_norm"] = (
        importance_df["importance_gain"] / importance_df["importance_gain"].sum()
    )
    importance_df["importance_cum"] = importance_df["importance_norm"].cumsum()

    # --- Optional SHAP refinement ---
    if use_shap:
        if verbose:
            print("[INFO] Computing SHAP values for refined importance...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):  # binary classifier
            shap_values = shap_values[1]
        shap_importance = np.abs(shap_values).mean(axis=0)
        importance_df["shap_mean_abs"] = shap_importance
        importance_df = importance_df.sort_values("shap_mean_abs", ascending=False)

    # --- Select top features ---
    if top_n is not None:
        selected_features = importance_df.head(top_n)["feature"].tolist()
        if verbose:
            print(f"[INFO] Keeping top {top_n} features by importance.")
    else:
        selected_features = importance_df.loc[
            importance_df["importance_cum"] <= importance_threshold, "feature"
        ].tolist()
        if verbose:
            print(f"[INFO] Keeping features explaining {importance_threshold*100:.1f}% cumulative importance ({len(selected_features)} features).")

    X_reduced = X[selected_features].copy()

    # --- Correlation pruning ---
    if remove_correlated:
        if verbose:
            print(f"[INFO] Performing correlation pruning (threshold={correlation_threshold})...")
        # Only compute correlation on numeric columns (categorical/object columns can't be correlated)
        numeric_cols = X_reduced.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0:
            corr_matrix = X_reduced[numeric_cols].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if any(upper[col] > correlation_threshold)]
            if verbose and len(to_drop) > 0:
                print(f"[INFO] Dropping {len(to_drop)} correlated features.")
            X_reduced.drop(columns=to_drop, inplace=True, errors="ignore")
            selected_features = [f for f in selected_features if f not in to_drop]
        elif verbose:
            print(f"[INFO] No numeric columns found for correlation pruning.")

    # --- Final feature summary ---
    importance_df["selected"] = importance_df["feature"].isin(selected_features)
    if verbose:
        print(f"[DONE] Selected {len(selected_features)} features out of {X.shape[1]}.")

    return X_reduced, importance_df
