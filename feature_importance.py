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
from pathlib import Path

def get_top_features_shap(model, X, y, max_fraud=5000, random_state: int = 42):
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
    rng = np.random.default_rng(random_state)
    selected_fraud_idx = rng.choice(fraud_idx, size=min(len(fraud_idx), max_fraud), replace=False)

    # equal-sized non-fraud sample
    selected_nonfraud_idx = rng.choice(nonfraud_idx, size=len(selected_fraud_idx), replace=False)

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

def save_shap_case_studies(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    out_dir: str | Path,
    n_cases: int = 40,
    random_state: int = 42,
    prediction_threshold: float = 0.5,
    max_display: int = 15,
    # NEW: per-bucket summaries/plots
    make_bucket_artifacts: bool = True,
    bucket_top_k: int = 15,
) -> pd.DataFrame:
    """Generate SHAP case studies and (optionally) per-bucket SHAP summaries/plots.

    Sampling strategy (thesis-friendly):
      - Balanced coverage across TP/TN/FP/FN buckets.
      - Within each bucket, select a mix of:
          * most confident cases (extremes)
          * most borderline cases (closest to threshold)

    Saves per-case:
      - case_studies_index.csv (metadata for all saved cases)
      - case_XX_summary.csv (feature values + SHAP values, sorted by |SHAP|)
      - case_XX_waterfall.png (best-effort, if plotting succeeds)

    NEW per-bucket artifacts (if make_bucket_artifacts=True):
      - bucket_feature_summary_all.csv (mean_abs_shap + mean_shap per feature per bucket)
      - bucket_<BUCKET>_top_features.csv (top K features for that bucket)
      - bucket_<BUCKET>_mean_abs_shap.png (horizontal bar plot of top K mean(|SHAP|))
      - bucket_<BUCKET>_mean_shap.png (horizontal bar plot of mean(SHAP) for same features)

    Returns:
      - index_df: one row per case study with metadata.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Align y to X to avoid silent misalignment bugs ---
    if isinstance(y, pd.Series):
        y = y.reindex(X.index)
    if pd.isna(y).any():
        raise ValueError("y contains NaNs after aligning to X.index. Check X/y alignment.")

    # --- Predicted probabilities ---
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        raw_scores = model.predict(X)
        y_prob = 1 / (1 + np.exp(-np.asarray(raw_scores)))

    y_prob = np.asarray(y_prob, dtype=float)
    y_true = np.asarray(y, dtype=int)
    y_pred = (y_prob >= prediction_threshold).astype(int)

    # --- Confusion-matrix buckets (indices are positional) ---
    tp_idx = np.where((y_true == 1) & (y_pred == 1))[0]
    tn_idx = np.where((y_true == 0) & (y_pred == 0))[0]
    fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]

    rng = np.random.default_rng(random_state)

    def pick_bucket(idxs: np.ndarray, k: int, prefer_high_prob: bool) -> np.ndarray:
        """Pick k indices from a bucket as half extremes + half borderline."""
        if k <= 0 or len(idxs) == 0:
            return np.array([], dtype=int)

        probs = y_prob[idxs]

        # Extremes: most confident for the model direction.
        order_ext = np.argsort(-probs) if prefer_high_prob else np.argsort(probs)
        sorted_ext = idxs[order_ext]

        k_ext = min(k // 2, len(sorted_ext))
        chosen_ext = sorted_ext[:k_ext]

        remaining = np.setdiff1d(idxs, chosen_ext, assume_unique=False)
        if len(remaining) == 0:
            return chosen_ext

        # Borderline: closest to threshold
        rem_probs = y_prob[remaining]
        order_border = np.argsort(np.abs(rem_probs - prediction_threshold))
        sorted_border = remaining[order_border]

        k_border = min(k - len(chosen_ext), len(sorted_border))
        chosen_border = sorted_border[:k_border]

        chosen = np.concatenate([chosen_ext, chosen_border])

        # Fill randomly if still short
        if len(chosen) < min(k, len(idxs)):
            leftover = np.setdiff1d(idxs, chosen, assume_unique=False)
            need = min(k - len(chosen), len(leftover))
            if need > 0:
                fill = rng.choice(leftover, size=need, replace=False)
                chosen = np.concatenate([chosen, fill])

        return chosen

    # --- Allocate cases across buckets (roughly equal) ---
    base_k = max(1, n_cases // 4)
    remainder = n_cases - 4 * base_k
    add = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for key in ["TP", "TN", "FP", "FN"][:remainder]:
        add[key] += 1

    k_tp = base_k + add["TP"]
    k_tn = base_k + add["TN"]
    k_fp = base_k + add["FP"]
    k_fn = base_k + add["FN"]

    selected_tp = pick_bucket(tp_idx, k_tp, prefer_high_prob=True)
    selected_tn = pick_bucket(tn_idx, k_tn, prefer_high_prob=False)
    selected_fp = pick_bucket(fp_idx, k_fp, prefer_high_prob=True)
    selected_fn = pick_bucket(fn_idx, k_fn, prefer_high_prob=False)

    selected_idx = np.concatenate([selected_tp, selected_tn, selected_fp, selected_fn])
    selected_idx = np.unique(selected_idx)  # safety
    rng.shuffle(selected_idx)

    if len(selected_idx) == 0:
        raise ValueError("No case studies selected. Check labels/predictions/threshold.")

    X_case = X.iloc[selected_idx].reset_index(drop=True)

    # --- SHAP explainer ---
    try:
        explainer = shap.TreeExplainer(model)
    except Exception as e:
        raise RuntimeError(
            "Error initializing SHAP TreeExplainer. Model must be a tree-based model.\n"
            f"Original error: {e}"
        )

    shap_values = explainer.shap_values(X_case)
    if isinstance(shap_values, list):
        # binary classification often returns [class0, class1]
        shap_values = shap_values[1]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, tuple, np.ndarray)):
        base_value = base_value[1] if len(base_value) > 1 else base_value[0]

    # --- Save per-case artifacts + collect rows for bucket aggregation ---
    index_rows = []
    long_rows = []  # for per-bucket aggregation across selected cases

    for i in range(X_case.shape[0]):
        case_id = f"case_{i+1:02d}"
        row_index = int(selected_idx[i])

        y_p = float(y_prob[row_index])
        y_t = int(y_true[row_index])
        y_hat = int(y_pred[row_index])

        if y_t == 1 and y_hat == 1:
            bucket = "TP"
        elif y_t == 0 and y_hat == 0:
            bucket = "TN"
        elif y_t == 0 and y_hat == 1:
            bucket = "FP"
        else:
            bucket = "FN"

        case_shap = np.asarray(shap_values[i], dtype=float)

        case_df = pd.DataFrame(
            {
                "feature": X_case.columns,
                "feature_value": X_case.iloc[i].values,
                "shap_value": case_shap,
                "abs_shap_value": np.abs(case_shap),
            }
        ).sort_values("abs_shap_value", ascending=False)

        case_df.to_csv(out_dir / f"{case_id}_summary.csv", index=False)

        # Best-effort waterfall plot
        try:
            explanation = shap.Explanation(
                values=case_shap,
                base_values=base_value,
                data=X_case.iloc[i],
                feature_names=X_case.columns,
            )
            shap.plots.waterfall(explanation, show=False, max_display=max_display)

            import matplotlib.pyplot as plt

            plt.tight_layout()
            plt.savefig(out_dir / f"{case_id}_waterfall.png", dpi=150, bbox_inches="tight")
            plt.close()
        except Exception:
            pass

        index_rows.append(
            {
                "case_id": case_id,
                "row_index": row_index,
                "true_label": y_t,
                "pred_prob": y_p,
                "pred_label": y_hat,
                "bucket": bucket,
            }
        )

        # collect for bucket summaries (one row per feature per case)
        # (store only what we need for aggregation)
        long_rows.append(
            pd.DataFrame(
                {
                    "case_id": case_id,
                    "bucket": bucket,
                    "feature": X_case.columns,
                    "shap_value": case_shap,
                    "abs_shap_value": np.abs(case_shap),
                }
            )
        )

    index_df = pd.DataFrame(index_rows).sort_values(["bucket", "pred_prob"], ascending=[True, False])
    index_df.to_csv(out_dir / "case_studies_index.csv", index=False)

    # --- NEW: per-bucket aggregation + plots ---
    if make_bucket_artifacts:
        import matplotlib.pyplot as plt

        all_long = pd.concat(long_rows, ignore_index=True)

        bucket_summary = (
            all_long.groupby(["bucket", "feature"])
            .agg(
                mean_abs_shap=("abs_shap_value", "mean"),
                mean_shap=("shap_value", "mean"),
                n_cases=("case_id", pd.Series.nunique),
            )
            .reset_index()
        )

        bucket_summary.to_csv(out_dir / "bucket_feature_summary_all.csv", index=False)

        for bucket in ["TP", "TN", "FP", "FN"]:
            bdf = bucket_summary[bucket_summary["bucket"] == bucket].copy()
            if bdf.empty:
                continue

            top = bdf.sort_values("mean_abs_shap", ascending=False).head(bucket_top_k)
            top.to_csv(out_dir / f"bucket_{bucket}_top_features.csv", index=False)

            # Plot 1: mean(|SHAP|)
            plot_df = top.sort_values("mean_abs_shap", ascending=True)  # barh bottom->top
            plt.figure()
            plt.barh(plot_df["feature"], plot_df["mean_abs_shap"])
            plt.xlabel("Mean(|SHAP|)")
            plt.ylabel("Feature")
            plt.title(f"{bucket}: Top {len(plot_df)} features by Mean(|SHAP|)")
            plt.tight_layout()
            plt.savefig(out_dir / f"bucket_{bucket}_mean_abs_shap.png", dpi=150, bbox_inches="tight")
            plt.close()

            # Plot 2: mean(SHAP) direction for the same top features
            plot_df2 = top.sort_values("mean_shap", ascending=True)
            plt.figure()
            plt.barh(plot_df2["feature"], plot_df2["mean_shap"])
            plt.axvline(0)
            plt.xlabel("Mean(SHAP) (direction)")
            plt.ylabel("Feature")
            plt.title(f"{bucket}: Mean(SHAP) direction for top features")
            plt.tight_layout()
            plt.savefig(out_dir / f"bucket_{bucket}_mean_shap.png", dpi=150, bbox_inches="tight")
            plt.close()

    return index_df
