"""
Importable training shell for XGBoost with Optuna + MLflow (no I/O, no preprocessing)
-------------------------------------------------------------------------------
- Accepts preprocessed X, y (pandas DataFrame/Series or NumPy arrays)
- Performs internal stratified train/validation split
- Tunes with Optuna (TPE) on PR-AUC (better for imbalanced fraud data)
- Uses early stopping; handles class imbalance via scale_pos_weight
- Auto-selects GPU (gpu_hist) if available, else CPU (hist)
- Logs to MLflow: parameters, metrics (roc_auc, pr_auc, precision, recall, f1),
  ROC/PR curves, confusion matrix, and the trained model
- Prints progress messages and shows a tqdm progress bar over trials

Usage from a notebook:
    from train_xgb_shell import train_xgb_optuna
    model, metrics, best_params = train_xgb_optuna(
        X, y,
        experiment_name="FraudXGB",
        run_name="xgb_optuna",
        n_trials=30,
        val_size=0.2,
    )

This file intentionally does not read files or preprocess; feed it ready-made features.
"""
from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna
import xgboost as xgb
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost

from sklearn.metrics import (
    average_precision_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


# -------------------------------
# Utilities
# -------------------------------

def _to_numpy_xy(X: Any, y: Any) -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd  # local import to avoid hard dependency if not needed
    if isinstance(X, (pd.DataFrame, pd.Series)):
        X = X.values
    if isinstance(y, (pd.DataFrame, pd.Series)):
        # If a DataFrame was passed, take first column
        y = y.values.ravel()
    y = np.asarray(y).astype(int)
    X = np.asarray(X)
    if y.ndim != 1:
        y = y.ravel()
    return X, y


def _scale_pos_weight(y: np.ndarray) -> float:
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    return float(neg / max(1, pos))


def _gpu_available() -> bool:
    try:
        # Attempt a tiny GPU-boosted fit; if it fails, fall back to CPU
        clf = xgb.XGBClassifier(
            n_estimators=1,
            tree_method="gpu_hist",
            max_depth=1,
            verbosity=0,
        )
        clf.fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
        return True
    except Exception:
        return False


def _metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _plot_artifacts(y_true: np.ndarray, y_prob: np.ndarray, out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    p = out_dir / "roc_curve.png"
    plt.savefig(p, bbox_inches="tight", dpi=150)
    plt.close()
    paths["roc_curve"] = p

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)
    plt.figure()
    plt.plot(rec, prec, label=f"PR curve (AP = {pr_auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    p = out_dir / "pr_curve.png"
    plt.savefig(p, bbox_inches="tight", dpi=150)
    plt.close()
    paths["pr_curve"] = p

    # Confusion matrix @ 0.5
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix @ 0.5")
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ["Non-Fraud", "Fraud"], rotation=45)
    plt.yticks(ticks, ["Non-Fraud", "Fraud"])
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    p = out_dir / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(p, bbox_inches="tight", dpi=150)
    plt.close()
    paths["confusion_matrix"] = p

    return paths


# -------------------------------
# Public API
# -------------------------------

def train_xgb_optuna(
    X: Any,
    y: Any,
    *,
    experiment_name: str = "FraudXGB",
    run_name: str = "xgb_optuna",
    tracking_uri: Optional[str] = None,
    n_trials: int = 30,
    val_size: float = 0.2,
    random_state: int = 42,
    early_stopping_rounds: int = 100,
    max_estimators: int = 4000,
    use_gpu: str = "auto",  # "auto" | True | False
    n_jobs: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[xgb.XGBClassifier, Dict[str, float], Dict[str, Any]]:
    """
    Train an XGBoost classifier with Optuna tuning and MLflow logging.

    Parameters
    ----------
    X, y : array-like or pandas (preprocessed). y must be binary (0/1).
    experiment_name : MLflow experiment name to create/use.
    run_name : MLflow run name.
    tracking_uri : Optional MLflow tracking URI (default: local ./mlruns).
    n_trials : Number of Optuna trials.
    val_size : Fraction for the validation split (stratified).
    random_state : Random seed for splitting and model.
    early_stopping_rounds : XGBoost early stopping rounds.
    max_estimators : Max trees; early stopping will halt earlier.
    use_gpu : "auto" to detect, True to force GPU, False to force CPU.
    n_jobs : Threads for XGBoost (default: os.cpu_count()).
    verbose : Print progress messages.

    Returns
    -------
    model : Trained xgboost.XGBClassifier (fit with early stopping on the internal validation split).
    metrics : dict with roc_auc, pr_auc, precision, recall, f1 on the validation set.
    best_params : dict of the best hyperparameters discovered by Optuna (including computed scale_pos_weight and device params).
    """

    import os

    if n_jobs is None:
        n_jobs = os.cpu_count() or 4

    X_np, y_np = _to_numpy_xy(X, y)

    # Split once; use validation for early stopping and objective
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_np, y_np,
        test_size=val_size,
        stratify=y_np,
        random_state=random_state,
    )

    # Device selection
    if use_gpu == "auto":
        gpu = _gpu_available()
    elif isinstance(use_gpu, bool):
        gpu = use_gpu
    else:
        gpu = False

    tree_method = "gpu_hist" if gpu else "hist"

    # Base (non-tuned) params
    spw = _scale_pos_weight(y_tr)
    base_params: Dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",  # focus on PR during early stopping
        "tree_method": tree_method,
        "random_state": random_state,
        "n_estimators": max_estimators,
        "n_jobs": n_jobs,
        "scale_pos_weight": spw,
    }

    if verbose:
        print(f"[INFO] Using {'GPU' if gpu else 'CPU'} with tree_method='{tree_method}'")
        print(f"[INFO] scale_pos_weight={spw:.3f}, val_size={val_size}, trials={n_trials}")

    # ---------- Optuna objective ----------
    def objective(trial: optuna.Trial) -> float:
        params = dict(base_params)  # copy
        params.update(
            {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-2, 3e-1, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
            }
        )

        model = xgb.XGBClassifier(**params, early_stopping_rounds=early_stopping_rounds)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        # Optimize PR-AUC
        y_prob = model.predict_proba(X_va)[:, 1]
        pr_auc = average_precision_score(y_va, y_prob)
        return float(pr_auc)

    study = optuna.create_study(direction="maximize", study_name="xgb_pr_auc")

    # tqdm progress bar across trials
    with tqdm(total=n_trials, desc="Optuna Trials", unit="trial") as pbar:
        def _tqdm_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            pbar.update(1)
            if verbose:
                print(f"[RUN] Trial {trial.number}: value={trial.value:.5f} | params={{'max_depth': {trial.params.get('max_depth')}, 'eta': {trial.params.get('learning_rate')}}}")
        study.optimize(objective, n_trials=n_trials, callbacks=[_tqdm_cb], show_progress_bar=False)

    best_params = dict(base_params)
    best_params.update(study.best_params)

    # ---------------- MLflow logging & final fit ----------------
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with tempfile.TemporaryDirectory() as td, mlflow.start_run(run_name=run_name):
        # Log study info
        mlflow.log_param("optuna_trials", n_trials)
        mlflow.log_param("use_gpu", gpu)
        for k, v in best_params.items():
            if k != "n_estimators":
                mlflow.log_param(k, v)
        mlflow.log_param("n_estimators", max_estimators)
        mlflow.log_param("early_stopping_rounds", early_stopping_rounds)

        # Fit final model with early stopping on the same validation split
        final_model = xgb.XGBClassifier(**best_params, early_stopping_rounds=early_stopping_rounds)
        final_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

        # Evaluate
        y_prob_va = final_model.predict_proba(X_va)[:, 1]
        metrics = _metrics(y_va, y_prob_va, threshold=0.5)
        mlflow.log_metrics(metrics)

        # Plots & artifacts
        artifacts_dir = Path(td)
        plot_paths = _plot_artifacts(y_va, y_prob_va, artifacts_dir / "plots")
        for name, path in plot_paths.items():
            mlflow.log_artifact(str(path), artifact_path="plots")

        # Feature importances (gain)
        booster = final_model.get_booster()
        gain = booster.get_score(importance_type="gain")
        if gain:
            import pandas as pd
            top = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)[:50]
            fi_df = pd.DataFrame(top, columns=["feature", "gain"])
            fi_path = artifacts_dir / "feature_importance_top50.csv"
            fi_df.to_csv(fi_path, index=False)
            mlflow.log_artifact(str(fi_path), artifact_path="importances")

        # Log model
        mlflow.xgboost.log_model(final_model, artifact_path="model")

    if verbose:
        print("[DONE] Best PR-AUC:", f"{study.best_value:.6f}")
        print("[DONE] Metrics:", {k: round(v, 6) for k, v in metrics.items()})

    return final_model, metrics, best_params
