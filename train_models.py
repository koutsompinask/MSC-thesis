"""
train_models.py
----------------
Reusable training & evaluation shell for thesis experiments.

Supports:
- XGBoost, CatBoost, LightGBM (Optuna-tuned)
- PR-AUC optimization for imbalanced datasets
- Unified evaluation (ROC, PR, F1, SHAP, calibration)
- MLflow experiment tracking with model tagging

Author: Kostas Koutsompinas
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import optuna
import mlflow as mlf
import mlflow.xgboost as mlf_xgboost, mlflow.catboost as mlf_catboost, mlflow.lightgbm as mlf_lightgbm, mlflow.sklearn as mlf_sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
    precision_recall_curve,
    confusion_matrix,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
from tqdm.auto import tqdm
from typing import Any, Dict, Optional, Tuple
import tempfile

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

# ---------------------------
#  UNIVERSAL EVALUATION LOGIC
# ---------------------------
def evaluate_and_log(
    model,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    experiment_name: str,
    run_name: str,
    model_type: str | None = None,
    best_params: dict | None = None,
    tracking_uri: str = "mlruns"
):
    """
    Evaluate a trained binary classifier and log results to MLflow.
    Includes metrics, SHAP explainability, calibration, and confusion matrix.

    Works with: XGBoost, CatBoost, LightGBM, or any sklearn-style model.
    """
    # --- Detect model type automatically if not provided ---
    if model_type is None:
        cname = model.__class__.__name__.lower()
        if "xgb" in cname:
            model_type = "xgboost"
        elif "cat" in cname:
            model_type = "catboost"
        elif "lgb" in cname:
            model_type = "lightgbm"
        else:
            model_type = "sklearn"

    mlf.set_tracking_uri(tracking_uri)
    mlf.set_experiment(experiment_name)

    with tempfile.TemporaryDirectory() as td, mlf.start_run(run_name=run_name):
        # Tag model type and Optuna params
        mlf.set_tag("model_type", model_type)
        if best_params:
            for k, v in best_params.items():
                mlf.log_param(k, v)

        # --- Predictions ---
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_va)[:, 1]
        else:
            y_prob = model.predict(X_va)
        y_pred = (y_prob >= 0.5).astype(int)

        # --- Metrics ---
        roc_auc = roc_auc_score(y_va, y_prob)
        pr_auc = average_precision_score(y_va, y_prob)
        precision, recall, f1, _ = precision_recall_fscore_support(y_va, y_pred, average="binary")
        metrics = {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        mlf.log_metrics(metrics)
        print(f"[INFO] Logged metrics: {metrics}")
        artifacts_dir = Path(td)
        plot_paths = _plot_artifacts(y_va, y_prob, artifacts_dir / "plots")
        for name, path in plot_paths.items():
            mlflow.log_artifact(str(path), artifact_path="plots")

        # --- SHAP Explainability ---
        try:
            fraud_idx = np.where(y_va == 1)[0]
            nonfraud_idx = np.where(y_va == 0)[0]
            n = min(len(fraud_idx), len(nonfraud_idx), 500)
            sel_idx = np.concatenate([
                np.random.choice(fraud_idx, size=n, replace=False),
                np.random.choice(nonfraud_idx, size=n, replace=False)
            ])
            np.random.shuffle(sel_idx)
            X_shap = X_va.iloc[sel_idx].reset_index(drop=True)
            y_shap = y_va.iloc[sel_idx].reset_index(drop=True)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap)

            plt.figure()
            shap.summary_plot(shap_values, X_shap, show=False)
            mlf.log_figure(plt.gcf(), "explainability/shap_summary.png")
            plt.close()
            print("[INFO] Logged SHAP summary plot.")
        except Exception as e:
            print(f"[WARN] SHAP skipped: {e}")

        # --- Calibration ---
        prob_true, prob_pred = calibration_curve(y_va, y_prob, n_bins=10)
        # Align weights to non-empty bins returned by calibration_curve
        bin_edges = np.linspace(0.0, 1.0, 11)
        counts, _ = np.histogram(y_prob, bins=bin_edges)
        non_empty_mask = counts > 0
        weights = counts[non_empty_mask]
        ece = np.average(np.abs(prob_true - prob_pred), weights=weights)
        brier = brier_score_loss(y_va, y_prob)
        mlf.log_metric("ece", float(ece))
        mlf.log_metric("brier_score", float(brier))

        plt.figure()
        plt.plot(prob_pred, prob_true, "o-", label="Model")
        plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
        plt.legend()
        plt.title(f"Calibration (ECE={ece:.4f}, Brier={brier:.4f})")
        mlf.log_figure(plt.gcf(), "calibration/reliability.png")
        plt.close()

        # --- Log Model ---
        input_example = X_va.iloc[:5]
        try:
            # Prefer generic API when available
            mlf.log_model(model, name="model", input_example=input_example)
        except Exception:
            # Fallback to flavor-specific APIs
            try:
                if model_type == "xgboost":
                    mlf_xgboost.log_model(model, name="model", input_example=input_example)
                elif model_type == "catboost":
                    mlf_catboost.log_model(model, name="model", input_example=input_example)
                elif model_type == "lightgbm":
                    mlf_lightgbm.log_model(model, name="model", input_example=input_example)
                else:
                    mlf_sklearn.log_model(model, name="model", input_example=input_example)
            except Exception as e:
                print(f"[WARN] Could not log model: {e}")

        print("[INFO] Evaluation complete and logged.")
    return metrics


# ---------------------------
#  MODEL-SPECIFIC TRAINING
# ---------------------------
def train_xgb_optuna(X, y, val_size=0.2, n_trials=30, random_state=42,
                     early_stopping_rounds=50, use_gpu=True):
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=val_size, stratify=y, random_state=random_state)

    def objective(trial):
        params = {
            "n_estimators": 500,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "tree_method": "gpu_hist" if  _gpu_available() and use_gpu else "hist",
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "n_jobs": -1
        }
        model = xgb.XGBClassifier(**params,  early_stopping_rounds=early_stopping_rounds)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        y_pred = model.predict_proba(X_va)[:, 1]
        score = average_precision_score(y_va, y_pred)
        pbar.set_postfix_str(f"Trial {trial.number+1}/{n_trials} PR-AUC={score:.4f}")
        pbar.update(1)
        return score

    study = optuna.create_study(direction="maximize", study_name="xgboost_pr_auc_optimization")
    with tqdm(total=n_trials, desc="[Optuna XGBoost Tuning]", unit="trial") as pbar:
        def _tqdm_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            pbar.update(1)
            print(f"[RUN] Trial {trial.number}: value={trial.value:.5f}")
        study.optimize(objective, n_trials=n_trials, callbacks=[_tqdm_cb], show_progress_bar=False, gc_after_trial=True)
    best_params = study.best_params
    print(f"[INFO] Best XGBoost params: {best_params}")

    best_model = xgb.XGBClassifier(**best_params,  early_stopping_rounds=early_stopping_rounds)
    best_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return best_model, best_params, X_va, y_va


def train_catboost_optuna(X, y, val_size=0.2, n_trials=30, random_state=42,
                          early_stopping_rounds=50, use_gpu=True):
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=val_size, stratify=y, random_state=random_state)
    cat_features = [i for i, c in enumerate(X_tr.columns) if str(X_tr[c].dtype) in ["object", "category"]]

    def objective(trial):
        params = {
            "iterations": 500,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_seed": random_state,
            "eval_metric": "PRAUC",
            "task_type": "GPU" if  _gpu_available() and use_gpu else "CPU",
            "verbose": False
        }
        model = CatBoostClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va),
                  cat_features=cat_features,
                  early_stopping_rounds=early_stopping_rounds,
                  verbose=False)
        y_pred = model.predict_proba(X_va)[:, 1]
        score = average_precision_score(y_va, y_pred)
        return score

    study = optuna.create_study(direction="maximize", study_name="catboost_pr_auc_optimization")
    with tqdm(total=n_trials, desc="[Optuna CatBoost Tuning]", unit="trial") as pbar:
        def _tqdm_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            pbar.update(1)
            print(f"[RUN] Trial {trial.number}: value={trial.value:.5f}")
        study.optimize(objective, n_trials=n_trials, callbacks=[_tqdm_cb], show_progress_bar=False, gc_after_trial=True)
    best_params = study.best_params
    print(f"[INFO] Best CatBoost params: {best_params}")

    best_model = CatBoostClassifier(
        **best_params, iterations=1000,
        random_seed=random_state, task_type="GPU" if  _gpu_available() and use_gpu else "CPU", verbose=False
    )
    best_model.fit(X_tr, y_tr, eval_set=(X_va, y_va),
                   cat_features=cat_features,
                   early_stopping_rounds=early_stopping_rounds,
                   verbose=False)
    return best_model, best_params, X_va, y_va

def train_lgbm_optuna(X, y, val_size=0.2, n_trials=30, random_state=42,
                      early_stopping_rounds=50, use_gpu=True):
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=val_size, stratify=y, random_state=random_state)

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "average_precision",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "device": "gpu" if  _gpu_available() and use_gpu else "cpu",
        }
        model = lgb.LGBMClassifier(**params, n_estimators=500)
        try:
            model.fit(X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric="average_precision",
                    callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
            y_pred = model.predict_proba(X_va)[:, 1]
            score = average_precision_score(y_va, y_pred)
            pbar.set_postfix_str(f"Trial {trial.number+1}/{n_trials} PR-AUC={score:.4f}")
            pbar.update(1)
            return score
        except Exception as e:
            score = 0.0
            pbar.set_postfix_str(f"Trial {trial.number+1}/{n_trials} PR-AUC={score:.4f}")
            pbar.update(1)
            print(f"[WARN] Trial {trial.number} failed with error: {e}")
            return score

    study = optuna.create_study(direction="maximize", study_name="lgbm_pr_auc_optimization")
    with tqdm(total=n_trials, desc="[Optuna LightGBM Tuning]", unit="trial") as pbar:
        def _tqdm_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            pbar.update(1)
            print(f"[RUN] Trial {trial.number}: value={trial.value:.5f}")
        study.optimize(objective, n_trials=n_trials, callbacks=[_tqdm_cb], show_progress_bar=False, gc_after_trial=True)
    best_params = study.best_params
    print(f"[INFO] Best LightGBM params: {best_params}")

    best_model = lgb.LGBMClassifier(**best_params, n_estimators=1000)
    best_model.fit(X_tr, y_tr,
                   eval_set=[(X_va, y_va)],
                   eval_metric="average_precision",
                   callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
    return best_model, best_params, X_va, y_va