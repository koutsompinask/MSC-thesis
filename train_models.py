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
import mlflow.xgboost as mlf_xgboost, mlflow.catboost as mlf_catboost, mlflow.lightgbm as mlf_lightgbm, mlflow.sklearn as mlf_sklearn, mlflow.pytorch as mlf_pytorch
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
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def _gpu_available() -> bool:
    """Check if GPU is available for XGBoost training.
    
    Attempts to create and fit a minimal XGBoost model with GPU acceleration.
    If successful, GPU is available; otherwise, falls back to CPU.
    
    Returns:
        bool: True if GPU is available and working, False otherwise.
    """
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



def _plot_artifacts(y_true: np.ndarray, y_prob: np.ndarray, out_dir: Path) -> dict[str, Path]:
    """Generate evaluation plots and save them to disk.
    
    Creates ROC curve, Precision-Recall curve, and confusion matrix plots
    for model evaluation and saves them as PNG files.
    
    Args:
        y_true: True binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class.
        out_dir: Directory path where plots will be saved.
        
    Returns:
        dict[str, Path]: Dictionary mapping plot names to file paths.
            Keys: 'roc_curve', 'pr_curve', 'confusion_matrix'
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

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
    tracking_uri: str = "mlruns",
    hp_search_history: pd.DataFrame | None = None,
    hp_search_plots: list[Path] | None = None,
):
    """Evaluate a trained binary classifier and log results to MLflow.
    
    Performs comprehensive evaluation including metrics calculation, SHAP explainability,
    calibration analysis, and visualization generation. Results are logged to MLflow
    with artifacts, metrics, and model storage.
    
    Args:
        model: Trained binary classifier (XGBoost, CatBoost, LightGBM, or sklearn-style).
        X_va: Validation features as pandas DataFrame.
        y_va: Validation labels as pandas Series.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.
        model_type: Model type string. If None, auto-detects from model class.
        best_params: Dictionary of best hyperparameters to log.
        tracking_uri: MLflow tracking URI (default: "mlruns").
        hp_search_history: Optional DataFrame of hyperparameter search history.
        hp_search_plots: Optional list of Paths to hyperparameter search diagnostic plots.
        
    Returns:
        dict: Dictionary containing evaluation metrics (roc_auc, pr_auc, precision, recall, f1).
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
        elif "simplenn" in cname or "module" in cname:
            model_type = "nn"
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
        if model_type == "nn":
            # Neural network models need scaled input as tensors
            # The scaler should be attached to the model when returned from train_nn_optuna
            if hasattr(model, "scaler"):
                X_va_scaled = model.scaler.transform(X_va)
            else:
                # If no scaler attached, assume X_va is already scaled (fallback)
                X_va_scaled = X_va.values if hasattr(X_va, "values") else X_va
            X_va_t = torch.tensor(X_va_scaled, dtype=torch.float32)
            device = next(model.parameters()).device
            model.eval()
            with torch.no_grad():
                logits = model(X_va_t.to(device))
                y_prob = torch.sigmoid(logits).cpu().numpy().flatten()
        elif hasattr(model, "predict_proba"):
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
            mlf.log_artifact(str(path), artifact_path="plots")

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
                elif model_type == "nn":
                    mlf_pytorch.log_model(model, name="model", input_example=input_example)
                else:
                    mlf_sklearn.log_model(model, name="model", input_example=input_example)
            except Exception as e:
                print(f"[WARN] Could not log model: {e}")
        
        if hp_search_history is not None:
            # save the full trial history as csv
            hp_dir = Path("hp_search_artifacts")
            hp_dir.mkdir(parents=True, exist_ok=True)
            hist_path = hp_dir / "optuna_trials.csv"
            hp_search_history.to_csv(hist_path, index=False)
            mlf.log_artifact(str(hist_path), artifact_path="hp_search")

            # also log summary stats for convenience
            mlf.log_metric("hp_best_score", float(hp_search_history["score"].max()))
            mlf.log_metric("hp_avg_score", float(hp_search_history["score"].mean()))
            mlf.log_metric("hp_std_score", float(hp_search_history["score"].std()))

        if hp_search_plots:
            for p in hp_search_plots:
                mlf.log_artifact(str(p), artifact_path="hp_search")

        print("[INFO] Evaluation complete and logged.")
    return metrics

def plot_param_vs_score(df: pd.DataFrame, param_name: str, out_dir: Path, metric_name: str = "score"):
    """
    Generate a diagnostic plot of model score vs hyperparameter value.
    Saves the plot to disk and returns the path.

    Parameters
    ----------
    df : pd.DataFrame
        Trial history DataFrame containing at least [param_name, metric_name].
    param_name : str
        Name of the hyperparameter column to plot.
    out_dir : pathlib.Path
        Directory to save the resulting plot.
    metric_name : str
        Name of the metric column in df (default: 'score').

    Returns
    -------
    Path or None
        Path to the saved PNG file, or None if column missing.
    """
    if param_name not in df.columns:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    col_type = df[param_name].dtype.kind
    if col_type in ("i", "O", "b"):  # integer, object, boolean
        grouped = df.groupby(param_name)[metric_name].mean().reset_index()
        grouped = grouped.sort_values(param_name)
        plt.bar(grouped[param_name].astype(str), grouped[metric_name], color="#1f77b4")
        plt.xlabel(param_name)
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} vs {param_name} (binned mean)")
        plt.xticks(rotation=45, ha="right")
    else:
        plt.scatter(df[param_name], df[metric_name], color="#1f77b4", alpha=0.7, s=40)
        plt.xlabel(param_name)
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} vs {param_name}")
    plt.tight_layout()

    out_path = out_dir / f"{param_name}_vs_{metric_name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

# ---------------------------
#  MODEL-SPECIFIC TRAINING
# ---------------------------
def train_xgb_optuna(X, y, val_size=0.2, n_trials=30, random_state=42,
                     early_stopping_rounds=50, use_gpu=True):
    """Train XGBoost classifier with Optuna hyperparameter optimization.
    
    Performs hyperparameter tuning using Optuna TPE sampler to optimize
    PR-AUC score. Uses early stopping and GPU acceleration when available.
    
    Args:
        X: Training features (pandas DataFrame or numpy array).
        y: Training labels (pandas Series or numpy array).
        val_size: Fraction of data to use for validation (default: 0.2).
        n_trials: Number of Optuna trials for hyperparameter search (default: 30).
        random_state: Random seed for reproducibility (default: 42).
        early_stopping_rounds: Early stopping rounds for XGBoost (default: 50).
        use_gpu: Whether to use GPU acceleration if available (default: True).
        
    Returns:
        tuple: (best_model, best_params, X_va, y_va)
            - best_model: Trained XGBoost classifier with best parameters
            - best_params: Dictionary of best hyperparameters found
            - X_va: Validation features
            - y_va: Validation labels
            - hist_df: DataFrame of hyperparameter trial history
            - plot_paths: List of Paths to diagnostic plots
    """
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=val_size, stratify=y, random_state=random_state)

    trial_history = []  # we'll collect params and scores here
    
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
        try:
            model = xgb.XGBClassifier(**params,  early_stopping_rounds=early_stopping_rounds)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            y_pred = model.predict_proba(X_va)[:, 1]
            score = average_precision_score(y_va, y_pred)
            # record params + score for later plotting
            rec = params.copy()
            rec["score"] = score
            rec["trial_number"] = trial.number
            trial_history.append(rec)
        except Exception as e:
            score = 0.0
            print(f"[WARN] Trial {trial.number} failed with error: {e}")
        return score

    study = optuna.create_study(direction="maximize", study_name="xgboost_pr_auc_optimization")
    with tqdm(total=n_trials, desc="[Optuna XGBoost Tuning]", unit="trial") as pbar:
        def _tqdm_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            pbar.update(1)
            pbar.set_postfix_str(f"Trial {trial.number+1}/{n_trials} PR-AUC={trial.value:.4f}")
        study.optimize(objective, n_trials=n_trials, callbacks=[_tqdm_cb], show_progress_bar=False, gc_after_trial=True)
    best_params = study.best_params
    print(f"[INFO] Best XGBoost params: {best_params}")

    best_model = xgb.XGBClassifier(**best_params,  early_stopping_rounds=early_stopping_rounds)
    best_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    # -----------------
    # Build history df + plots
    # -----------------
    hist_df = pd.DataFrame(trial_history)
    plots_dir = Path("hp_search_plots/xgb")
    plot_paths = []
    for p in ["learning_rate", "max_depth", "subsample", "colsample_bytree",
              "min_child_weight", "gamma"]:
        path_maybe = plot_param_vs_score(hist_df, p, plots_dir)
        if path_maybe is not None:
            plot_paths.append(path_maybe)

    return best_model, best_params, X_va, y_va, hist_df, plot_paths




def train_catboost_optuna(X, y, val_size=0.2, n_trials=30, random_state=42,
                          early_stopping_rounds=50, use_gpu=True):
    """Train CatBoost classifier with Optuna hyperparameter optimization.
    
    Performs hyperparameter tuning using Optuna TPE sampler to optimize
    PR-AUC score. Handles categorical features automatically and uses
    GPU acceleration when available.
    
    Args:
        X: Training features (pandas DataFrame or numpy array).
        y: Training labels (pandas Series or numpy array).
        val_size: Fraction of data to use for validation (default: 0.2).
        n_trials: Number of Optuna trials for hyperparameter search (default: 30).
        random_state: Random seed for reproducibility (default: 42).
        early_stopping_rounds: Early stopping rounds for CatBoost (default: 50).
        use_gpu: Whether to use GPU acceleration if available (default: True).
        
    Returns:
        tuple: (best_model, best_params, X_va, y_va)
            - best_model: Trained CatBoost classifier with best parameters
            - best_params: Dictionary of best hyperparameters found
            - X_va: Validation features
            - y_va: Validation labels
            - hist_df: DataFrame of hyperparameter trial history
            - plot_paths: List of Paths to diagnostic plots
    """
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=val_size, stratify=y, random_state=random_state)
    cat_features = [i for i, c in enumerate(X_tr.columns) if str(X_tr[c].dtype) in ["object", "category"]]

    trial_history = []  # we'll collect params and scores here

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
        try:
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va),
                    cat_features=cat_features,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False)
            y_pred = model.predict_proba(X_va)[:, 1]
            score = average_precision_score(y_va, y_pred)
            # record params + score for later plotting
            rec = params.copy()
            rec["score"] = score
            rec["trial_number"] = trial.number
            trial_history.append(rec)
        except Exception as e:
            score = 0.0
            print(f"[WARN] Trial {trial.number} failed with error: {e}")
        return score

    study = optuna.create_study(direction="maximize", study_name="catboost_pr_auc_optimization")
    with tqdm(total=n_trials, desc="[Optuna CatBoost Tuning]", unit="trial") as pbar:
        def _tqdm_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            pbar.update(1)
            pbar.set_postfix_str(f"Trial {trial.number+1}/{n_trials} PR-AUC={trial.value:.4f}")
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
        # -----------------
    # Build history df + plots
    # -----------------
    hist_df = pd.DataFrame(trial_history)
    plots_dir = Path("hp_search_plots/cat")
    plot_paths = []
    for p in ["learning_rate", "depth", "subsample", "l2_leaf_reg", "border_count"]:
        path_maybe = plot_param_vs_score(hist_df, p, plots_dir)
        if path_maybe is not None:
            plot_paths.append(path_maybe)

    return best_model, best_params, X_va, y_va, hist_df, plot_paths

def train_lgbm_optuna(X, y, val_size=0.2, n_trials=30, random_state=42,
                      early_stopping_rounds=50, use_gpu=True):
    """Train LightGBM classifier with Optuna hyperparameter optimization.
    
    Performs hyperparameter tuning using Optuna TPE sampler to optimize
    PR-AUC score. Uses early stopping and GPU acceleration when available.
    
    Args:
        X: Training features (pandas DataFrame or numpy array).
        y: Training labels (pandas Series or numpy array).
        val_size: Fraction of data to use for validation (default: 0.2).
        n_trials: Number of Optuna trials for hyperparameter search (default: 30).
        random_state: Random seed for reproducibility (default: 42).
        early_stopping_rounds: Early stopping rounds for LightGBM (default: 50).
        use_gpu: Whether to use GPU acceleration if available (default: True).
        
    Returns:
        tuple: (best_model, best_params, X_va, y_va)
            - best_model: Trained LightGBM classifier with best parameters
            - best_params: Dictionary of best hyperparameters found
            - X_va: Validation features
            - y_va: Validation labels
            - hist_df: DataFrame of hyperparameter trial history
            - plot_paths: List of Paths to diagnostic plots
    """
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=val_size, stratify=y, random_state=random_state)

    trial_history = []  # we'll collect params and scores here
    
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
            # record params + score for later plotting
            rec = params.copy()
            rec["score"] = score
            rec["trial_number"] = trial.number
            trial_history.append(rec)
        except Exception as e:
            score = 0.0
            print(f"[WARN] Trial {trial.number} failed with error: {e}")

        return score

    study = optuna.create_study(direction="maximize", study_name="lgbm_pr_auc_optimization")
    with tqdm(total=n_trials, desc="[Optuna LightGBM Tuning]", unit="trial") as pbar:
        def _tqdm_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            pbar.update(1)
            pbar.set_postfix_str(f"Trial {trial.number+1}/{n_trials} PR-AUC={trial.value:.4f}")
        study.optimize(objective, n_trials=n_trials, callbacks=[_tqdm_cb], show_progress_bar=False, gc_after_trial=True)
    best_params = study.best_params
    print(f"[INFO] Best LightGBM params: {best_params}")

    best_model = lgb.LGBMClassifier(**best_params, n_estimators=1000)
    best_model.fit(X_tr, y_tr,
                   eval_set=[(X_va, y_va)],
                   eval_metric="average_precision",
                   callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
    
    # -----------------
    # Build history df + plots
    # -----------------
    hist_df = pd.DataFrame(trial_history)
    plots_dir = Path("hp_search_plots/lgbm")
    plot_paths = []
    for p in ["learning_rate", "max_depth", "num_leaves", "feature_fraction",
              "bagging_fraction", "bagging_freq", "min_child_samples", "lambda_l1", "lambda_l2"]:
        path_maybe = plot_param_vs_score(hist_df, p, plots_dir)
        if path_maybe is not None:
            plot_paths.append(path_maybe)

    return best_model, best_params, X_va, y_va, hist_df, plot_paths


class SimpleNN(nn.Module):
    def __init__(self, input_dim, n_layers, hidden_dim, activation_fn, dropout=0.0):
        super().__init__()
        layers = []
        act_layer = {
            "relu": nn.ReLU(),
            "leakyrelu": nn.LeakyReLU(0.1),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }[activation_fn]

        last_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(act_layer)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        # Don't add sigmoid here - we'll use BCEWithLogitsLoss which is more stable
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
    def predict_proba(self, x):
        """Predict probabilities by applying sigmoid to logits."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits).cpu().numpy()


def train_nn_optuna(
    X,
    y,
    val_size=0.2,
    n_trials=30,
    random_state=42,
    max_epochs=30,
    batch_size=512,
    device=None,
):
    """Train a simple feedforward neural network with Optuna hyperparameter tuning.

    Args:
        X: Training features (pandas DataFrame or numpy array).
        y: Training labels (pandas Series or numpy array).
        val_size: Fraction of data for validation.
        n_trials: Number of Optuna trials.
        random_state: Random seed.
        max_epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        device: 'cuda' or 'cpu'. Auto-detects if None.

    Returns:
        tuple: (best_model, best_params, X_va, y_va, hist_df, plot_paths, scaler)
            - best_model: Trained neural network model
            - best_params: Dictionary of best hyperparameters found
            - X_va: Validation features (original, unscaled)
            - y_va: Validation labels
            - hist_df: DataFrame of hyperparameter trial history
            - plot_paths: List of Paths to diagnostic plots
            - scaler: StandardScaler fitted on training data (needed for evaluation)
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import StandardScaler

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=val_size, stratify=y, random_state=random_state)

    # Normalize features for neural network training
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)

    X_tr_t = torch.tensor(X_tr_scaled, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr.values, dtype=torch.float32).unsqueeze(1)
    X_va_t = torch.tensor(X_va_scaled, dtype=torch.float32)
    y_va_t = torch.tensor(y_va.values, dtype=torch.float32).unsqueeze(1)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    val_ds = TensorDataset(X_va_t, y_va_t)

    trial_history = []

    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 1, 4)
        hidden_dim = trial.suggest_int("hidden_dim", 32, 512, log=True)
        activation = trial.suggest_categorical("activation", ["relu", "leakyrelu", "tanh", "sigmoid"])
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)

        model = SimpleNN(X_tr_t.shape[1], n_layers, hidden_dim, activation, dropout).to(device)
        criterion = nn.BCEWithLogitsLoss()  # More numerically stable than BCELoss + Sigmoid
        optimizer = optim.Adam(model.parameters(), lr=lr)
        best_val_loss = float("inf")
        patience, patience_counter = 5, 0

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Initialize validation predictions and labels
        val_preds = []
        val_true = []

        for epoch in range(max_epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                # Check for NaN/Inf in logits
                if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
                    return 0.0
                # Ensure targets are in valid range [0, 1]
                yb_clamped = torch.clamp(yb, min=0.0, max=1.0)
                loss = criterion(logits, yb_clamped)
                # Check for invalid loss values
                if torch.isnan(loss) or torch.isinf(loss):
                    return 0.0
                loss.backward()
                optimizer.step()

            # validation
            model.eval()
            with torch.no_grad():
                val_preds_epoch = []
                val_true_epoch = []
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    # Get logits, apply sigmoid to get probabilities
                    logits = model(xb)
                    preds = torch.sigmoid(logits).cpu().numpy()
                    val_preds_epoch.extend(preds.flatten())
                    val_true_epoch.extend(yb.numpy().flatten())

                val_loss = np.mean((np.array(val_preds_epoch) - np.array(val_true_epoch)) ** 2)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Update best validation predictions
                    val_preds = val_preds_epoch.copy()
                    val_true = val_true_epoch.copy()
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    break

        # Use stored best validation predictions for PR-AUC
        if len(val_preds) == 0:
            # Fallback: collect all validation predictions if early stopping happened before any update
            model.eval()
            with torch.no_grad():
                val_preds = []
                val_true = []
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    # Get logits, apply sigmoid to get probabilities
                    logits = model(xb)
                    preds = torch.sigmoid(logits).cpu().numpy()
                    val_preds.extend(preds.flatten())
                    val_true.extend(yb.numpy().flatten())

        # compute PR-AUC
        pr_auc = average_precision_score(val_true, val_preds)

        # record params
        rec = {
            "trial_number": trial.number,
            "score": pr_auc,
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "activation": activation,
            "learning_rate": lr,
            "dropout": dropout,
        }
        trial_history.append(rec)
        return pr_auc

    study = optuna.create_study(direction="maximize", study_name="nn_pr_auc_optimization")
    with tqdm(total=n_trials, desc="[Optuna NN Tuning]", unit="trial") as pbar:
        def _tqdm_cb(study, trial):
            pbar.update(1)
            pbar.set_postfix_str(f"Trial {trial.number+1}/{n_trials} PR-AUC={trial.value:.4f}")
        study.optimize(objective, n_trials=n_trials, callbacks=[_tqdm_cb], gc_after_trial=True)

    best_params = study.best_params
    print(f"[INFO] Best NN params: {best_params}")

    # Train best model fully
    dropout = best_params.get("dropout", 0.0)
    model = SimpleNN(X_tr_t.shape[1], best_params["n_layers"], best_params["hidden_dim"], 
                     best_params["activation"], dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=best_params["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()  # More numerically stable
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            # Check for NaN/Inf
            if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
                print(f"[WARN] Invalid logits detected in final training, skipping epoch {epoch}")
                continue
            yb_clamped = torch.clamp(yb, min=0.0, max=1.0)
            loss = criterion(logits, yb_clamped)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARN] Invalid loss detected in final training, skipping epoch {epoch}")
                continue
            loss.backward()
            optimizer.step()

    hist_df = pd.DataFrame(trial_history)
    plots_dir = Path("hp_search_plots/nn")
    plot_paths = []
    for p in ["n_layers", "hidden_dim", "activation", "learning_rate", "dropout"]:
        path_maybe = plot_param_vs_score(hist_df, p, plots_dir)
        if path_maybe:
            plot_paths.append(path_maybe)

    # Attach scaler to model for convenience (optional, for evaluate_and_log)
    model.scaler = scaler
    
    return model, best_params, X_va, y_va, hist_df, plot_paths, scaler
