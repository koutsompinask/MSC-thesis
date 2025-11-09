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
from sklearn.model_selection import train_test_split, StratifiedKFold
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
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
from tqdm.auto import tqdm
import tempfile

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

def _scale_pos_weight(y: np.ndarray) -> float:
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    return float(neg / max(1, pos))


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
def train_xgb_optuna(X, y, val_size=0.2, test_size=None, n_trials=30, random_state=42,
                     early_stopping_rounds=50, use_gpu=True):
    """Train XGBoost classifier with Optuna hyperparameter optimization.
    
    Performs hyperparameter tuning using Optuna TPE sampler to optimize
    PR-AUC score. Uses early stopping and GPU acceleration when available.
    
    Args:
        X: Training features (pandas DataFrame or numpy array).
        y: Training labels (pandas Series or numpy array).
        val_size: Fraction of data to use for validation (default: 0.2).
        test_size: Optional fraction of data to use for test set (default: None).
            If provided, creates train/val/test split. If None, uses train/val split only.
        n_trials: Number of Optuna trials for hyperparameter search (default: 30).
        random_state: Random seed for reproducibility (default: 42).
        early_stopping_rounds: Early stopping rounds for XGBoost (default: 50).
        use_gpu: Whether to use GPU acceleration if available (default: True).
        
    Returns:
        tuple: (best_model, best_params, X_va, y_va, hist_df, plot_paths) or
               (best_model, best_params, X_va, y_va, X_test, y_test, hist_df, plot_paths) if test_size is provided
            - best_model: Trained XGBoost classifier with best parameters
            - best_params: Dictionary of best hyperparameters found
            - X_va: Validation features
            - y_va: Validation labels
            - X_test: Test features (only if test_size is provided)
            - y_test: Test labels (only if test_size is provided)
            - hist_df: DataFrame of hyperparameter trial history
            - plot_paths: List of Paths to diagnostic plots
    """
    if test_size is not None:
        # Create train/val/test split
        X_tr, X_temp, y_tr, y_temp = train_test_split(
            X, y, test_size=(val_size + test_size), stratify=y, random_state=random_state
        )
        # Split temp into val and test
        val_ratio = val_size / (val_size + test_size)
        X_va, X_test, y_va, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio), stratify=y_temp, random_state=random_state
        )
    else:
        # Standard train/val split
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=val_size, stratify=y, random_state=random_state)
        X_test, y_test = None, None

    trial_history = []  # we'll collect params and scores here
    
    def objective(trial):
        params = {
            "n_estimators": 500,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 8, 15),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 1, log=True),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "tree_method": "gpu_hist" if  _gpu_available() and use_gpu else "hist",
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "scale_pos_weight": _scale_pos_weight(y_tr),
            "n_jobs": -1,
            "random_state": random_state,
        }
        try:
            model = xgb.XGBClassifier(**params,  early_stopping_rounds=early_stopping_rounds)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            y_pred = model.predict_proba(X_va)[:, 1]
            score = average_precision_score(y_va, y_pred)
            y_pred_train = model.predict_proba(X_tr)[:, 1]
            training_score = average_precision_score(y_tr, y_pred_train)
            # record params + score for later plotting
            rec = params.copy()
            rec["score"] = score
            rec["trial_number"] = trial.number
            rec["training_score"] = training_score
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
    
    # Check if any trials succeeded
    if len(trial_history) == 0:
        raise ValueError("No successful trials completed. All Optuna trials failed. Check your data and parameters.")
    
    best_params = study.best_params
    print(f"[INFO] Best XGBoost params: {best_params}")
    
    # Validate best_params contains required keys
    required_keys = ["learning_rate", "max_depth", "subsample", "colsample_bytree", 
                     "min_child_weight", "gamma"]
    missing_keys = [k for k in required_keys if k not in best_params]
    if missing_keys:
        raise ValueError(f"Best params missing required keys: {missing_keys}")

    # Retrain with best params: use n_estimators=1000 for final model (increased from 500 used in trials)
    # This allows the model to potentially improve further with more trees
    best_params_final = best_params.copy()
    best_params_final["n_estimators"] = 1000
    best_params_final["tree_method"] = "gpu_hist" if _gpu_available() and use_gpu else "hist"
    best_params_final["objective"] = "binary:logistic"
    best_params_final["eval_metric"] = "aucpr"
    best_params_final["scale_pos_weight"] = _scale_pos_weight(y_tr)
    best_params_final["n_jobs"] = -1
    best_params_final["random_state"] = random_state
    best_params_final["verbosity"] = 0
    
    best_model = xgb.XGBClassifier(**best_params_final, early_stopping_rounds=early_stopping_rounds)
    best_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    
    # Update best_params to include n_estimators for logging consistency
    best_params["n_estimators"] = 1000

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
        path_maybe = plot_param_vs_score(hist_df, p, plots_dir, "training_score")
        if path_maybe is not None:
            plot_paths.append(path_maybe)

    if test_size is not None:
        return best_model, best_params, X_va, y_va, X_test, y_test, hist_df, plot_paths
    else:
        return best_model, best_params, X_va, y_va, hist_df, plot_paths




def train_catboost_optuna(X, y, val_size=0.2, test_size=None, n_trials=30, random_state=42,
                          early_stopping_rounds=50, use_gpu=True):
    """Train CatBoost classifier with Optuna hyperparameter optimization.
    
    Performs hyperparameter tuning using Optuna TPE sampler to optimize
    PR-AUC score. Handles categorical features automatically and uses
    GPU acceleration when available.
    
    Args:
        X: Training features (pandas DataFrame or numpy array).
        y: Training labels (pandas Series or numpy array).
        val_size: Fraction of data to use for validation (default: 0.2).
        test_size: Optional fraction of data to use for test set (default: None).
            If provided, creates train/val/test split. If None, uses train/val split only.
        n_trials: Number of Optuna trials for hyperparameter search (default: 30).
        random_state: Random seed for reproducibility (default: 42).
        early_stopping_rounds: Early stopping rounds for CatBoost (default: 50).
        use_gpu: Whether to use GPU acceleration if available (default: True).
        
    Returns:
        tuple: (best_model, best_params, X_va, y_va, hist_df, plot_paths) or
               (best_model, best_params, X_va, y_va, X_test, y_test, hist_df, plot_paths) if test_size is provided
            - best_model: Trained CatBoost classifier with best parameters
            - best_params: Dictionary of best hyperparameters found
            - X_va: Validation features
            - y_va: Validation labels
            - X_test: Test features (only if test_size is provided)
            - y_test: Test labels (only if test_size is provided)
            - hist_df: DataFrame of hyperparameter trial history
            - plot_paths: List of Paths to diagnostic plots
    """
    if test_size is not None:
        # Create train/val/test split
        X_tr, X_temp, y_tr, y_temp = train_test_split(
            X, y, test_size=(val_size + test_size), stratify=y, random_state=random_state
        )
        # Split temp into val and test
        val_ratio = val_size / (val_size + test_size)
        X_va, X_test, y_va, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio), stratify=y_temp, random_state=random_state
        )
    else:
        # Standard train/val split
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=val_size, stratify=y, random_state=random_state)
        X_test, y_test = None, None
    cat_features = [i for i, c in enumerate(X_tr.columns) if str(X_tr[c].dtype) in ["object", "category"]]

    trial_history = []  # we'll collect params and scores here

    def objective(trial):
        params = {
            "iterations": 500,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 10, 16),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "eval_metric": "PRAUC",
            "task_type": "GPU" if  _gpu_available() and use_gpu else "CPU",
            "verbose": False,
            "scale_pos_weight": _scale_pos_weight(y_tr),
            "random_state": random_state,
        }
        model = CatBoostClassifier(**params)
        try:
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va),
                    cat_features=cat_features,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False)
            y_pred = model.predict_proba(X_va)[:, 1]
            score = average_precision_score(y_va, y_pred)
            y_pred_train = model.predict_proba(X_tr)[:, 1]
            training_score = average_precision_score(y_tr, y_pred_train)
            # record params + score for later plotting
            rec = params.copy()
            rec["score"] = score
            rec["trial_number"] = trial.number
            rec["training_score"] = training_score
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
    
    # Check if any trials succeeded
    if len(trial_history) == 0:
        raise ValueError("No successful trials completed. All Optuna trials failed. Check your data and parameters.")
    
    best_params = study.best_params
    print(f"[INFO] Best CatBoost params: {best_params}")
    
    # Validate best_params contains required keys
    required_keys = ["learning_rate", "depth", "l2_leaf_reg", "subsample", "border_count"]
    missing_keys = [k for k in required_keys if k not in best_params]
    if missing_keys:
        raise ValueError(f"Best params missing required keys: {missing_keys}")

    # Retrain with best params: use iterations=1000 for final model (increased from 500 used in trials)
    # This allows the model to potentially improve further with more iterations
    best_params_final = best_params.copy()
    best_params_final["iterations"] = 1000
    best_params_final["eval_metric"] = "PRAUC"
    best_params_final["task_type"] = "GPU" if _gpu_available() and use_gpu else "CPU"
    best_params_final["scale_pos_weight"] = _scale_pos_weight(y_tr)
    best_params_final["random_seed"] = random_state
    best_params_final["verbose"] = False
    
    best_model = CatBoostClassifier(**best_params_final)
    best_model.fit(X_tr, y_tr, eval_set=(X_va, y_va),
                   cat_features=cat_features,
                   early_stopping_rounds=early_stopping_rounds,
                   verbose=False)
    
    # Update best_params to include iterations for logging consistency
    best_params["iterations"] = 1000
    # Convert random_state to random_seed for consistency
    if "random_state" in best_params:
        best_params["random_seed"] = best_params.pop("random_state")
    
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
        path_maybe = plot_param_vs_score(hist_df, p, plots_dir, "training_score")
        if path_maybe is not None:
            plot_paths.append(path_maybe)

    if test_size is not None:
        return best_model, best_params, X_va, y_va, X_test, y_test, hist_df, plot_paths
    else:
        return best_model, best_params, X_va, y_va, hist_df, plot_paths

def train_lgbm_optuna(X, y, val_size=0.2, test_size=None, n_trials=30, random_state=42,
                      early_stopping_rounds=50, use_gpu=True):
    """Train LightGBM classifier with Optuna hyperparameter optimization.
    
    Performs hyperparameter tuning using Optuna TPE sampler to optimize
    PR-AUC score. Uses early stopping and GPU acceleration when available.
    
    Args:
        X: Training features (pandas DataFrame or numpy array).
        y: Training labels (pandas Series or numpy array).
        val_size: Fraction of data to use for validation (default: 0.2).
        test_size: Optional fraction of data to use for test set (default: None).
            If provided, creates train/val/test split. If None, uses train/val split only.
        n_trials: Number of Optuna trials for hyperparameter search (default: 30).
        random_state: Random seed for reproducibility (default: 42).
        early_stopping_rounds: Early stopping rounds for LightGBM (default: 50).
        use_gpu: Whether to use GPU acceleration if available (default: True).
        
    Returns:
        tuple: (best_model, best_params, X_va, y_va, hist_df, plot_paths) or
               (best_model, best_params, X_va, y_va, X_test, y_test, hist_df, plot_paths) if test_size is provided
            - best_model: Trained LightGBM classifier with best parameters
            - best_params: Dictionary of best hyperparameters found
            - X_va: Validation features
            - y_va: Validation labels
            - X_test: Test features (only if test_size is provided)
            - y_test: Test labels (only if test_size is provided)
            - hist_df: DataFrame of hyperparameter trial history
            - plot_paths: List of Paths to diagnostic plots
    """
    if test_size is not None:
        # Create train/val/test split
        X_tr, X_temp, y_tr, y_temp = train_test_split(
            X, y, test_size=(val_size + test_size), stratify=y, random_state=random_state
        )
        # Split temp into val and test
        val_ratio = val_size / (val_size + test_size)
        X_va, X_test, y_va, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio), stratify=y_temp, random_state=random_state
        )
    else:
        # Standard train/val split
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=val_size, stratify=y, random_state=random_state)
        X_test, y_test = None, None

    trial_history = []  # we'll collect params and scores here
    
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "average_precision",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", 10, 20),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "device": "gpu" if  _gpu_available() and use_gpu else "cpu",
            "random_state": random_state,
            "scale_pos_weight": _scale_pos_weight(y_tr),
        }
        model = lgb.LGBMClassifier(**params, n_estimators=500)
        try:
            model.fit(X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric="average_precision",
                    callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
            y_pred = model.predict_proba(X_va)[:, 1]
            score = average_precision_score(y_va, y_pred)
            y_pred_train = model.predict_proba(X_tr)[:, 1]
            training_score = average_precision_score(y_tr, y_pred_train)
            # record params + score for later plotting
            rec = params.copy()
            rec["score"] = score
            rec["trial_number"] = trial.number
            rec["training_score"] = training_score
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
    
    # Check if any trials succeeded
    if len(trial_history) == 0:
        raise ValueError("No successful trials completed. All Optuna trials failed. Check your data and parameters.")
    
    best_params = study.best_params
    print(f"[INFO] Best LightGBM params: {best_params}")
    
    # Validate best_params contains required keys
    required_keys = ["num_leaves", "max_depth", "learning_rate", "feature_fraction", 
                     "bagging_fraction", "bagging_freq", "min_child_samples", "lambda_l1", "lambda_l2"]
    missing_keys = [k for k in required_keys if k not in best_params]
    if missing_keys:
        raise ValueError(f"Best params missing required keys: {missing_keys}")

    # Retrain with best params: use n_estimators=1000 for final model (increased from 500 used in trials)
    # This allows the model to potentially improve further with more trees
    best_params_final = best_params.copy()
    best_params_final["n_estimators"] = 1000
    best_params_final["objective"] = "binary"
    best_params_final["metric"] = "average_precision"
    best_params_final["verbosity"] = -1
    best_params_final["boosting_type"] = "gbdt"
    best_params_final["device"] = "gpu" if _gpu_available() and use_gpu else "cpu"
    best_params_final["scale_pos_weight"] = _scale_pos_weight(y_tr)
    best_params_final["random_state"] = random_state
    
    best_model = lgb.LGBMClassifier(**best_params_final)
    best_model.fit(X_tr, y_tr,
                   eval_set=[(X_va, y_va)],
                   eval_metric="average_precision",
                   callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
    
    # Update best_params to include n_estimators for logging consistency
    best_params["n_estimators"] = 1000
    
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
        path_maybe = plot_param_vs_score(hist_df, p, plots_dir, "training_score")
        if path_maybe is not None:
            plot_paths.append(path_maybe)

    if test_size is not None:
        return best_model, best_params, X_va, y_va, X_test, y_test, hist_df, plot_paths
    else:
        return best_model, best_params, X_va, y_va, hist_df, plot_paths

def train_best_base_models_from_mlflow(
    X: pd.DataFrame,
    y: pd.Series,
    experiment_name: str,
    val_size: float = 0.2,
    test_size: float | None = None,
    metric: str = "pr_auc",
    random_state: int = 42,
    tracking_uri: str = "mlruns",
    early_stopping_rounds: int = 50,
    use_gpu: bool = True,
) -> tuple:
    """Load best models from MLflow and retrain them on the current dataset."""
    
    mlf.set_tracking_uri(tracking_uri)
    experiment = mlf.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found in MLflow.")
    experiment_id = experiment.experiment_id

    # Split data
    if test_size is not None:
        X_tr, X_temp, y_tr, y_temp = train_test_split(
            X, y, test_size=(val_size + test_size), stratify=y, random_state=random_state
        )
        val_ratio = val_size / (val_size + test_size)
        X_va, X_test, y_va, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio), stratify=y_temp, random_state=random_state
        )
    else:
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=val_size, stratify=y, random_state=random_state
        )
        X_test, y_test = None, None

    base_models = {}
    best_params_dict = {}

    for model_type in ["xgboost", "catboost", "lightgbm"]:
        print(f"[INFO] Searching for best {model_type} model in experiment '{experiment_name}'...")

        runs = mlf.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.model_type = '{model_type}'",
            order_by=[f"metrics.{metric} DESC"],
            max_results=1,
        )

        if runs.empty:
            print(f"[WARN] No {model_type} runs found — skipping.")
            continue

        best_run = runs.iloc[0]
        run_id = best_run["run_id"]
        print(f"[INFO] Found best {model_type} run {run_id} with {metric}={best_run[f'metrics.{metric}']:.4f}")

        client = mlf.tracking.MlflowClient(tracking_uri=tracking_uri)
        params = client.get_run(run_id).data.params

        # Convert param types
        best_params = {}
        for k, v in params.items():
            try:
                if "." in v:
                    best_params[k] = float(v)
                elif v.lower() in ["true", "false"]:
                    best_params[k] = v.lower() == "true"
                else:
                    best_params[k] = int(v)
            except Exception:
                best_params[k] = v
        best_params_dict[model_type] = best_params

        # Retrain model
        print(f"[INFO] Retraining {model_type} with best parameters...")

        if model_type == "xgboost":
            xgb_params = best_params.copy()
            xgb_params.update({
                "n_estimators": 1000,
                "objective": "binary:logistic",
                "eval_metric": "aucpr",
                "tree_method": "gpu_hist" if _gpu_available() and use_gpu else "hist",
                "n_jobs": -1,
                "random_state": random_state,
            })
            model = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=early_stopping_rounds)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            base_models["xgboost"] = model

        elif model_type == "catboost":
            cat_params = best_params.copy()
            cat_params.update({
                "iterations": 1000,
                "eval_metric": "PRAUC",
                "task_type": "GPU" if _gpu_available() and use_gpu else "CPU",
                "random_seed": random_state,
                "verbose": False,
            })
            cat_features = [i for i, c in enumerate(X_tr.columns) if str(X_tr[c].dtype) in ["object", "category"]]
            model = CatBoostClassifier(**cat_params)
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va), cat_features=cat_features or None, verbose=False)
            base_models["catboost"] = model

        elif model_type == "lightgbm":
            lgb_params = best_params.copy()
            lgb_params.update({
                "n_estimators": 1000,
                "objective": "binary",
                "metric": "average_precision",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "device": "gpu" if _gpu_available() and use_gpu else "cpu",
                "random_state": random_state,
            })
            model = lgb.LGBMClassifier(**lgb_params)
            model.fit(
                X_tr, y_tr, eval_set=[(X_va, y_va)],
                eval_metric="average_precision",
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
            )
            base_models["lightgbm"] = model

    if not base_models:
        raise ValueError("No base models were successfully trained.")

    return base_models, X_tr, X_va, y_tr, y_va, X_test, y_test, best_params_dict

def train_ensemble(
    base_models: dict,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
    n_trials: int = 30,
    random_state: int = 42,
):
    """Train Optuna-tuned logistic regression meta-learner using base model predictions."""
    
    print("[INFO] Generating meta-features...")
    # Build train meta features
    X_meta = np.column_stack([m.predict_proba(X_tr)[:, 1] for m in base_models.values()])
    X_meta_val = np.column_stack([m.predict_proba(X_va)[:, 1] for m in base_models.values()])

    # Display base model individual validation PR-AUC
    for name, model in base_models.items():
        y_pred_val = model.predict_proba(X_va)[:, 1]
        print(f"[INFO] {name} validation PR-AUC: {average_precision_score(y_va, y_pred_val):.4f}")

    # Prepare Optuna optimization
    trial_history = []

    def objective(trial):
        # Core hyperparameters
        C = trial.suggest_float("C", 1e-4, 100.0, log=True)
        max_iter = 1000
        class_weight = trial.suggest_categorical("class_weight", ["balanced", None])

        # Define valid penalty–solver combinations
        valid_combinations = {
            "lbfgs": ["l2", None],
            "liblinear": ["l1", "l2"],
            "newton-cg": ["l2", None],
            "sag": ["l2", None],
            "saga": ["l1", "l2", "elasticnet"],
        }

        # Sample solver first
        solver = trial.suggest_categorical("solver", list(valid_combinations.keys()))

        # Now define penalty parameter with unique name per solver
        penalty_param_name = f"penalty_{solver}"
        penalty = trial.suggest_categorical(penalty_param_name, valid_combinations[solver])
        trial.set_user_attr("penalty", penalty)
        # Handle l1_ratio for elasticnet
        l1_ratio = None
        if penalty == "elasticnet":
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

        # Build model parameters
        params = {
            "C": C,
            "penalty": penalty,
            "solver": solver,
            "max_iter": max_iter,
            "class_weight": class_weight,
            "random_state": random_state,
        }
        if l1_ratio is not None:
            params["l1_ratio"] = l1_ratio

        try:
            model = LogisticRegression(**params)
            model.fit(X_meta, y_tr)
            preds = model.predict_proba(X_meta_val)[:, 1]
            score = average_precision_score(y_va, preds)

            # Also compute training score
            y_pred_train = model.predict_proba(X_meta)[:, 1]
            training_score = average_precision_score(y_tr, y_pred_train)
        

            # Record trial info
            rec = params.copy()
            rec["score"] = score
            rec["trial_number"] = trial.number
            trial_history.append(rec)
            rec["training_score"] = training_score
            trial_history.append(rec)
            return score

        except Exception as e:
            print(f"[WARN] Trial {trial.number} failed ({e})")
            return 0.0

    print("[INFO] Optimizing meta-learner with Optuna...")
    study = optuna.create_study(direction="maximize", study_name="ensemble_meta_learner_pr_auc_optimization")
    with tqdm(total=n_trials, desc="[Optuna Ensemble Meta-Learner Tuning]", unit="trial") as pbar:
        def _tqdm_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            pbar.update(1)
            pbar.set_postfix_str(f"Trial {trial.number+1}/{n_trials} PR-AUC={trial.value:.4f}")
        study.optimize(objective, n_trials=n_trials, callbacks=[_tqdm_cb], show_progress_bar=False, gc_after_trial=True)
    
    best_params = study.best_params
    print(f"[INFO] Best meta-learner params: {best_params}")

    for key in best_params.keys():
        if key.startswith("penalty_"):
            best_params["penalty"] = best_params[key]
            del best_params[key]
            break
    
    # Only include l1_ratio if penalty is elasticnet
    if best_params.get("penalty") != "elasticnet" and "l1_ratio" in best_params:
        best_params.pop("l1_ratio")
    
    # Final model
    ensemble = LogisticRegression(**best_params, random_state=random_state)
    ensemble.fit(X_meta, y_tr)
    # Prepare best_params for logging (convert None to string for MLflow compatibility)
    best_params_log = best_params.copy()
    best_params_log["random_state"] = random_state
    if "class_weight" in best_params_log and best_params_log["class_weight"] is None:
        best_params_log["class_weight"] = "None"
    # Handle None penalty for MLflow logging
    if best_params_log.get("penalty") is None:
        best_params_log["penalty"] = "None"
    # Remove l1_ratio if not elasticnet for cleaner logging
    if best_params_log.get("penalty") != "elasticnet" and "l1_ratio" in best_params_log:
        best_params_log.pop("l1_ratio")
    
    # Evaluate ensemble on validation set
    y_prob_ensemble = ensemble.predict_proba(X_meta_val)[:, 1]
    ensemble_pr_auc = average_precision_score(y_va, y_prob_ensemble)
    ensemble_roc_auc = roc_auc_score(y_va, y_prob_ensemble)
    
    print(f"[INFO] Ensemble validation PR-AUC: {ensemble_pr_auc:.4f}")
    print(f"[INFO] Ensemble validation ROC-AUC: {ensemble_roc_auc:.4f}")
    
    X_meta_test = None
    # Evaluate on test set if available
    if X_test is not None:
        test_base_predictions = []
        for name, model in base_models.items():
            y_prob = model.predict_proba(X_test)[:, 1]
            test_base_predictions.append(y_prob)
        X_meta_test = np.column_stack(test_base_predictions)
        y_prob_ensemble_test = ensemble.predict_proba(X_meta_test)[:, 1]
        ensemble_pr_auc_test = average_precision_score(y_test, y_prob_ensemble_test)
        ensemble_roc_auc_test = roc_auc_score(y_test, y_prob_ensemble_test)
        
        print(f"[INFO] Ensemble test PR-AUC: {ensemble_pr_auc_test:.4f}")
        print(f"[INFO] Ensemble test ROC-AUC: {ensemble_roc_auc_test:.4f}")
    
    # Build history df + plots
    hist_df = pd.DataFrame(trial_history)
    plots_dir = Path("hp_search_plots/ensemble")
    plot_paths = []
    for p in ["C", "penalty", "solver", "max_iter", "class_weight", "l1_ratio"]:
        path_maybe = plot_param_vs_score(hist_df, p, plots_dir)
        if path_maybe is not None:
            plot_paths.append(path_maybe)
        path_maybe = plot_param_vs_score(hist_df, p, plots_dir, "training_score")
        if path_maybe is not None:
            plot_paths.append(path_maybe)

        if X_test is not None:
            return ensemble, base_models, X_meta_val, y_va, X_meta_test, y_test, best_params_log, hist_df, plot_paths
        else:
            return ensemble, base_models, X_va, y_va, best_params_log, hist_df, plot_paths
