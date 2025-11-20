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
import optuna
from optuna.pruners import MedianPruner
import mlflow as mlf
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
from tqdm.auto import tqdm

def _gpu_available() -> bool:
    """Check if GPU is available for XGBoost training.
    
    Attempts to create and fit a minimal XGBoost model with GPU acceleration.
    If successful, GPU is available; otherwise, falls back to CPU.
    
    Returns:
        bool: True if GPU is available and working with XGBoost, False otherwise.
    
    Example:
        >>> use_gpu = _gpu_available()
        >>> print(f"GPU acceleration available: {use_gpu}")
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

def _scale_pos_weight(y: np.ndarray, factor = 1.0) -> float:
    """Calculate scaling factor for positive class weight in imbalanced datasets.
    
    Args:
        y: Array of binary labels (0 or 1).
        factor: Multiplier factor to adjust the scale of the weight. Defaults to 1.
        
    Returns:
        float: Ratio of negative to positive samples, used to balance class weights.
        
    Example:
        >>> y = np.array([0, 0, 0, 1])  # imbalanced dataset
        >>> weight = _scale_pos_weight(y)
        >>> print(f"Positive class weight scaling: {weight}")  # would print 3.0
    """
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    return factor*float(neg / max(1, pos))

def eval_function(y_true, y_prob, threshold: float = 0.5) -> float:
    """Custom evaluation function combining false negatives and false positives.
    
    Args:
        y_true: True binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class.
        threshold: Classification threshold for converting probabilities to binary predictions. Defaults to 0.5.
        
    Returns:
        float: Custom fraud cost metric = (100*FN + FP) / total_samples.
        
    Example:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        >>> cost = eval_function(y_true, y_prob, threshold=0.5)
        >>> print(f"Fraud cost: {cost:.4f}")
    """
    fp = np.sum((y_true == 0) & (y_prob >= threshold))
    fn = np.sum((y_true == 1) & (y_prob < threshold))
    return (100*fn + fp)/len(y_true)

def eval_function_lgbm(y_true, y_prob, threshold: float = 0.5) -> tuple[str, float, bool]:
    """Custom evaluation function wrapper for LightGBM.
    
    Args:
        y_true: True binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class.
        threshold: Classification threshold for converting probabilities to binary predictions. Defaults to 0.5.
        
    Returns:
        tuple: (metric_name, metric_value, is_higher_better) where metric_name is "fraud_cost".
    """
    return ('fraud_cost', eval_function(y_true, y_prob, threshold), False)

def plot_param_vs_score(df: pd.DataFrame, param_name: str, out_dir: Path, metric_name: str = "score"):
    """Generate a diagnostic plot showing hyperparameter value vs model performance.
    
    Creates a visualization to analyze the relationship between hyperparameter values
    and model performance metrics. The plot type automatically adapts based on the
    parameter type:
    - For categorical/integer/boolean parameters: Bar plot showing mean scores
    - For continuous parameters: Scatter plot of individual trials
    
    Args:
        df: Trial history DataFrame with at least [param_name, metric_name] columns.
        param_name: Name of the hyperparameter column to visualize.
        out_dir: Directory where the plot will be saved.
        metric_name: Name of the metric column in df. Defaults to "score".
        
    Returns:
        Path | None: Path to the saved PNG plot file if successful, None if the 
            parameter column is missing from the DataFrame.
            
    Example:
        >>> history_df = pd.DataFrame({
        ...     "learning_rate": [0.01, 0.1, 0.3],
        ...     "score": [0.85, 0.90, 0.87]
        ... })
        >>> path = plot_param_vs_score(
        ...     df=history_df,
        ...     param_name="learning_rate",
        ...     out_dir=Path("hyperparameter_plots")
        ... )
        >>> print(f"Diagnostic plot saved to: {path}")
        
    Note:
        - Plots are saved as PNG files with 150 DPI resolution
        - For categorical parameters, values are sorted before plotting
        - Bar plots include error bars showing score variance
        - Scatter plots use partial transparency to show point density
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
def train_xgb_optuna(X, y, test_size=0.2, n_trials=30, random_state=42,
                     early_stopping_rounds=50, use_gpu=True):
    """Train an XGBoost classifier with Optuna hyperparameter optimization.
    
    Uses Optuna's TPE sampler to optimize hyperparameters targeting PR-AUC score.
    Features include GPU acceleration (if available), early stopping, and 
    comprehensive hyperparameter visualization.
    
    Args:
        X: Training features as pandas DataFrame or numpy array.
        y: Training labels as pandas Series or numpy array.
        test_size: Fraction of data to use for test set. Defaults to 0.2.
        n_trials: Number of Optuna trials for hyperparameter search. Defaults to 30.
        random_state: Random seed for reproducibility. Defaults to 42.
        early_stopping_rounds: Number of rounds with no improvement to trigger early
            stopping. Defaults to 50.
        use_gpu: Whether to use GPU acceleration if available. Defaults to True.
        
    Returns:
        tuple: A tuple containing:
            best_model: Trained XGBoost classifier with best parameters
            best_params: Dictionary of best hyperparameters found
            X_va: Validation features
            y_va: Validation labels
            hist_df: DataFrame containing trial history with parameters and scores
            plot_paths: List of Paths to hyperparameter visualization plots
    
    Raises:
        ValueError: If no successful trials complete or best parameters are missing
            required keys.
    
    Example:
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=1000, random_state=42)
        >>> model, params, X_val, y_val, history, plots = train_xgb_optuna(X, y)
        >>> print(f"Best PR-AUC: {history['score'].max():.4f}")
    
    Note:
        Hyperparameters optimized include:
        - learning_rate: [0.01, 0.3] log-uniform
        - max_depth: [8, 15]
        - subsample: [0.5, 1.0]
        - colsample_bytree: [0.5, 1.0]
        - min_child_weight: [1e-3, 1.0] log-uniform
        - gamma: [0, 5]
    """
    # Standard train/val split
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    trial_history = []  # we'll collect params and scores here
    
    def objective(trial):
        factor = trial.suggest_float("weight_factor", 100.0, 1000.0)
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
            "eval_metric": eval_function,
            "scale_pos_weight": _scale_pos_weight(y_tr, factor),
            "n_jobs": -1,
            "random_state": random_state,
        }
        try:
            # 5-fold cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            cv_scores = []
            cv_training_scores = []
            
            for train_idx, val_idx in cv.split(X_tr, y_tr):
                X_train_fold, X_val_fold = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
                y_train_fold, y_val_fold = y_tr.iloc[train_idx], y_tr.iloc[val_idx]
                
                model = xgb.XGBClassifier(**params, early_stopping_rounds=early_stopping_rounds, enable_categorical=True)
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    verbose=False
                )
                
                # Evaluate fold
                y_prob = model.predict_proba(X_val_fold)[:, 1]
                fold_score = eval_function(y_val_fold, y_prob)
                cv_scores.append(fold_score)
                
                # Training score for this fold
                y_prob_train = model.predict_proba(X_train_fold)[:, 1]
                train_score = eval_function(y_train_fold, y_prob_train)
                cv_training_scores.append(train_score)
            
            # Average scores across folds
            score = np.mean(cv_scores)
            train_score = np.mean(cv_training_scores)
            
            # Record trial results including CV stats
            trial_history.append({
                **params,
                "score": score,
                "training_score": train_score,
                "cv_scores": cv_scores,
                "cv_std": np.std(cv_scores),
                "weight_factor": factor
            })
        except Exception as e:
            score = 0.0
            print(f"[WARN] Trial {trial.number} failed with error: {e}")
        return score

    study = optuna.create_study(direction="minimize", study_name="xgboost_customloss_optimization", pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=5))
    with tqdm(total=n_trials, desc="[Optuna XGBoost Tuning]", unit="trial") as pbar:
        def _tqdm_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            pbar.update(1)
            pbar.set_postfix_str(f"Trial {trial.number+1}/{n_trials} loss={trial.value:.4f}")
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
    best_params_final["scale_pos_weight"] = _scale_pos_weight(y_tr)
    best_params_final["n_jobs"] = -1
    best_params_final["random_state"] = random_state
    best_params_final["verbosity"] = 0
    
    best_model = xgb.XGBClassifier(**best_params_final, early_stopping_rounds=early_stopping_rounds, enable_categorical=True)
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
              "min_child_weight", "gamma", "weight_factor"]:
        path_maybe = plot_param_vs_score(hist_df, p, plots_dir)
        if path_maybe is not None:
            plot_paths.append(path_maybe)
        path_maybe = plot_param_vs_score(hist_df, p, plots_dir, "training_score")
        if path_maybe is not None:
            plot_paths.append(path_maybe)

    return best_model, best_params, X_va, y_va, hist_df, plot_paths


def train_catboost_optuna(X, y, test_size=0.2, n_trials=30, random_state=42,
                          early_stopping_rounds=50, use_gpu=True):
    """Train a CatBoost classifier with automatic hyperparameter optimization.
    
    Uses Optuna's TPE sampler to optimize hyperparameters targeting PR-AUC score.
    Automatically handles categorical features and includes GPU acceleration support.
    
    Args:
        X: Training features as pandas DataFrame or numpy array.
        y: Training labels as pandas Series or numpy array.
        test_size: Fraction of data to use for test set. Defaults to 0.2.
        n_trials: Number of Optuna trials for hyperparameter search. Defaults to 30.
        random_state: Random seed for reproducibility. Defaults to 42.
        early_stopping_rounds: Number of rounds with no improvement to trigger early
            stopping. Defaults to 50.
        use_gpu: Whether to use GPU acceleration if available. Defaults to True.
        
    Returns:
        tuple: (best_model, best_params, X_va, y_va, hist_df, plot_paths) or
               (best_model, best_params, X_va, y_va, X_test, y_test, hist_df, plot_paths) if test_size is provided
            - best_model: Trained CatBoost classifier with best parameters
            - best_params: Dictionary of best hyperparameters found
            - X_va: Validation features
            - y_va: Validation labels
            - hist_df: DataFrame of hyperparameter trial history
            - plot_paths: List of Paths to diagnostic plots
    """
    # Standard train/val split
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    trial_history = []  # we'll collect params and scores here
    categorical_features = X.select_dtypes(include=['category', 'object']).columns.tolist()

    def objective(trial):
        factor = trial.suggest_float("weight_factor", 100.0, 1000.0)
        params = {
            "iterations": 500,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 10, 16),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "eval_metric": "Recall",
            "task_type": "GPU" if  _gpu_available() and use_gpu else "CPU",
            "verbose": False,
            "scale_pos_weight": _scale_pos_weight(y_tr, factor),
            "random_state": random_state,
        }
        model = CatBoostClassifier(**params, cat_features=categorical_features)
        try:
            # 5-fold cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            cv_scores = []
            cv_training_scores = []
            
            for train_idx, val_idx in cv.split(X_tr, y_tr):
                X_train_fold, X_val_fold = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
                y_train_fold, y_val_fold = y_tr.iloc[train_idx], y_tr.iloc[val_idx]
                
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False
                )
                # Evaluate fold
                y_prob = model.predict_proba(X_val_fold)[:, 1]
                fold_score = eval_function(y_val_fold, y_prob)
                cv_scores.append(fold_score)
                
                # Training score for this fold
                y_prob_train = model.predict_proba(X_train_fold)[:, 1]
                train_score = eval_function(y_train_fold, y_prob_train)
                cv_training_scores.append(train_score)
            
            # Average scores across folds
            score = np.mean(cv_scores)
            train_score = np.mean(cv_training_scores)
            
            # Record trial results including CV stats
            trial_history.append({
                **params,
                "score": score,
                "training_score": train_score,
                "cv_scores": cv_scores,
                "cv_std": np.std(cv_scores),
                "weight_factor": factor
            })
        except Exception as e:
            score = 0.0
            print(f"[WARN] Trial {trial.number} failed with error: {e}")
        return score

    study = optuna.create_study(direction="minimize", study_name="catboost_custom_cost_optimization")
    with tqdm(total=n_trials, desc="[Optuna CatBoost Tuning]", unit="trial") as pbar:
        def _tqdm_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            pbar.update(1)
            pbar.set_postfix_str(f"Trial {trial.number+1}/{n_trials} score={trial.value:.4f}")
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
    best_params_final.pop("weight_factor", None)  # remove weight_factor from final params
    
    best_model = CatBoostClassifier(**best_params_final)
    best_model.fit(X_tr, y_tr, eval_set=(X_va, y_va),
                   early_stopping_rounds=early_stopping_rounds,
                   verbose=False,
                   cat_features=categorical_features)
    
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
    for p in ["learning_rate", "depth", "subsample", "l2_leaf_reg", "border_count","weight_factor"]:
        path_maybe = plot_param_vs_score(hist_df, p, plots_dir)
        if path_maybe is not None:
            plot_paths.append(path_maybe)
        path_maybe = plot_param_vs_score(hist_df, p, plots_dir, "training_score")
        if path_maybe is not None:
            plot_paths.append(path_maybe)
    
    return best_model, best_params, X_va, y_va, hist_df, plot_paths

def train_lgbm_optuna(X, y, test_size=0.2, n_trials=30, random_state=42,
                      early_stopping_rounds=50, use_gpu=True):
    """Train a LightGBM classifier with Optuna hyperparameter optimization.
    
    Uses Optuna's TPE sampler to optimize hyperparameters targeting PR-AUC score.
    Features include GPU acceleration support, early stopping, and gradient-based
    boosting optimizations specific to LightGBM.
    
    Args:
        X: Training features as pandas DataFrame or numpy array.
        y: Training labels as pandas Series or numpy array.
        test_size: Fraction of data to use for test set. Defaults to 0.2.
        n_trials: Number of Optuna trials for hyperparameter search. Defaults to 30.
        random_state: Random seed for reproducibility. Defaults to 42.
        early_stopping_rounds: Number of rounds with no improvement to trigger early
            stopping. Defaults to 50.
        use_gpu: Whether to use GPU acceleration if available. Defaults to True.
        
    Returns:
        tuple: A tuple containing:
            best_model: Trained LightGBM classifier with best parameters
            best_params: Dictionary of best hyperparameters found
            X_va: Validation features
            y_va: Validation labels
            hist_df: DataFrame containing trial history with parameters and scores
            plot_paths: List of Paths to hyperparameter visualization plots
    
    Raises:
        ValueError: If no successful trials complete or best parameters are missing
            required keys.
    
    Example:
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=1000, random_state=42)
        >>> model, params, X_val, y_val, history, plots = train_lgbm_optuna(X, y)
        >>> print(f"Best PR-AUC: {history['score'].max():.4f}")
    
    Note:
        Hyperparameters optimized include:
        - num_leaves: [16, 256]
        - max_depth: [10, 20]
        - learning_rate: [0.01, 0.3] log-uniform
        - feature_fraction: [0.6, 1.0]
        - bagging_fraction: [0.6, 1.0]
        - bagging_freq: [1, 10]
        - min_child_samples: [10, 100]
        - lambda_l1: [1e-8, 10.0] log-uniform
        - lambda_l2: [1e-8, 10.0] log-uniform
    """
    # Standard train/val split
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    trial_history = []  # we'll collect params and scores here
    categorical_features = X.select_dtypes(include=['category', 'object']).columns.tolist()
    def objective(trial):
        factor = trial.suggest_float("weight_factor", 100.0, 1000.0)
        params = {
            "objective": "binary",
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
            "scale_pos_weight": _scale_pos_weight(y_tr, factor),
        }
        model = lgb.LGBMClassifier(**params, n_estimators=500, categorical_feature=categorical_features)
        try:
            # 5-fold cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            cv_scores = []
            cv_training_scores = []
            
            for train_idx, val_idx in cv.split(X_tr, y_tr):
                X_train_fold, X_val_fold = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
                y_train_fold, y_val_fold = y_tr.iloc[train_idx], y_tr.iloc[val_idx]

                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    eval_metric=eval_function_lgbm,
                    callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
                )
                
                # Evaluate fold
                y_prob = model.predict_proba(X_val_fold)[:, 1]
                fold_score = eval_function(y_val_fold, y_prob)
                cv_scores.append(fold_score)
                
                # Training score for this fold
                y_prob_train = model.predict_proba(X_train_fold)[:, 1]
                train_score = eval_function(y_train_fold, y_prob_train)
                cv_training_scores.append(train_score)
            
            # Average scores across folds
            score = np.mean(cv_scores)
            train_score = np.mean(cv_training_scores)
            
            # Record trial results including CV stats
            trial_history.append({
                **params,
                "score": score,
                "training_score": train_score,
                "cv_scores": cv_scores,
                "cv_std": np.std(cv_scores),
                "weight_factor": factor
            })
        except Exception as e:
            score = 0.0
            print(f"[WARN] Trial {trial.number} failed with error: {e}")

        return score

    study = optuna.create_study(direction="minimize", study_name="lgbm_custom_cost_optimization")
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
    
    best_model = lgb.LGBMClassifier(**best_params_final, categorical_feature=categorical_features)
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
              "bagging_fraction", "bagging_freq", "min_child_samples", "lambda_l1", "lambda_l2", "weight_factor"]:
        path_maybe = plot_param_vs_score(hist_df, p, plots_dir)
        if path_maybe is not None:
            plot_paths.append(path_maybe)
        path_maybe = plot_param_vs_score(hist_df, p, plots_dir, "training_score")
        if path_maybe is not None:
            plot_paths.append(path_maybe)

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
    """Load the best performing models from MLflow and retrain them on new data.
    
    Searches MLflow for the best XGBoost, CatBoost, and LightGBM models based on
    their performance metric, loads their optimal hyperparameters, and retrains
    them on the provided dataset.
    
    Args:
        X: Feature DataFrame for training/validation/testing.
        y: Target Series with binary labels.
        experiment_name: Name of MLflow experiment to search.
        val_size: Fraction of data for validation. Defaults to 0.2.
        test_size: Optional fraction for test set. If None, uses train/val split only.
        metric: Metric to use for finding best models. Defaults to "pr_auc".
        random_state: Seed for reproducibility. Defaults to 42.
        tracking_uri: MLflow tracking URI. Defaults to "mlruns".
        early_stopping_rounds: Early stopping rounds. Defaults to 50.
        use_gpu: Whether to use GPU acceleration. Defaults to True.
        
    Returns:
        tuple: Contains:
            base_models: Dict mapping model types to trained models
            X_tr: Training features
            X_va: Validation features
            y_tr: Training labels
            y_va: Validation labels
            X_test: Test features (if test_size provided, else None)
            y_test: Test labels (if test_size provided, else None)
            best_params_dict: Dict mapping model types to their best parameters
            
    Raises:
        ValueError: If experiment not found or no models were successfully trained.
        
    Example:
        >>> models, X_tr, X_va, y_tr, y_va, _, _, params = train_best_base_models_from_mlflow(
        ...     X=features_df,
        ...     y=target_series,
        ...     experiment_name="fraud_detection_v1"
        ... )
        >>> print(f"Loaded {len(models)} best models from MLflow")
    """
    
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
            model = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=early_stopping_rounds, enable_categorical=True)
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
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
                categorical_feature=[c for c in X_tr.select_dtypes(include=['category', 'object']).columns] or None
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
    """Train an ensemble model by stacking base models with a logistic regression meta-learner.
    
    Uses Optuna to optimize a logistic regression model that combines predictions from base models
    (XGBoost, CatBoost, LightGBM). The meta-learner's hyperparameters are tuned to maximize PR-AUC.
    
    Args:
        base_models: Dictionary mapping model names to trained base classifiers.
        X_tr: Training feature DataFrame.
        y_tr: Training label Series.
        X_va: Validation feature DataFrame.
        y_va: Validation label Series.
        X_test: Test feature DataFrame or None.
        y_test: Test label Series or None.
        n_trials: Number of Optuna trials for meta-learner optimization. Defaults to 30.
        random_state: Random seed for reproducibility. Defaults to 42.
        
    Returns:
        If X_test is not None:
            tuple containing:
                ensemble: Trained meta-learner model
                base_models: Dictionary of base models
                X_meta_val: Meta-features for validation set
                y_va: Validation labels
                X_meta_test: Meta-features for test set
                y_test: Test labels
                best_params: Meta-learner's best hyperparameters
                hist_df: DataFrame of optimization trial history
                plot_paths: List of Paths to hyperparameter plots
        Else:
            tuple containing first 7 elements listed above
            
    Raises:
        ValueError: If solver-penalty combination is invalid.
        
    Example:
        >>> ensemble, models, X_val, y_val, params, history, plots = train_ensemble(
        ...     base_models={"xgb": xgb_model, "lgbm": lgbm_model},
        ...     X_tr=train_features,
        ...     y_tr=train_labels,
        ...     X_va=val_features,
        ...     y_va=val_labels,
        ...     X_test=None,
        ...     y_test=None
        ... )
        >>> print(f"Ensemble validation PR-AUC: {average_precision_score(y_val, ensemble.predict_proba(X_val)[:, 1]):.4f}")
    
    Note:
        The meta-learner is optimized for:
        - Regularization strength (C)
        - Penalty type (l1, l2, elasticnet)
        - Solver algorithm
        - Class weights
        - L1 ratio (for elasticnet only)
    """
    
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

def train_neural_network_ensemble(
    base_models: dict,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    n_trials: int = 30,
    epochs: int = 100,
    batch_size: int = 32,
    random_state: int = 42,
    device: str = "cpu",
):
    """Train a PyTorch neural network meta-learner on base model predictions.
    
    Uses Optuna to optimize a multi-layer perceptron (MLP) that combines predictions 
    from base models (XGBoost, CatBoost, LightGBM). The meta-learner's architecture 
    and training hyperparameters are tuned using custom loss function (eval_function).
    
    Args:
        base_models: Dictionary mapping model names to trained base classifiers.
        X_tr: Training feature DataFrame.
        y_tr: Training label Series.
        X_va: Validation feature DataFrame.
        y_va: Validation label Series.
        X_test: Optional test feature DataFrame. Defaults to None.
        y_test: Optional test label Series. Defaults to None.
        n_trials: Number of Optuna trials for hyperparameter optimization. Defaults to 30.
        epochs: Number of training epochs per trial. Defaults to 100.
        batch_size: Batch size for training. Defaults to 32.
        random_state: Random seed for reproducibility. Defaults to 42.
        device: Device to use for training ("cpu" or "cuda"). Defaults to "cpu".
        
    Returns:
        If X_test is not None:
            tuple containing:
                best_model: Trained PyTorch neural network
                base_models: Dictionary of base models
                X_meta_val: Meta-features (predictions) for validation set
                y_va: Validation labels
                X_meta_test: Meta-features (predictions) for test set
                y_test: Test labels
                best_params: Neural network's best hyperparameters
                hist_df: DataFrame of optimization trial history
                plot_paths: List of Paths to hyperparameter visualization plots
        Else:
            tuple containing first 7 elements listed above
            
    Raises:
        ImportError: If PyTorch is not installed.
        ValueError: If no successful trials complete or if base_models is empty.
        
    Example:
        >>> model, base_mdls, X_val, y_val, params, history, plots = train_neural_network_ensemble(
        ...     base_models={"xgb": xgb_model, "lgbm": lgbm_model},
        ...     X_tr=train_features,
        ...     y_tr=train_labels,
        ...     X_va=val_features,
        ...     y_va=val_labels,
        ...     X_test=test_features,
        ...     y_test=test_labels,
        ...     n_trials=30,
        ...     epochs=100
        ... )
        >>> print(f"Best validation loss: {history['score'].min():.4f}")
        
    Note:
        - The neural network architecture includes:
          - Input layer: number of base models (typically 3)
          - Hidden layers: 1-3 layers with 16-256 neurons each
          - Output layer: 1 neuron (binary classification)
        - Activation functions: ReLU for hidden layers, Sigmoid for output
        - Optimizer: Adam (learning rate tuned via Optuna)
        - Loss: Binary cross-entropy
        - Custom metric: eval_function (fraud_cost)
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        raise ImportError(
            "PyTorch is required to use train_neural_network_ensemble. "
            "Install it with: pip install torch"
        )
    
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    
    if not base_models:
        raise ValueError("base_models dictionary cannot be empty.")
    
    print("[INFO] Generating meta-features from base models...")
    # Build meta-features (base model predictions)
    X_meta = np.column_stack([m.predict_proba(X_tr)[:, 1] for m in base_models.values()])
    X_meta_val = np.column_stack([m.predict_proba(X_va)[:, 1] for m in base_models.values()])
    
    # Convert to tensors
    X_meta_tensor = torch.FloatTensor(X_meta).to(device)
    y_tr_tensor = torch.FloatTensor(y_tr.values).unsqueeze(1).to(device)
    X_meta_val_tensor = torch.FloatTensor(X_meta_val).to(device)
    y_va_tensor = torch.FloatTensor(y_va.values).unsqueeze(1).to(device)
    
    # Display base model individual validation scores
    for name, model in base_models.items():
        y_pred_val = model.predict_proba(X_va)[:, 1]
        score = eval_function(y_va.values, y_pred_val)
        print(f"[INFO] {name} validation custom_loss: {score:.4f}")
    
    # Define neural network architecture
    class MLPEnsemble(nn.Module):
        def __init__(self, input_size, hidden_sizes, dropout_rate=0.5):
            super(MLPEnsemble, self).__init__()
            layers = []
            prev_size = input_size
            
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                prev_size = hidden_size
            
            # Output layer
            layers.append(nn.Linear(prev_size, 1))
            layers.append(nn.Sigmoid())
            
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)
    
    trial_history = []
    
    def objective(trial):
        # Suggest hyperparameters
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_sizes = []
        for i in range(n_layers):
            hidden_size = trial.suggest_int(f"hidden_size_{i}", 16, 256, log=True)
            hidden_sizes.append(hidden_size)
        
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        
        try:
            # Create model
            model = MLPEnsemble(
                input_size=X_meta.shape[1],
                hidden_sizes=hidden_sizes,
                dropout_rate=dropout_rate
            ).to(device)
            
            # Setup optimizer and loss
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            criterion = nn.BCELoss()
            
            # Create data loader
            train_dataset = TensorDataset(X_meta_tensor, y_tr_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            best_val_score = float('inf')
            patience = 15
            patience_counter = 0
            
            # Training loop
            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * len(X_batch)
                
                train_loss /= len(X_meta)
                
                # Validation
                model.eval()
                with torch.no_grad():
                    y_val_pred_proba = model(X_meta_val_tensor).cpu().numpy().flatten()
                    val_score = eval_function(y_va.values, y_val_pred_proba)
                
                # Early stopping
                if val_score < best_val_score:
                    best_val_score = val_score
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    break
            
            # Record trial results
            trial_history.append({
                "n_layers": n_layers,
                "hidden_sizes": str(hidden_sizes),
                "dropout_rate": dropout_rate,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "score": best_val_score,
                "training_loss": train_loss
            })
            
            return best_val_score
        
        except Exception as e:
            print(f"[WARN] Trial {trial.number} failed with error: {e}")
            return float('inf')
    
    print("[INFO] Optimizing neural network meta-learner with Optuna...")
    study = optuna.create_study(
        direction="minimize",
        study_name="neural_network_ensemble_optimization",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=3)
    )
    
    with tqdm(total=n_trials, desc="[Optuna NN Ensemble Tuning]", unit="trial") as pbar:
        def _tqdm_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            pbar.update(1)
            pbar.set_postfix_str(f"Trial {trial.number+1}/{n_trials} loss={trial.value:.4f}")
        
        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[_tqdm_cb],
            show_progress_bar=False,
            gc_after_trial=True
        )
    
    # Check if any trials succeeded
    if len(trial_history) == 0:
        raise ValueError("No successful trials completed. All Optuna trials failed.")
    
    best_trial = study.best_trial
    best_params = best_trial.params
    print(f"[INFO] Best neural network params: {best_params}")
    
    # Train final model with best params
    n_layers = best_params["n_layers"]
    hidden_sizes = [best_params[f"hidden_size_{i}"] for i in range(n_layers)]
    dropout_rate = best_params["dropout_rate"]
    learning_rate = best_params["learning_rate"]
    weight_decay = best_params["weight_decay"]
    
    best_model = MLPEnsemble(
        input_size=X_meta.shape[1],
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout_rate
    ).to(device)
    
    optimizer = optim.Adam(best_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    train_dataset = TensorDataset(X_meta_tensor, y_tr_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    best_model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = best_model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
    
    # Evaluate on validation set
    best_model.eval()
    with torch.no_grad():
        y_va_pred_proba = best_model(X_meta_val_tensor).cpu().numpy().flatten()
    
    ensemble_custom_loss = eval_function(y_va.values, y_va_pred_proba)
    ensemble_pr_auc = average_precision_score(y_va, y_va_pred_proba)
    ensemble_roc_auc = roc_auc_score(y_va, y_va_pred_proba)
    
    print(f"[INFO] Neural network ensemble validation custom_loss: {ensemble_custom_loss:.4f}")
    print(f"[INFO] Neural network ensemble validation PR-AUC: {ensemble_pr_auc:.4f}")
    print(f"[INFO] Neural network ensemble validation ROC-AUC: {ensemble_roc_auc:.4f}")
    
    X_meta_test = None
    # Evaluate on test set if available
    if X_test is not None:
        test_base_predictions = []
        for name, model in base_models.items():
            y_prob = model.predict_proba(X_test)[:, 1]
            test_base_predictions.append(y_prob)
        X_meta_test = np.column_stack(test_base_predictions)
        X_meta_test_tensor = torch.FloatTensor(X_meta_test).to(device)
        
        with torch.no_grad():
            y_test_pred_proba = best_model(X_meta_test_tensor).cpu().numpy().flatten()
        
        test_custom_loss = eval_function(y_test.values, y_test_pred_proba)
        test_pr_auc = average_precision_score(y_test, y_test_pred_proba)
        test_roc_auc = roc_auc_score(y_test, y_test_pred_proba)
        
        print(f"[INFO] Neural network ensemble test custom_loss: {test_custom_loss:.4f}")
        print(f"[INFO] Neural network ensemble test PR-AUC: {test_pr_auc:.4f}")
        print(f"[INFO] Neural network ensemble test ROC-AUC: {test_roc_auc:.4f}")
    
    # Build history df + plots
    hist_df = pd.DataFrame(trial_history)
    plots_dir = Path("hp_search_plots/nn_ensemble")
    plot_paths = []
    
    for p in ["dropout_rate", "learning_rate", "weight_decay"]:
        path_maybe = plot_param_vs_score(hist_df, p, plots_dir)
        if path_maybe is not None:
            plot_paths.append(path_maybe)
        path_maybe = plot_param_vs_score(hist_df, p, plots_dir, "training_loss")
        if path_maybe is not None:
            plot_paths.append(path_maybe)
    
    if X_test is not None:
        return best_model, base_models, X_meta_val, y_va, X_meta_test, y_test, best_params, hist_df, plot_paths
    else:
        return best_model, base_models, X_meta_val, y_va, best_params, hist_df, plot_paths
