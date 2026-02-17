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
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier, Pool, cv
import xgboost as xgb
import lightgbm as lgb
from tqdm.auto import tqdm

def _gpu_available() -> bool:
    """Check whether GPU training is available for XGBoost.

    Returns:
        bool: `True` if a tiny GPU-based XGBoost fit succeeds, otherwise `False`.
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
    """Compute the imbalance ratio used as positive-class weight.

    Args:
        y: Binary labels.
        factor: Optional multiplier applied to the imbalance ratio.

    Returns:
        float: `factor * (n_negative / max(1, n_positive))`.
    """
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    return factor*float(neg / max(1, pos))

def train_baseline_models(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> dict[str, object]:
    """Train baseline classifiers used for comparison.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        random_state: Random seed used by baseline estimators.

    Returns:
        dict[str, object]: Mapping of model name to fitted estimator.
    """
    models = {
        "log_reg": LogisticRegression(
            max_iter=2000,
            n_jobs=-1,
            class_weight="balanced",
            random_state=random_state,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=100,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=50,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=random_state,
        ),
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained

def eval_function(y_true, y_prob, threshold: float = 0.5) -> float:
    """Compute the custom fraud loss from probabilities.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted positive-class probabilities.
        threshold: Probability threshold for converting to class predictions.

    Returns:
        float: `(100 * FN + FP) / n_samples`.
    """
    fp = np.sum((y_true == 0) & (y_prob >= threshold))
    fn = np.sum((y_true == 1) & (y_prob < threshold))
    return (100*fn + fp)/len(y_true)

def minimize_eval_metric_with_threshold(model, X, y_true):
    """Find the best threshold under `eval_function`.

    Args:
        model: Fitted classifier with `predict_proba` or `predict`.
        X: Feature matrix for inference.
        y_true: Ground-truth binary labels.

    Returns:
        tuple[float, float]: Best threshold and corresponding minimum custom loss.
    """
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = model.predict(X)

    # force pandas Series for consistent iloc use
    y_prob = pd.Series(y_prob).reset_index(drop=True)
    y_true = pd.Series(y_true).reset_index(drop=True)

    minProb = 0.5
    minScore = eval_function(y_true, y_prob)

    for i in range(len(y_true)):
        if y_prob.iloc[i] < minProb and y_true.iloc[i] == 1:
            score = eval_function(y_true, y_prob, y_prob.iloc[i])
            if score < minScore:
                minProb = y_prob.iloc[i]
                minScore = score

    return minProb, minScore

def plot_param_vs_score(df: pd.DataFrame, param_name: str, out_dir: Path, metric_name: str = "score"):
    """Plot a hyperparameter against an optimization metric.

    Args:
        df: Trial history DataFrame containing parameter and metric columns.
        param_name: Parameter column to plot.
        out_dir: Directory where the PNG plot is saved.
        metric_name: Metric column to plot on the y-axis.

    Returns:
        Path | None: Saved plot path, or `None` if `param_name` is not in `df`.
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
def train_xgb_optuna(X, y, n_trials=30, random_state=42,
                     early_stopping_rounds=50, use_gpu=True):
    """Train an XGBoost classifier with Optuna tuning.

    Args:
        X: Training features.
        y: Training labels.
        n_trials: Number of Optuna trials.
        random_state: Random seed.
        early_stopping_rounds: Early stopping rounds for fold training.
        use_gpu: Whether GPU should be used when available.

    Returns:
        tuple: `(best_model, best_params, hist_df, plot_paths)`.

    Raises:
        ValueError: If all Optuna trials fail.
    """
    n = len(X)
    split_idx1 = int(n * 0.8)

    X_train = X.iloc[:split_idx1]
    X_valid = X.iloc[split_idx1:]
    y_train = y.iloc[:split_idx1]
    y_valid = y.iloc[split_idx1:]

    trial_history = []  # we'll collect params and scores here
    
    def objective(trial):
        params = {
            "n_estimators": 300,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.03),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 1, log=True),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "tree_method": "gpu_hist" if  _gpu_available() and use_gpu else "hist",
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "scale_pos_weight": _scale_pos_weight(y),
            "n_jobs": -1,
            "random_state": random_state,
        }
        try:
            # 5-fold cross-validation
            cv = TimeSeriesSplit(n_splits=3)
            cv_scores = []
            cv_training_scores = []
            
            for train_idx, val_idx in cv.split(X, y):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                model = xgb.XGBClassifier(**params, early_stopping_rounds=early_stopping_rounds, enable_categorical=True)
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    verbose=False
                )
                
                # Evaluate fold
                y_prob = model.predict_proba(X_val_fold)[:, 1]
                fold_score = roc_auc_score(y_val_fold, y_prob)
                cv_scores.append(fold_score)
                
                # Training score for this fold
                y_prob_train = model.predict_proba(X_train_fold)[:, 1]
                train_score = roc_auc_score(y_train_fold, y_prob_train)
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
            })
        except Exception as e:
            score = 0.0
            print(f"[WARN] Trial {trial.number} failed with error: {e}")
        return score

    study = optuna.create_study(direction="maximize", study_name="xgboost_aucpr_optimization", pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=5))
    with tqdm(total=n_trials, desc="[Optuna XGBoost Tuning]", unit="trial") as pbar:
        def _tqdm_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            pbar.update(1)
            pbar.set_postfix_str(f"Trial {trial.number+1}/{n_trials} aucpr={trial.value:.4f}")
        study.optimize(objective, n_trials=n_trials, callbacks=[_tqdm_cb], show_progress_bar=False, gc_after_trial=True)
    
    # Check if any trials succeeded
    if len(trial_history) == 0:
        raise ValueError("No successful trials completed. All Optuna trials failed. Check your data and parameters.")
    
    best_params = study.best_params
    print(f"[INFO] Best XGBoost params: {best_params}")

    # Retrain with best params: use n_estimators=1000 for final model (increased from 500 used in trials)
    # This allows the model to potentially improve further with more trees
    best_params_final = best_params.copy()
    best_params_final["n_estimators"] = 5000
    best_params_final["tree_method"] = "gpu_hist" if _gpu_available() and use_gpu else "hist"
    best_params_final["objective"] = "binary:logistic"
    best_params_final["scale_pos_weight"] = _scale_pos_weight(y_train)
    best_params_final["n_jobs"] = -1
    best_params_final["random_state"] = random_state
    best_params_final["verbosity"] = 0
    best_params_final["eval_metric"] = "auc"

    best_model = xgb.XGBClassifier(**best_params_final, early_stopping_rounds=100, enable_categorical=True)
    best_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=100)
    

    # Update best_params to include n_estimators for logging consistency
    best_params["n_estimators"] = 5000

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

    return best_model, best_params, hist_df, plot_paths

def train_catboost_optuna(X, y, n_trials=30, random_state=42,
                          early_stopping_rounds=50, use_gpu=True):
    """Train a CatBoost classifier with Optuna tuning.

    Args:
        X: Training features.
        y: Training labels.
        n_trials: Number of Optuna trials.
        random_state: Random seed.
        early_stopping_rounds: Early stopping rounds for fold training.
        use_gpu: Whether GPU should be used when available.

    Returns:
        tuple: `(best_model, best_params, hist_df, plot_paths)`.

    Raises:
        ValueError: If all Optuna trials fail.
    """
    trial_history = []  # we'll collect params and scores here
    n = len(X)
    split_idx1 = int(n * 0.8)

    X_train = X.iloc[:split_idx1]
    X_valid = X.iloc[split_idx1:]
    y_train = y.iloc[:split_idx1]
    y_valid = y.iloc[split_idx1:]

    categorical_features = X.select_dtypes(include=['category', 'object']).columns.tolist()

    # train_pool = Pool(X, y, cat_features=categorical_features)

    def objective(trial):
        params = {
            "iterations": 300,
            "bootstrap_type": "Bernoulli",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.03),
            "depth": trial.suggest_int("depth", 3, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 128),
            "eval_metric": "AUC",
            "task_type": "GPU" if  _gpu_available() and use_gpu else "CPU",
            "loss_function": "Logloss",
            "random_state": random_state,
            "scale_pos_weight": _scale_pos_weight(y),
        }
        try:
            # 5-fold cross-validation
            cv = TimeSeriesSplit(n_splits=3)
            cv_scores = []
            cv_training_scores = []
            
            for train_idx, val_idx in cv.split(X, y):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                model = CatBoostClassifier(**params, cat_features=categorical_features)

                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False
                )
                # Evaluate fold
                y_prob = model.predict_proba(X_val_fold)[:, 1]
                fold_score = roc_auc_score(y_val_fold, y_prob)
                cv_scores.append(fold_score)
                
                # Training score for this fold
                y_prob_train = model.predict_proba(X_train_fold)[:, 1]
                train_score = roc_auc_score(y_train_fold, y_prob_train)
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
                "cv_std": np.std(cv_scores)
            })
        except Exception as e:
            score = 0.0
            print(f"[WARN] Trial {trial.number} failed with error: {e}")
        return score

    study = optuna.create_study(direction="maximize", study_name="catboost_aucpr_optimization", pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=5), )
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

    # Retrain with best params: use iterations=1000 for final model (increased from 500 used in trials)
    # This allows the model to potentially improve further with more iterations
    best_params_final = best_params.copy()
    best_params_final["iterations"] = 5000
    best_params_final["eval_metric"] = "AUC"
    best_params_final["task_type"] = "GPU" if _gpu_available() and use_gpu else "CPU"
    best_params_final["random_seed"] = random_state
    best_params_final["scale_pos_weight"] = _scale_pos_weight(y_train)
    best_params_final["verbose"] = False
    
    best_model = CatBoostClassifier(**best_params_final)
    best_model.fit(X_train, y_train, eval_set=(X_valid, y_valid),
                   early_stopping_rounds=100,
                   verbose=100,
                   cat_features=categorical_features)
    
    # Update best_params to include iterations for logging consistency
    best_params["iterations"] = 5000
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
    
    return best_model, best_params, hist_df, plot_paths

def train_lgbm_optuna(X, y, n_trials=30, random_state=42,
                      early_stopping_rounds=50, use_gpu=True):
    """Train a LightGBM classifier with Optuna tuning.

    Args:
        X: Training features.
        y: Training labels.
        n_trials: Number of Optuna trials.
        random_state: Random seed.
        early_stopping_rounds: Early stopping rounds for fold training.
        use_gpu: Whether GPU should be used when available.

    Returns:
        tuple: `(best_model, best_params, hist_df, plot_paths)`.

    Raises:
        ValueError: If all Optuna trials fail.
    """
    trial_history = []  # we'll collect params and scores here
    n = len(X)
    split_idx1 = int(n * 0.8)

    X_train = X.iloc[:split_idx1]
    X_valid = X.iloc[split_idx1:]
    y_train = y.iloc[:split_idx1]
    y_valid = y.iloc[split_idx1:]
    
    categorical_features = X.select_dtypes(include=['category', 'object']).columns.tolist()
    def objective(trial):
        params = {
            "n_estimators": 300,
            "objective": "binary",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 128, 512),
            "max_depth": trial.suggest_int("max_depth", 10, 20),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.03),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "device": "gpu" if  _gpu_available() and use_gpu else "cpu",
            "random_state": random_state,
            "scale_pos_weight": _scale_pos_weight(y),
            'reg_alpha': trial.suggest_float("reg_alpha", 0.0, 0.5),
            'reg_lambda': trial.suggest_float("reg_lambda", 0.0, 0.5),
            'colsample_bytree': trial.suggest_float("colsample_bytree", 0.8, 1),
            "subsample": trial.suggest_float("subsample", 0.8, 1)
        }
        model = lgb.LGBMClassifier(**params, categorical_feature=categorical_features)
        try:
            # 5-fold cross-validation
            cv = TimeSeriesSplit(n_splits=3)
            cv_scores = []
            cv_training_scores = []
            
            for train_idx, val_idx in cv.split(X, y):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    eval_metric="auc",
                    callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
                )
                
                # Evaluate fold
                y_prob = model.predict_proba(X_val_fold)[:, 1]
                fold_score = roc_auc_score(y_val_fold, y_prob)
                cv_scores.append(fold_score)
                
                # Training score for this fold
                y_prob_train = model.predict_proba(X_train_fold)[:, 1]
                train_score = roc_auc_score(y_train_fold, y_prob_train)
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
                "cv_std": np.std(cv_scores)
            })
        except Exception as e:
            score = 0.0
            print(f"[WARN] Trial {trial.number} failed with error: {e}")

        return score

    study = optuna.create_study(direction="maximize", study_name="lgbm_aucpr_optimization")
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

    # Retrain with best params: use n_estimators=1000 for final model (increased from 500 used in trials)
    # This allows the model to potentially improve further with more trees
    best_params_final = best_params.copy()
    best_params_final["n_estimators"] = 5000
    best_params_final["objective"] = "binary"
    best_params_final["verbosity"] = -1
    best_params_final["boosting_type"] = "gbdt"
    best_params_final["device"] = "gpu" if _gpu_available() and use_gpu else "cpu"
    best_params_final["random_state"] = random_state
    best_params_final["scale_pos_weight"] = _scale_pos_weight(y_train)
    best_params_final["eval_metric"] = "auc"
    
    best_model = lgb.LGBMClassifier(**best_params_final, categorical_feature=categorical_features)
    best_model.fit(X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    eval_metric="auc",
                    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=50)])
    
    # Update best_params to include n_estimators for logging consistency
    best_params["n_estimators"] = 5000
    
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

    return best_model, best_params, hist_df, plot_paths

def train_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 10,
    random_state: int = 42,
):
    """Train a logistic-regression ensemble model with Optuna.

    Args:
        X: Training features.
        y: Training labels.
        n_trials: Number of Optuna trials for meta-learner tuning.
        random_state: Random seed.

    Returns:
        tuple: `(ensemble, best_params_log, hist_df, plot_paths)`.
    """
    
    print("[INFO] Generating meta-features...")
    # Prepare Optuna optimization
    trial_history = []
    n = len(X)
    split_idx1 = int(n * 0.8)

    X_train = X.iloc[:split_idx1]
    X_valid = X.iloc[split_idx1:]
    y_train = y.iloc[:split_idx1]
    y_valid = y.iloc[split_idx1:]

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
            # 5-fold cross-validation
            cv = TimeSeriesSplit(n_splits=3)
            cv_scores = []
            cv_training_scores = []
            
            for train_idx, val_idx in cv.split(X, y):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                    
                model = LogisticRegression(**params)
                model.fit(X_train_fold, y_train_fold)
                 
                # Evaluate fold
                y_prob = model.predict_proba(X_val_fold)[:, 1]
                fold_score = roc_auc_score(y_val_fold, y_prob)
                cv_scores.append(fold_score)
                
                # Training score for this fold
                y_prob_train = model.predict_proba(X_train_fold)[:, 1]
                train_score = roc_auc_score(y_train_fold, y_prob_train)
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
            })
        except Exception as e:
            print(f"[WARN] Trial {trial.number} failed ({e})")
            score = 0.0
        return score

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
    ensemble.fit(X_train, y_train)
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
    y_prob_ensemble = ensemble.predict_proba(X_valid)[:, 1]
    ensemble_pr_auc = average_precision_score(y_valid, y_prob_ensemble)
    ensemble_roc_auc = roc_auc_score(y_valid, y_prob_ensemble)
    
    print(f"[INFO] Ensemble validation PR-AUC: {ensemble_pr_auc:.4f}")
    print(f"[INFO] Ensemble validation ROC-AUC: {ensemble_roc_auc:.4f}")
    
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

    return ensemble, best_params_log, hist_df, plot_paths
