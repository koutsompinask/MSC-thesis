"""
Training pipeline for IEEE-CIS fraud detection using XGBoost with Optuna and GridSearchCV.
- Robust preprocessing for mixed numeric/categorical features
- Handles missing values and class imbalance
- Hyperparameter tuning via Optuna TPE, optional GridSearchCV refinement
- MLflow experiment tracking with metrics, params, artifacts, and model registry-ready output

Designed for reproducibility and clarity to support MSc thesis documentation.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import GridSearchCV

import xgboost as xgb

try:
    import optuna
except Exception:  # pragma: no cover
    optuna = None  # type: ignore

try:
    import mlflow
    import mlflow.sklearn
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore


@dataclass
class TrainConfig:
    data_dir: Path
    seed: int = 42
    test_size: float = 0.2
    n_splits: int = 5
    n_trials: int = 30
    max_time_minutes: Optional[int] = 30
    use_grid_refine: bool = True
    experiment_name: str = "ieee-cis-xgb"
    label_col: str = "isFraud"
    id_col: str = "TransactionID"
    categorical_threshold: int = 20
    results_dir: Path = Path("artifacts")


class FeatureBuilder:
    """Builds a preprocessing pipeline for numeric and categorical columns."""

    def __init__(self, categorical_threshold: int = 20) -> None:
        self.categorical_threshold = categorical_threshold
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.preprocessor: Optional[ColumnTransformer] = None

    def fit(self, df: pd.DataFrame, label_col: str, id_col: Optional[str]) -> "FeatureBuilder":
        feature_cols = [c for c in df.columns if c not in {label_col, id_col}]
        numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        object_like_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]

        categorical_cols: List[str] = []
        for c in object_like_cols:
            unique_count = df[c].nunique(dropna=True)
            if unique_count <= self.categorical_threshold:
                categorical_cols.append(c)

        self.numeric_features = numeric_cols
        self.categorical_features = categorical_cols

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features),
            ],
            remainder="drop",
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None:
            raise RuntimeError("FeatureBuilder is not fitted")
        return self.preprocessor.transform(df)

    def fit_transform(self, df: pd.DataFrame, label_col: str, id_col: Optional[str]) -> np.ndarray:
        return self.fit(df, label_col, id_col).transform(df)


def load_data(data_dir: Path, label_col: str, id_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_identity = pd.read_csv(data_dir / "train_identity.csv")
    train_transaction = pd.read_csv(data_dir / "train_transaction.csv")
    train = pd.merge(train_transaction, train_identity, on=id_col, how="left")
    train.columns = [c.replace('-', '_') for c in train.columns]
    if label_col not in train.columns:
        raise ValueError(f"Label column '{label_col}' not found in training data")
    return train, pd.DataFrame()


def stratified_split(train: pd.DataFrame, label_col: str, seed: int, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, valid_df = train_test_split(
        train,
        test_size=test_size,
        stratify=train[label_col],
        random_state=seed,
    )
    return train_df, valid_df


def objective_factory(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
    n_splits: int,
    seed: int,
):
    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "eval_metric": "aucpr",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-1, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores: List[float] = []
        for train_idx, valid_idx in cv.split(X, y):
            X_tr, X_va = X[train_idx], X[valid_idx]
            y_tr, y_va = y[train_idx], y[valid_idx]
            if sample_weight is not None:
                w_tr = sample_weight[train_idx]
            else:
                w_tr = None

            model = xgb.XGBClassifier(
                n_estimators=1000,
                **params,
                n_jobs=max(1, os.cpu_count() or 1),
                random_state=seed,
            )
            model.fit(
                X_tr,
                y_tr,
                sample_weight=w_tr,
                eval_set=[(X_va, y_va)],
                early_stopping_rounds=100,
                verbose=False,
            )
            preds = model.predict_proba(X_va)[:, 1]
            score = average_precision_score(y_va, preds)
            scores.append(score)

        return float(np.mean(scores))

    return objective


def grid_refine(
    X: np.ndarray,
    y: np.ndarray,
    base_params: Dict[str, object],
    seed: int,
    sample_weight: Optional[np.ndarray],
) -> Dict[str, object]:
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_estimators=1200,
        n_jobs=max(1, os.cpu_count() or 1),
        random_state=seed,
        **base_params,
    )
    param_grid = {
        "learning_rate": [max(0.01, float(base_params.get("learning_rate", 0.1)) * f) for f in [0.5, 1.0, 1.5]],
        "max_depth": [max(3, int(base_params.get("max_depth", 6)) + d) for d in [-1, 0, 1]],
        "subsample": [min(1.0, max(0.5, float(base_params.get("subsample", 0.8)) + d)) for d in [-0.1, 0.0, 0.1]],
        "colsample_bytree": [min(1.0, max(0.5, float(base_params.get("colsample_bytree", 0.8)) + d)) for d in [-0.1, 0.0, 0.1]],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="average_precision",
        n_jobs=1,
        verbose=0,
    )
    fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
    gs.fit(X, y, **fit_kwargs)
    refined = base_params.copy()
    refined.update(gs.best_params_)
    return refined


def fit_best_model(
    X: np.ndarray,
    y: np.ndarray,
    best_params: Dict[str, object],
    seed: int,
    sample_weight: Optional[np.ndarray],
) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_estimators=3000,
        **best_params,
        n_jobs=max(1, os.cpu_count() or 1),
        random_state=seed,
    )
    model.fit(X, y, sample_weight=sample_weight, verbose=False)
    return model


def evaluate(model: xgb.XGBClassifier, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    probs = model.predict_proba(X)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y, probs)),
        "average_precision": float(average_precision_score(y, probs)),
    }


def run_training(cfg: TrainConfig) -> None:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    train_df, _ = load_data(cfg.data_dir, cfg.label_col, cfg.id_col)
    train_part, valid_part = stratified_split(train_df, cfg.label_col, cfg.seed, cfg.test_size)

    builder = FeatureBuilder(categorical_threshold=cfg.categorical_threshold)
    X_train = builder.fit_transform(train_part, label_col=cfg.label_col, id_col=cfg.id_col)
    y_train = train_part[cfg.label_col].values.astype(int)
    X_valid = builder.transform(valid_part)
    y_valid = valid_part[cfg.label_col].values.astype(int)

    sample_weight_train = compute_sample_weight(class_weight="balanced", y=y_train)
    sample_weight_valid = compute_sample_weight(class_weight="balanced", y=y_valid)

    if optuna is None:
        raise RuntimeError("optuna is not installed. Please install requirements-ml.txt")

    study = optuna.create_study(
        direction="maximize",
        study_name=f"xgb_optuna_{int(time.time())}",
    )
    objective = objective_factory(
        X=X_train,
        y=y_train,
        sample_weight=sample_weight_train,
        n_splits=cfg.n_splits,
        seed=cfg.seed,
    )

    timeout = None
    if cfg.max_time_minutes is not None:
        timeout = int(cfg.max_time_minutes * 60)

    study.optimize(objective, n_trials=cfg.n_trials, timeout=timeout, show_progress_bar=False)
    best_params = study.best_trial.params

    if cfg.use_grid_refine:
        best_params = grid_refine(
            X=X_train,
            y=y_train,
            base_params=best_params,
            seed=cfg.seed,
            sample_weight=sample_weight_train,
        )

    model = fit_best_model(
        X=np.vstack([X_train, X_valid]),
        y=np.concatenate([y_train, y_valid]),
        best_params=best_params,
        seed=cfg.seed,
        sample_weight=np.concatenate([sample_weight_train, sample_weight_valid]),
    )

    model_split = fit_best_model(
        X=X_train,
        y=y_train,
        best_params=best_params,
        seed=cfg.seed,
        sample_weight=sample_weight_train,
    )
    metrics_train = evaluate(model_split, X_train, y_train)
    metrics_valid = evaluate(model_split, X_valid, y_valid)

    art_dir = cfg.results_dir / f"run_{int(time.time())}"
    art_dir.mkdir(parents=True, exist_ok=True)
    with open(art_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    with open(art_dir / "metrics_train.json", "w", encoding="utf-8") as f:
        json.dump(metrics_train, f, indent=2)
    with open(art_dir / "metrics_valid.json", "w", encoding="utf-8") as f:
        json.dump(metrics_valid, f, indent=2)

    if mlflow is not None:
        mlflow.set_experiment(cfg.experiment_name)
        with mlflow.start_run(run_name=f"xgb_optuna_{cfg.n_trials}t_{cfg.n_splits}cv"):
            mlflow.log_params(best_params)
            mlflow.log_params({
                "seed": cfg.seed,
                "n_splits": cfg.n_splits,
                "n_trials": cfg.n_trials,
                "test_size": cfg.test_size,
                "categorical_threshold": cfg.categorical_threshold,
                "grid_refine": cfg.use_grid_refine,
            })
            for k, v in metrics_train.items():
                mlflow.log_metric(f"train_{k}", v)
            for k, v in metrics_valid.items():
                mlflow.log_metric(f"valid_{k}", v)
            mlflow.log_artifacts(str(art_dir))
            mlflow.sklearn.log_model(model, artifact_path="model")

    print("Training complete. Key metrics (split model):")
    print("Train:", metrics_train)
    print("Valid:", metrics_valid)


def parse_args(argv: Optional[List[str]] = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train XGBoost model for IEEE-CIS fraud detection")
    parser.add_argument("--data-dir", type=str, default="ieee-fraud-detection-data", help="Path to data directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--max-time-minutes", type=int, default=30)
    parser.add_argument("--no-grid-refine", action="store_true")
    parser.add_argument("--experiment-name", type=str, default="ieee-cis-xgb")
    parser.add_argument("--results-dir", type=str, default="artifacts")
    args = parser.parse_args(argv)
    cfg = TrainConfig(
        data_dir=Path(args.data_dir),
        seed=args.seed,
        test_size=args.test_size,
        n_splits=args.n_splits,
        n_trials=args.n_trials,
        max_time_minutes=args.max_time_minutes,
        use_grid_refine=not args.no_grid_refine,
        experiment_name=args.experiment_name,
        results_dir=Path(args.results_dir),
    )
    return cfg


if __name__ == "__main__":
    config = parse_args()
    run_training(config)
