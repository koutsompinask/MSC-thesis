import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import mlflow as mlf
import mlflow.xgboost as mlf_xgboost, mlflow.catboost as mlf_catboost, mlflow.lightgbm as mlf_lightgbm, mlflow.sklearn as mlf_sklearn
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    auc,
    precision_recall_fscore_support,
    precision_recall_curve,
    confusion_matrix,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
import tempfile
from train_models_util import eval_function

def _plot_artifacts(y_true: np.ndarray, y_prob: np.ndarray, out_dir: Path, threshold: float = 0.5) -> dict[str, Path]:
    """Generate and save evaluation plots for binary classification results.
    
    Creates and saves four key evaluation plots:
    - ROC curve showing true positive vs false positive rate trade-off
    - Precision-Recall curve showing precision vs recall trade-off
    - Confusion matrix showing prediction counts at specified threshold
    - FP/FN over time showing cumulative false positive and negative rates across samples
    
    Args:
        y_true: True binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class (between 0 and 1).
        out_dir: Directory path where plots will be saved.
        threshold: Classification threshold for converting probabilities to binary predictions.
            Used for confusion matrix and FP/FN analysis. Defaults to 0.5.
        
    Returns:
        dict[str, Path]: Dictionary mapping plot names to their saved file paths.
            Keys: 
                'roc_curve': Path to ROC curve plot
                'pr_curve': Path to precision-recall curve plot
                'confusion_matrix': Path to confusion matrix plot
                'fp_fn_over_time': Path to FP/FN rates over time plot
            
    Example:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_prob = np.array([0.1, 0.9, 0.8, 0.3])
        >>> paths = _plot_artifacts(y_true, y_prob, Path("evaluation_plots"))
        >>> print(f"ROC curve saved to: {paths['roc_curve']}")
    
    Note:
        All plots are saved as PNG files with 150 DPI resolution.
        The plots include relevant metrics like AUC-ROC and AP scores.
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

    # Confusion matrix @ threshold
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix @ {threshold}")
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

    # FP/FN over time
    y_pred_binary = (y_prob >= threshold).astype(int)
    fp = (y_pred_binary == 1) & (y_true == 0)
    fn = (y_pred_binary == 0) & (y_true == 1)

    cum_fp = np.cumsum(fp)
    cum_fn = np.cumsum(fn)
    cum_neg = np.cumsum(y_true == 0)
    cum_pos = np.cumsum(y_true == 1)

    fpr = np.where(cum_neg == 0, 0, cum_fp / cum_neg)
    fnr = np.where(cum_pos == 0, 0, cum_fn / cum_pos)

    plt.figure(figsize=(12, 6))
    plt.plot(fpr, label="False Positive Rate (FPR)")
    plt.plot(fnr, label="False Negative Rate (FNR)")
    plt.xlabel("Transaction Index (Time)")
    plt.ylabel("Rate")
    plt.title("False Positive & False Negative Rates Over Time")
    plt.legend()
    plt.grid(True)
    p = out_dir / "fp_fn_over_time.png"
    plt.savefig(p, bbox_inches="tight", dpi=150)
    plt.close()
    paths["fp_fn_over_time"] = p

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
    prediction_threshold: float = 0.5,
):
    """Evaluate a binary classifier and log comprehensive results to MLflow.
    
    Performs extensive model evaluation including:
    - Threshold-independent metrics (ROC-AUC, PR-AUC on probabilities)
    - Threshold-dependent metrics (F1, precision, recall using custom threshold)
    - SHAP feature importance analysis
    - Calibration assessment
    - Performance visualization
    
    Automatically detects model type and logs everything to MLflow for tracking.
    
    Args:
        model: Any binary classifier supporting predict_proba (XGBoost, CatBoost,
            LightGBM, or scikit-learn style).
        X_va: Validation features DataFrame.
        y_va: Validation target Series.
        experiment_name: Name of the MLflow experiment for logging.
        run_name: Name for this specific MLflow run.
        model_type: Optional model type string for tagging. If None, auto-detects.
            Valid values: "xgboost", "catboost", "lightgbm", "sklearn".
        best_params: Optional dictionary of best hyperparameters to log.
        tracking_uri: MLflow tracking URI. Defaults to "mlruns".
        hp_search_history: Optional DataFrame with hyperparameter optimization history.
        hp_search_plots: Optional list of hyperparameter visualization plot paths.
        prediction_threshold: Classification probability threshold for converting probabilities
            to binary predictions. Affects F1, precision, recall, custom_loss metrics, and plots.
            Defaults to 0.5. ROC-AUC and PR-AUC are always computed on probabilities regardless
            of this threshold.
        
    Returns:
        dict: Performance metrics including:
            - roc_auc: Area under ROC curve (threshold-independent, on probabilities)
            - pr_auc: Area under precision-recall curve (threshold-independent, on probabilities)
            - precision: Precision at specified threshold
            - recall: Recall at specified threshold
            - f1: F1 score at specified threshold
            - custom_loss: Custom fraud cost metric at specified threshold
            
    Example:
        >>> model = XGBClassifier()
        >>> model.fit(X_train, y_train)
        >>> metrics = evaluate_and_log(
        ...     model=model,
        ...     X_va=X_val,
        ...     y_va=y_val,
        ...     experiment_name="fraud_detection",
        ...     run_name="xgboost_v1",
        ...     prediction_threshold=0.3
        ... )
        >>> print(f"PR-AUC: {metrics['pr_auc']:.4f}")
        >>> print(f"Custom loss @ threshold=0.3: {metrics['custom_loss']:.4f}")
        
    Note:
        - SHAP analysis is performed on a balanced subset of validation data
          (up to 500 samples per class) to keep computation tractable.
        - ROC-AUC and PR-AUC are always computed on predicted probabilities and are 
          threshold-independent. Only precision, recall, F1, and custom_loss depend on the threshold.
        - All plots use the specified prediction_threshold for threshold-dependent visualizations.
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

    if prediction_threshold != 0.5: 
        run_name = f"{run_name}_threshold_{prediction_threshold}"

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
        y_pred = (y_prob >= prediction_threshold).astype(int)

        # --- Metrics ---
        # ROC-AUC and PR-AUC are computed on probabilities (threshold-independent)
        roc_auc = roc_auc_score(y_va, y_prob)
        pr_auc = average_precision_score(y_va, y_prob)
        # Custom loss and classification metrics use the specified threshold
        custom_loss = eval_function(y_va, y_prob, threshold=prediction_threshold)
        precision, recall, f1, _ = precision_recall_fscore_support(y_va, y_pred, average="binary")
        metrics = {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "custom_loss": custom_loss
        }
        mlf.log_metrics(metrics)
        print(f"[INFO] Logged metrics: {metrics}")
        artifacts_dir = Path(td)
        plot_paths = _plot_artifacts(y_va, y_prob, artifacts_dir / "plots", threshold=prediction_threshold)
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


        # --- Log Model ---
        try:
            # Prefer generic API when available
            mlf.log_model(model, name="model")
        except Exception:
            # Fallback to flavor-specific APIs
            try:
                if model_type == "xgboost":
                    mlf_xgboost.log_model(model, name="model")
                elif model_type == "catboost":
                    mlf_catboost.log_model(model, name="model")
                elif model_type == "lightgbm":
                    mlf_lightgbm.log_model(model, name="model")
                else:
                    mlf_sklearn.log_model(model, name="model")
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