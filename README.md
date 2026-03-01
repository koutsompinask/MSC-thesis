# Fraud Detection MSc Thesis (IEEE-CIS)

This repository contains experiments and tooling for fraud detection on card transactions using the IEEE-CIS Fraud Detection dataset.

It includes:
- EDA and preprocessing notebooks
- model training and evaluation utilities
- Optuna-based hyperparameter search
- MLflow tracking and model logging
- a FastAPI inference service for a logged model

## Dataset

- Competition page: https://www.kaggle.com/competitions/ieee-fraud-detection/data
- Local dataset folder in this repo: `ieee-fraud-detection-data/`
- Detailed dataset notes: `Detailed-Description.md`

## Repository Structure

- `EDA_and_preprocessing.ipynb`: exploratory analysis and feature engineering workflow
- `training.ipynb`: experiment/training notebook
- `train_models_util.py`: reusable model training helpers (XGBoost, CatBoost, LightGBM, baselines)
- `evaluate_models_util.py`: evaluation metrics, plots, threshold selection, MLflow logging
- `feature_importance.py`: SHAP-based global and case-level explainability helpers
- `cleanup.py`: utility to delete MLflow runs marked as deleted
- `mlruns/`: MLflow tracking artifacts
- `fastapi/`: API service code (`main.py`, request schema, config)

## Environment Setup

1. Create and activate a Python environment (recommended: Python 3.10+).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Typical Workflow

1. Place/unzip the IEEE-CIS data under `ieee-fraud-detection-data/`.
2. Run `EDA_and_preprocessing.ipynb` to prepare features.
3. Run `training.ipynb` for model training and experiment tracking.
4. Use utilities from:
   - `train_models_util.py` for model fitting/tuning
   - `evaluate_models_util.py` for model evaluation and artifact generation
   - `feature_importance.py` for SHAP explainability outputs

## MLflow

Start the MLflow UI from the project root:

```bash
mlflow ui
```

Then open the local URL shown in terminal (commonly `http://127.0.0.1:5000`).

## FastAPI Inference Service

The API is under `fastapi/` and loads a model from an MLflow model path configured in `fastapi/main.py`.

1. Set/update API key in `fastapi/config.py`.
2. Verify `MODEL_PATH` in `fastapi/main.py` points to an existing local MLflow model.
3. Start the service from project root:

```bash
uvicorn fastapi.main:app --reload --host 0.0.0.0 --port 8000
```

4. Call prediction endpoint:
- `POST /predict`
- header: `X-API-Key: <your_key>`
- body: JSON matching `fastapi/model.py` (`PredictionRequest`)

## Notes

- This repository tracks many generated artifacts (`mlruns/`, search plots, etc.).
- `cleanup.py` can remove run folders whose `meta.yaml` has `lifecycle_stage: deleted`.
