import numpy as np
import pandas as pd


def prepare_lightgbm_input(model, data: pd.DataFrame) -> pd.DataFrame:
    """Match the dataframe schema LightGBM saw during training."""
    prepared = data.reindex(columns=model.feature_name_, fill_value=np.nan).copy()

    categorical_features = getattr(model, "categorical_feature", None) or []
    pandas_categories = getattr(model.booster_, "pandas_categorical", None) or []
    categorical_set = set(categorical_features)

    for column, categories in zip(categorical_features, pandas_categories):
        if column not in prepared.columns:
            continue

        values = prepared[column]
        if "missing" in categories:
            values = values.fillna("missing")

        prepared[column] = pd.Categorical(values, categories=categories)

    for column in prepared.columns:
        if column in categorical_set:
            continue
        if prepared[column].dtype == "object":
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    return prepared


def json_safe_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value
