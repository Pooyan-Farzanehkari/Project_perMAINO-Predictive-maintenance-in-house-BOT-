"""Functions for scaling / normalizing numeric columns."""

from __future__ import annotations

import pandas as pd


def _numeric_columns(df: pd.DataFrame, columns: list[str] | None) -> list[str]:
    return columns if columns else df.select_dtypes(include="number").columns.tolist()


def min_max_scale(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    feature_min: float = 0.0,
    feature_max: float = 1.0,
) -> pd.DataFrame:
    """Rescale columns to [feature_min, feature_max] (all numeric columns if omitted)."""
    df = df.copy()
    for col in _numeric_columns(df, columns):
        col_min, col_max = df[col].min(), df[col].max()
        if col_max == col_min:
            df[col] = feature_min
        else:
            df[col] = (df[col] - col_min) / (col_max - col_min) * (feature_max - feature_min) + feature_min
    return df


def z_score_scale(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Standardize columns to zero mean and unit variance (all numeric columns if omitted)."""
    df = df.copy()
    for col in _numeric_columns(df, columns):
        mean, std = df[col].mean(), df[col].std()
        df[col] = 0.0 if std == 0 else (df[col] - mean) / std
    return df


def robust_scale(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Scale columns using median and IQR, robust to outliers (all numeric columns if omitted)."""
    df = df.copy()
    for col in _numeric_columns(df, columns):
        median = df[col].median()
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = 0.0 if iqr == 0 else (df[col] - median) / iqr
    return df
