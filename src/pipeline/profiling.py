"""Build a compact, JSON-serializable summary of a dataset.

This is what gets shown to the LLM instead of raw rows: shape, dtypes,
missing-value counts, and basic numeric stats per column. Small and cheap
regardless of how many rows the underlying DataFrame has.
"""

from __future__ import annotations

import pandas as pd


def profile_dataset(df: pd.DataFrame, max_categorical_unique: int = 20) -> dict:
    profile: dict = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": {},
    }

    if df.index.name and len(df):
        profile["index"] = {
            "name": df.index.name,
            "dtype": str(df.index.dtype),
            "start": str(df.index.min()),
            "end": str(df.index.max()),
        }

    for col in df.columns:
        series = df[col]
        has_data = series.notna().any()
        info: dict = {
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_pct": round(float(series.isna().mean() * 100), 4),
        }

        if pd.api.types.is_numeric_dtype(series):
            info.update(
                {
                    "min": float(series.min()) if has_data else None,
                    "max": float(series.max()) if has_data else None,
                    "mean": float(series.mean()) if has_data else None,
                    "std": float(series.std()) if has_data else None,
                    "unique_count": int(series.nunique()),
                }
            )
        else:
            info["unique_count"] = int(series.nunique())
            if series.nunique() <= max_categorical_unique:
                info["unique_values"] = series.dropna().unique().tolist()

        profile["columns"][col] = info

    return profile
