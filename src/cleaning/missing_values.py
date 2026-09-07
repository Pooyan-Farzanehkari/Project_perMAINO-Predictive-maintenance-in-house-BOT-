"""Functions for detecting and handling missing values."""

from __future__ import annotations

import pandas as pd


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column missing counts and percentages."""
    counts = df.isna().sum()
    pct = (df.isna().mean() * 100).round(4)
    return pd.DataFrame({"missing_count": counts, "missing_pct": pct})


def drop_missing_rows(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    how: str = "any",
) -> pd.DataFrame:
    """Drop rows with missing values in the given columns (all columns if omitted)."""
    return df.dropna(subset=columns, how=how)


def fill_missing(
    df: pd.DataFrame,
    strategy: str = "mean",
    columns: list[str] | None = None,
    fill_value: float | None = None,
) -> pd.DataFrame:
    """Impute missing values in the given columns (all columns if omitted).

    strategy: "mean" | "median" | "mode" | "constant" | "forward_fill" | "backward_fill"
    """
    df = df.copy()
    cols = columns if columns else df.columns.tolist()

    for col in cols:
        if strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            mode = df[col].mode()
            if not mode.empty:
                df[col] = df[col].fillna(mode.iloc[0])
        elif strategy == "constant":
            df[col] = df[col].fillna(fill_value)
        elif strategy == "forward_fill":
            df[col] = df[col].ffill()
        elif strategy == "backward_fill":
            df[col] = df[col].bfill()
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

    return df


def interpolate_missing(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "linear",
) -> pd.DataFrame:
    """Fill missing values by interpolation (good for time-series sensor data)."""
    df = df.copy()
    cols = columns if columns else df.columns.tolist()
    df[cols] = df[cols].interpolate(method=method)
    return df
