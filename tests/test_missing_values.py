import numpy as np
import pandas as pd
import pytest

from src.cleaning.missing_values import (
    drop_missing_rows,
    fill_missing,
    interpolate_missing,
    missing_value_summary,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, np.nan, 5.0],
            "b": [10.0, 20.0, np.nan, 40.0, 50.0],
            "c": ["x", "y", "x", "x", np.nan],
        }
    )


def test_missing_value_summary(sample_df):
    summary = missing_value_summary(sample_df)
    assert summary.loc["a", "missing_count"] == 2
    assert summary.loc["b", "missing_count"] == 1
    assert summary.loc["a", "missing_pct"] == 40.0


def test_drop_missing_rows_any(sample_df):
    result = drop_missing_rows(sample_df)
    assert len(result) == 1  # only row 0 has no NaN anywhere


def test_drop_missing_rows_subset(sample_df):
    result = drop_missing_rows(sample_df, columns=["a"])
    assert result["a"].isna().sum() == 0
    assert len(result) == 3


def test_fill_missing_mean(sample_df):
    result = fill_missing(sample_df, strategy="mean", columns=["a"])
    assert result["a"].isna().sum() == 0
    assert result["a"].iloc[1] == pytest.approx(3.0)


def test_fill_missing_median(sample_df):
    result = fill_missing(sample_df, strategy="median", columns=["b"])
    assert result["b"].isna().sum() == 0


def test_fill_missing_constant(sample_df):
    result = fill_missing(sample_df, strategy="constant", columns=["a"], fill_value=-1.0)
    assert result["a"].iloc[1] == -1.0
    assert result["a"].iloc[3] == -1.0


def test_fill_missing_forward_fill(sample_df):
    result = fill_missing(sample_df, strategy="forward_fill", columns=["a"])
    assert result["a"].iloc[1] == 1.0


def test_fill_missing_backward_fill(sample_df):
    result = fill_missing(sample_df, strategy="backward_fill", columns=["a"])
    assert result["a"].iloc[1] == 3.0


def test_fill_missing_unknown_strategy_raises(sample_df):
    with pytest.raises(ValueError):
        fill_missing(sample_df, strategy="bogus")


def test_interpolate_missing(sample_df):
    result = interpolate_missing(sample_df, columns=["a"])
    assert result["a"].iloc[1] == pytest.approx(2.0)


def test_drop_missing_rows_does_not_mutate_input(sample_df):
    original = sample_df.copy()
    drop_missing_rows(sample_df)
    pd.testing.assert_frame_equal(sample_df, original)
