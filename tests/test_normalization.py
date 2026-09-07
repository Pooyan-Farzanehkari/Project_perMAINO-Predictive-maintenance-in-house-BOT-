import pandas as pd
import pytest

from src.features.normalization import min_max_scale, robust_scale, z_score_scale


@pytest.fixture
def sample_df():
    return pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [10.0, 10.0, 10.0, 10.0, 10.0]})


def test_min_max_scale_default_range(sample_df):
    result = min_max_scale(sample_df, columns=["x"])
    assert result["x"].min() == 0.0
    assert result["x"].max() == 1.0


def test_min_max_scale_custom_range(sample_df):
    result = min_max_scale(sample_df, columns=["x"], feature_min=-1.0, feature_max=1.0)
    assert result["x"].min() == -1.0
    assert result["x"].max() == 1.0


def test_min_max_scale_constant_column(sample_df):
    result = min_max_scale(sample_df, columns=["y"], feature_min=0.0, feature_max=1.0)
    assert (result["y"] == 0.0).all()


def test_z_score_scale(sample_df):
    result = z_score_scale(sample_df, columns=["x"])
    assert result["x"].mean() == pytest.approx(0.0, abs=1e-9)
    assert result["x"].std() == pytest.approx(1.0)


def test_z_score_scale_constant_column(sample_df):
    result = z_score_scale(sample_df, columns=["y"])
    assert (result["y"] == 0.0).all()


def test_robust_scale(sample_df):
    result = robust_scale(sample_df, columns=["x"])
    assert result["x"].median() == pytest.approx(0.0)


def test_robust_scale_constant_column(sample_df):
    result = robust_scale(sample_df, columns=["y"])
    assert (result["y"] == 0.0).all()


def test_scale_defaults_to_all_numeric_columns(sample_df):
    df = sample_df.assign(label=["a", "b", "c", "d", "e"])
    result = min_max_scale(df)
    assert result["x"].max() == 1.0
    assert list(result["label"]) == list(df["label"])
