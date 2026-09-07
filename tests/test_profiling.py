import json

import numpy as np
import pandas as pd

from src.pipeline.profiling import profile_dataset


def test_profile_dataset_shape_and_columns():
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": ["x", "y", "z"]})
    profile = profile_dataset(df)
    assert profile["n_rows"] == 3
    assert profile["n_columns"] == 2
    assert set(profile["columns"]) == {"a", "b"}


def test_profile_dataset_numeric_stats():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan]})
    profile = profile_dataset(df)
    col = profile["columns"]["a"]
    assert col["missing_count"] == 1
    assert col["missing_pct"] == 25.0
    assert col["min"] == 1.0
    assert col["max"] == 3.0
    assert col["mean"] == 2.0


def test_profile_dataset_categorical_unique_values():
    df = pd.DataFrame({"cat": ["x", "y", "x"]})
    profile = profile_dataset(df, max_categorical_unique=5)
    assert set(profile["columns"]["cat"]["unique_values"]) == {"x", "y"}


def test_profile_dataset_is_json_serializable():
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan], "cat": ["x", "y", "z"]})
    profile = profile_dataset(df)
    json.dumps(profile)  # should not raise
