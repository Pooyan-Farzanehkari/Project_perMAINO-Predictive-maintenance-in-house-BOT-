import pandas as pd
import pytest

from src.llm.tool_registry import get_tool
from src.llm.tools import execute_tool, get_tool_schemas

MUTATING_TOOLS = {
    "drop_missing_rows",
    "fill_missing",
    "interpolate_missing",
    "min_max_scale",
    "z_score_scale",
    "robust_scale",
}


NON_MUTATING_TOOLS = [
    "profile_dataset",
    "read_description_file",
    "record_data_context",
    "get_data_context",
    "get_tool_call_log",
]


def test_all_expected_tools_registered():
    names = {s["name"] for s in get_tool_schemas()}
    assert names == MUTATING_TOOLS | set(NON_MUTATING_TOOLS)


def test_mutating_tools_flagged_returns_df():
    for name in MUTATING_TOOLS:
        assert get_tool(name).returns_df is True


def test_non_mutating_tools_not_flagged_returns_df():
    for name in NON_MUTATING_TOOLS:
        assert get_tool(name).returns_df is False


def test_intake_tools_flagged_needs_df_false():
    for name in ["read_description_file", "record_data_context", "get_data_context", "get_tool_call_log"]:
        assert get_tool(name).needs_df is False


def test_schemas_have_required_keys():
    for schema in get_tool_schemas():
        assert "name" in schema
        assert "description" in schema
        assert schema["input_schema"]["type"] == "object"


def test_execute_tool_profile_dataset():
    df = pd.DataFrame({"a": [1.0, 2.0, None]})
    profile = execute_tool("profile_dataset", df)
    assert profile["n_rows"] == 3


def test_execute_tool_fill_missing():
    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    result = execute_tool("fill_missing", df, strategy="mean", columns=["a"])
    assert result["a"].isna().sum() == 0


def test_execute_tool_get_tool_call_log_returns_list():
    result = execute_tool("get_tool_call_log")
    assert isinstance(result, list)


def test_execute_unknown_tool_raises():
    df = pd.DataFrame({"a": [1.0]})
    with pytest.raises(KeyError):
        execute_tool("does_not_exist", df)
