import pandas as pd
import pytest

from src.llm.tool_registry import _REGISTRY, execute_tool, get_tool, register_tool


@pytest.fixture
def df_free_tool():
    name = "_test_only_df_free_tool"
    register_tool(
        name=name,
        description="test",
        input_schema={"type": "object", "properties": {}, "required": []},
        func=lambda greeting="hi": {"echo": greeting},
        needs_df=False,
    )
    yield name
    del _REGISTRY[name]


@pytest.fixture
def df_returning_tool():
    name = "_test_only_df_returning_tool"
    register_tool(
        name=name,
        description="test",
        input_schema={"type": "object", "properties": {}, "required": []},
        func=lambda df: df.assign(flag=1),
        needs_df=True,
        returns_df=True,
    )
    yield name
    del _REGISTRY[name]


def test_needs_df_false_dispatches_without_df(df_free_tool):
    assert execute_tool(df_free_tool, greeting="hello") == {"echo": "hello"}


def test_needs_df_true_without_df_raises(df_returning_tool):
    with pytest.raises(ValueError):
        execute_tool(df_returning_tool)


def test_returns_df_flag_and_dispatch(df_returning_tool):
    df = pd.DataFrame({"a": [1]})
    result = execute_tool(df_returning_tool, df)
    assert get_tool(df_returning_tool).returns_df is True
    assert "flag" in result.columns


def test_get_tool_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        get_tool("_does_not_exist")


def test_register_duplicate_name_raises(df_free_tool):
    with pytest.raises(ValueError):
        register_tool(
            name=df_free_tool,
            description="dup",
            input_schema={"type": "object", "properties": {}, "required": []},
            func=lambda: None,
            needs_df=False,
        )
