"""A small registry mapping tool names to Python callables + their JSON schema.

Keeps the "what can be called" catalog decoupled from the preprocessing
functions themselves, so cleaning/features modules stay plain, reusable
pandas code with no knowledge of the LLM layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    func: Callable[..., Any]
    needs_df: bool = True
    returns_df: bool = False


_REGISTRY: dict[str, Tool] = {}


def register_tool(
    name: str,
    description: str,
    input_schema: dict[str, Any],
    func: Callable,
    needs_df: bool = True,
    returns_df: bool = False,
) -> None:
    if name in _REGISTRY:
        raise ValueError(f"Tool already registered: {name!r}")
    _REGISTRY[name] = Tool(
        name=name,
        description=description,
        input_schema=input_schema,
        func=func,
        needs_df=needs_df,
        returns_df=returns_df,
    )


def get_tool(name: str) -> Tool:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown tool: {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def get_tool_schemas() -> list[dict[str, Any]]:
    """Anthropic tool-use format: [{name, description, input_schema}, ...]."""
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in _REGISTRY.values()
    ]


def execute_tool(name: str, df: pd.DataFrame | None = None, **kwargs) -> Any:
    tool = get_tool(name)
    if tool.needs_df:
        if df is None:
            raise ValueError(f"Tool {name!r} requires a DataFrame but none was provided.")
        return tool.func(df, **kwargs)
    return tool.func(**kwargs)
