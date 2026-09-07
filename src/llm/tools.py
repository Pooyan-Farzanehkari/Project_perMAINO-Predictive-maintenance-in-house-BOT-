"""Catalog of preprocessing tools exposed to the LLM agent.

Registers each preprocessing function with a name, a description, and a
JSON schema for its arguments. The LLM only ever sees this catalog plus a
compact `profile_dataset` summary (see src/pipeline/profiling.py) -- never
the raw DataFrame -- and decides which tool to call based on that.
"""

from src.cleaning.missing_values import (
    drop_missing_rows,
    fill_missing,
    interpolate_missing,
)
from src.features.normalization import min_max_scale, robust_scale, z_score_scale
from src.llm.tool_registry import execute_tool, get_tool_schemas, register_tool
from src.pipeline.profiling import profile_dataset

_COLUMNS_PARAM = {
    "type": "array",
    "items": {"type": "string"},
    "description": "Columns to apply this to. Omit to apply to all (numeric) columns.",
}

register_tool(
    name="profile_dataset",
    description=(
        "Get a compact summary of the current dataset: shape, dtypes, missing-value "
        "counts/percentages, and basic numeric stats (min/max/mean/std) per column. "
        "Returns no raw rows. Call this first, and again after transformations, to "
        "decide what preprocessing is still needed."
    ),
    input_schema={"type": "object", "properties": {}, "required": []},
    func=profile_dataset,
)

register_tool(
    name="drop_missing_rows",
    description="Drop rows that contain missing values in the given columns.",
    input_schema={
        "type": "object",
        "properties": {
            "columns": _COLUMNS_PARAM,
            "how": {
                "type": "string",
                "enum": ["any", "all"],
                "description": "Drop a row if 'any' checked column is missing, or only if 'all' are.",
                "default": "any",
            },
        },
        "required": [],
    },
    func=drop_missing_rows,
)

register_tool(
    name="fill_missing",
    description="Impute missing values using a simple strategy: mean, median, mode, a constant, or forward/backward fill.",
    input_schema={
        "type": "object",
        "properties": {
            "strategy": {
                "type": "string",
                "enum": ["mean", "median", "mode", "constant", "forward_fill", "backward_fill"],
                "description": "Imputation strategy to apply.",
            },
            "columns": _COLUMNS_PARAM,
            "fill_value": {
                "type": "number",
                "description": "Value to use when strategy is 'constant'.",
            },
        },
        "required": ["strategy"],
    },
    func=fill_missing,
)

register_tool(
    name="interpolate_missing",
    description="Fill missing values by interpolating between neighboring values, in row order. Good for time-series sensor data.",
    input_schema={
        "type": "object",
        "properties": {
            "columns": _COLUMNS_PARAM,
            "method": {
                "type": "string",
                "enum": ["linear", "time", "nearest"],
                "description": "Interpolation method.",
                "default": "linear",
            },
        },
        "required": [],
    },
    func=interpolate_missing,
)

register_tool(
    name="min_max_scale",
    description="Rescale numeric columns to a fixed range (default 0-1) via min-max normalization.",
    input_schema={
        "type": "object",
        "properties": {
            "columns": _COLUMNS_PARAM,
            "feature_min": {"type": "number", "default": 0.0},
            "feature_max": {"type": "number", "default": 1.0},
        },
        "required": [],
    },
    func=min_max_scale,
)

register_tool(
    name="z_score_scale",
    description="Standardize numeric columns to zero mean and unit variance. Sensitive to outliers.",
    input_schema={
        "type": "object",
        "properties": {"columns": _COLUMNS_PARAM},
        "required": [],
    },
    func=z_score_scale,
)

register_tool(
    name="robust_scale",
    description="Scale numeric columns using median and IQR. Robust to outliers.",
    input_schema={
        "type": "object",
        "properties": {"columns": _COLUMNS_PARAM},
        "required": [],
    },
    func=robust_scale,
)

__all__ = ["get_tool_schemas", "execute_tool"]
