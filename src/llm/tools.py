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
from src.pipeline.data_context import extract_file_text, load_data_context, record_data_context
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
    returns_df=True,
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
    returns_df=True,
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
    returns_df=True,
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
    returns_df=True,
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
    returns_df=True,
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
    returns_df=True,
)

register_tool(
    name="read_description_file",
    description=(
        "Read a text or PDF file (e.g. a dataset description or sensor spec sheet) and "
        "return its extracted text so you can read it. Path may be absolute or relative "
        "to the project root."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file, e.g. 'metropt+3+dataset/Data Description_Metro.pdf'.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters to return; truncates with a note if exceeded.",
                "default": 20000,
            },
        },
        "required": ["path"],
    },
    func=extract_file_text,
    needs_df=False,
)

register_tool(
    name="record_data_context",
    description=(
        "Save structured knowledge about the dataset's sensors/columns and the asset they "
        "describe, so this only needs to be established once and persists across sessions. "
        "Merges into any previously saved context -- call it incrementally as you learn "
        "things, it will not erase columns recorded earlier."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "asset_description": {
                "type": "string",
                "description": "Short description of the asset/machine this data comes from.",
            },
            "columns": {
                "type": "array",
                "description": "One entry per sensor/column.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Must match a column in the dataset."},
                        "sensor_type": {"type": "string"},
                        "unit": {"type": "string"},
                        "expected_min": {"type": "number"},
                        "expected_max": {"type": "number"},
                        "notes": {"type": "string"},
                    },
                    "required": ["name"],
                },
            },
        },
        "required": ["columns"],
    },
    func=record_data_context,
    needs_df=False,
)

register_tool(
    name="get_data_context",
    description=(
        "Retrieve the previously saved data context (sensor metadata + asset description), "
        "if any. Call this at the start of a session before asking the engineer to "
        "re-describe the dataset."
    ),
    input_schema={"type": "object", "properties": {}, "required": []},
    func=lambda: load_data_context() or {"status": "no_data_context_saved_yet"},
    needs_df=False,
)

__all__ = ["get_tool_schemas", "execute_tool"]
