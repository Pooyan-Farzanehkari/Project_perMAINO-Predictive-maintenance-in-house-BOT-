"""Conversational agent: Claude decides which preprocessing tool to call.

The LLM never sees the raw DataFrame -- only profile_dataset()'s compact
summary and short confirmations after mutating tool calls (see tool_registry
"returns_df" handling below).
"""

from __future__ import annotations

import json
import os

import anthropic
import pandas as pd
from dotenv import load_dotenv

from src.llm.tool_registry import execute_tool, get_tool, get_tool_schemas
from src.pipeline.load_data import PROCESSED_PARQUET_PATH, load_raw_data

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
MAX_TOKENS = 16000
MAX_TOOL_ITERATIONS = 8

SYSTEM_PROMPT = """You are a data preprocessing assistant for a predictive-maintenance \
project. You help an engineer clean and prepare sensor time-series data (currently the \
MetroPT-3 air-compressor dataset) before it is used for modeling.

You have tools to:
- Inspect the dataset (profile_dataset) without ever seeing raw rows.
- Clean missing values (drop_missing_rows, fill_missing, interpolate_missing).
- Normalize/scale numeric columns (min_max_scale, z_score_scale, robust_scale).
- Read a description file such as a sensor spec sheet or dataset README \
(read_description_file).
- Save and load a structured "data context" describing each column's sensor type, unit, \
expected range, and notes, plus a short description of the asset the data comes from \
(record_data_context, get_data_context) -- so this only has to be established once and \
persists across sessions.

At the start of a conversation, call get_data_context first. If no context is saved yet, \
offer to read a description file (ask the engineer for its path if you don't know it) \
and/or ask the engineer directly, then call record_data_context to save what you learn. \
Don't re-ask for information already present in the saved context.

Always call profile_dataset before and after any transformation that changes the data, so \
you and the engineer can see the effect. Explain what you're about to do and why before \
calling a mutating tool, and summarize the result afterward in plain language (rows/columns \
affected, missing-value counts, etc.) -- never dump raw values back to the engineer.

Only use the tools listed above. Outlier detection, time-series-specific handling, and \
derived feature engineering are out of scope for now -- if the engineer asks for one of \
those, say it isn't available yet rather than improvising with the existing tools."""


class AgentSession:
    def __init__(
        self,
        df: pd.DataFrame,
        client: anthropic.Anthropic | None = None,
        model: str = DEFAULT_MODEL,
    ):
        self.df = df
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.messages: list[dict] = []

    def run_turn(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})

        for _ in range(MAX_TOOL_ITERATIONS):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=get_tool_schemas(),
                output_config={"effort": "medium"},
                messages=self.messages,
            )
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                return "".join(b.text for b in response.content if b.type == "text")

            tool_results = [
                self._execute_tool_block(block)
                for block in response.content
                if block.type == "tool_use"
            ]
            self.messages.append({"role": "user", "content": tool_results})

        return "Reached the tool-call limit for this turn -- please rephrase or continue."

    def _execute_tool_block(self, block) -> dict:
        try:
            tool = get_tool(block.name)
            result = execute_tool(block.name, df=self.df, **block.input)
        except Exception as exc:
            return {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": f"Error: {exc}",
                "is_error": True,
            }

        if tool.returns_df:
            self.df = result
            content = f"Applied. New shape: {result.shape[0]} rows x {result.shape[1]} columns."
        else:
            content = json.dumps(result, default=str)

        return {"type": "tool_result", "tool_use_id": block.id, "content": content}


def _load_working_df() -> pd.DataFrame:
    if PROCESSED_PARQUET_PATH.exists():
        return pd.read_parquet(PROCESSED_PARQUET_PATH)
    return load_raw_data()


def main() -> None:
    load_dotenv()
    df = _load_working_df()
    session = AgentSession(df=df)
    print("perMAINO agent ready. Type 'exit' to quit.")
    while True:
        try:
            user_text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break
        print(session.run_turn(user_text))


if __name__ == "__main__":
    main()
