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

from src.llm.audit_log import AUDIT_LOG_PATH, log_tool_call
from src.llm.tool_registry import execute_tool, get_tool, get_tool_schemas
from src.pipeline.load_data import PROCESSED_PARQUET_PATH, load_raw_data

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
DEFAULT_EFFORT = os.environ.get("ANTHROPIC_EFFORT", "medium")
MAX_TOKENS = 4096
MAX_TOOL_ITERATIONS = 8
MAX_UNRELIABLE_RETRIES = 2

# Two distinct unreliable-reply patterns seen in testing, both treated the same way:
# (1) the model writes a fake tool call/result as plain text instead of using the real
#     tool-calling mechanism, and (2) the model falsely claims no tool interface is
#     available and refuses to call anything -- catch either rather than trust it.
_FABRICATION_MARKERS = ("invoke", "function_calls", "function_results")
_FALSE_REFUSAL_MARKERS = (
    "no tool-calling interface",
    "not wired up",
    "no callable function",
    "cannot call any tool",
    "can't call any tool",
    "don't have access to any tool",
    "no tool interface",
)

SYSTEM_PROMPT = """You are a data preprocessing assistant for a predictive-maintenance \
project. You help an engineer clean and prepare sensor time-series data (currently the \
MetroPT-3 air-compressor dataset) before it is used for modeling.

Use the real, native tool-calling mechanism the API gives you for every one of these \
actions. Do not describe, simulate, or write out a tool call or its result yourself in \
your response text -- only state facts (row counts, missing-value counts, column stats) \
that came from an actual tool result you just received, even for a dataset you recognize. \
Make one tool call, wait for its real result, then decide the next step.

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
- Look up the real, independently-logged record of recent tool calls and their real \
results (get_tool_call_log) -- use this if the engineer asks you to justify a claim.

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
        effort: str = DEFAULT_EFFORT,
        audit_log_path=AUDIT_LOG_PATH,
    ):
        self.df = df
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.audit_log_path = audit_log_path
        self.messages: list[dict] = []

    def run_turn(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        unreliable_retries = 0

        for _ in range(MAX_TOOL_ITERATIONS):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=get_tool_schemas(),
                output_config={"effort": self.effort},
                messages=self.messages,
            )
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                text = "".join(b.text for b in response.content if b.type == "text")

                if self._looks_unreliable(text):
                    log_tool_call(
                        "_unreliable_reply_detected",
                        {"attempt": unreliable_retries},
                        text[:2000],
                        is_error=True,
                        path=self.audit_log_path,
                    )
                    if unreliable_retries >= MAX_UNRELIABLE_RETRIES:
                        return (
                            "[warning: could not get a reliable reply this turn after "
                            f"{MAX_UNRELIABLE_RETRIES} retries -- please try rephrasing as "
                            "a single, simple step, e.g. \"call get_data_context\"]"
                        )
                    unreliable_retries += 1
                    self.messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Tools are available and working correctly in this "
                                "conversation -- there is no problem with the tool-calling "
                                "mechanism. Call one real tool now, exactly as instructed."
                            ),
                        }
                    )
                    continue

                return text

            tool_results = [
                self._execute_tool_block(block)
                for block in response.content
                if block.type == "tool_use"
            ]

            if not tool_results:
                # stop_reason was "tool_use" but no real tool_use block was found --
                # the API rejects an empty tool-results message, and looping again with
                # nothing new would likely just repeat. Surface this rather than crash.
                log_tool_call(
                    "_empty_tool_use_response",
                    {},
                    "stop_reason was tool_use but response.content had no tool_use block",
                    is_error=True,
                    path=self.audit_log_path,
                )
                return (
                    "[warning: the model signaled a tool call but didn't actually specify "
                    "one -- please rephrase your request]"
                )

            self.messages.append({"role": "user", "content": tool_results})

        return "Reached the tool-call limit for this turn -- please rephrase or continue."

    @staticmethod
    def _looks_unreliable(text: str) -> bool:
        lowered = text.lower()
        return any(marker in lowered for marker in _FABRICATION_MARKERS + _FALSE_REFUSAL_MARKERS)

    def _execute_tool_block(self, block) -> dict:
        try:
            tool = get_tool(block.name)
            result = execute_tool(block.name, df=self.df, **block.input)
        except Exception as exc:
            log_tool_call(block.name, block.input, str(exc), is_error=True, path=self.audit_log_path)
            return {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": f"Error: {exc}",
                "is_error": True,
            }

        if tool.returns_df:
            self.df = result
            content = f"Applied. New shape: {result.shape[0]} rows x {result.shape[1]} columns."
            log_tool_call(block.name, block.input, {"new_shape": list(result.shape)}, path=self.audit_log_path)
        else:
            content = json.dumps(result, default=str)
            log_tool_call(block.name, block.input, result, path=self.audit_log_path)

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
