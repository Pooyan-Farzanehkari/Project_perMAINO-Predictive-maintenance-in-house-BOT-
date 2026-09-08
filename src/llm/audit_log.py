"""Append-only audit log of every real tool call the agent makes.

Independent ground truth: what actually happened, decoupled from anything the
model says in its prose reply. A fabrication-detection heuristic can miss a
case; this log can't, since it's only ever written from real Python results,
never from model text.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.pipeline.load_data import PROJECT_ROOT

AUDIT_LOG_PATH = PROJECT_ROOT / "data" / "processed" / "tool_call_log.jsonl"


def log_tool_call(
    tool_name: str,
    tool_input: dict[str, Any],
    result: Any,
    is_error: bool = False,
    path: Path = AUDIT_LOG_PATH,
) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "input": tool_input,
        "result": result,
        "is_error": is_error,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def read_audit_log(path: Path = AUDIT_LOG_PATH) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]
