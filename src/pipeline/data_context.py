"""Persist and retrieve structured knowledge about the dataset's sensors/asset.

Established once (via chat or a description file), then reused across
sessions so the agent doesn't have to re-derive sensor semantics every run.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.pipeline.load_data import PROJECT_ROOT

DATA_CONTEXT_PATH = PROJECT_ROOT / "data" / "processed" / "data_context.json"


def save_data_context(context: dict, path: Path = DATA_CONTEXT_PATH) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(context, indent=2))
    return context


def load_data_context(path: Path = DATA_CONTEXT_PATH) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def record_data_context(
    columns: list[dict],
    asset_description: str | None = None,
    path: Path = DATA_CONTEXT_PATH,
) -> dict:
    """Upsert columns into any existing saved context (does not erase prior entries)."""
    context = load_data_context(path=path) or {"asset_description": None, "columns": {}}
    if asset_description is not None:
        context["asset_description"] = asset_description
    for col in columns:
        name = col["name"]
        context["columns"][name] = {k: v for k, v in col.items() if k != "name"}
    return save_data_context(context, path=path)


def extract_file_text(path: str, max_chars: int = 20000) -> str:
    """Extract text from a .pdf or plain-text file.

    `path` may be absolute or relative to the project root.
    """
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = file_path.read_text()

    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[... truncated, showing {max_chars} of {len(text)} characters ...]"
    return text
