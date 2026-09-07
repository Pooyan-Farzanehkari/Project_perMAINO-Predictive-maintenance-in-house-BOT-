from pathlib import Path

import pytest

from src.pipeline.data_context import (
    extract_file_text,
    load_data_context,
    record_data_context,
    save_data_context,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_PDF = PROJECT_ROOT / "metropt+3+dataset" / "Data Description_Metro.pdf"


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "context.json"
    context = {"asset_description": "air compressor", "columns": {"TP2": {"unit": "bar"}}}

    save_data_context(context, path=path)
    loaded = load_data_context(path=path)

    assert loaded == context


def test_load_missing_returns_none(tmp_path):
    assert load_data_context(path=tmp_path / "missing.json") is None


def test_record_data_context_merges_across_calls(tmp_path):
    path = tmp_path / "context.json"

    record_data_context(
        columns=[{"name": "TP2", "sensor_type": "pressure"}],
        asset_description="air compressor unit 1",
        path=path,
    )
    result = record_data_context(columns=[{"name": "Oil_temperature", "sensor_type": "temperature"}], path=path)

    assert result["asset_description"] == "air compressor unit 1"
    assert result["columns"]["TP2"]["sensor_type"] == "pressure"
    assert result["columns"]["Oil_temperature"]["sensor_type"] == "temperature"


def test_record_data_context_updates_existing_column(tmp_path):
    path = tmp_path / "context.json"

    record_data_context(columns=[{"name": "TP2", "unit": "bar"}], path=path)
    result = record_data_context(columns=[{"name": "TP2", "unit": "psi"}], path=path)

    assert result["columns"]["TP2"]["unit"] == "psi"


def test_extract_file_text_plain_text(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello sensor world")

    assert extract_file_text(str(path)) == "hello sensor world"


def test_extract_file_text_truncates(tmp_path):
    path = tmp_path / "long.txt"
    path.write_text("x" * 100)

    result = extract_file_text(str(path), max_chars=10)

    assert result.startswith("x" * 10)
    assert "truncated" in result


def test_extract_file_text_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_file_text(str(tmp_path / "nope.txt"))


@pytest.mark.skipif(not REAL_PDF.exists(), reason="dataset not present locally (gitignored, download separately)")
def test_extract_file_text_real_pdf():
    text = extract_file_text(str(REAL_PDF))

    assert isinstance(text, str)
    assert len(text) > 0
