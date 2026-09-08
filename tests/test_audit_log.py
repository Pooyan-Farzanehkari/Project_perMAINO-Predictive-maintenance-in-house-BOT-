from src.llm.audit_log import log_tool_call, read_audit_log


def test_read_audit_log_missing_file_returns_empty(tmp_path):
    assert read_audit_log(path=tmp_path / "missing.jsonl") == []


def test_log_and_read_round_trip(tmp_path):
    path = tmp_path / "audit.jsonl"

    log_tool_call("profile_dataset", {}, {"n_rows": 3}, path=path)
    log_tool_call("fill_missing", {"strategy": "mean"}, {"new_shape": [3, 1]}, path=path)

    entries = read_audit_log(path=path)

    assert len(entries) == 2
    assert entries[0]["tool"] == "profile_dataset"
    assert entries[0]["result"] == {"n_rows": 3}
    assert entries[0]["is_error"] is False
    assert entries[1]["tool"] == "fill_missing"
    assert "timestamp" in entries[0]


def test_log_tool_call_records_errors(tmp_path):
    path = tmp_path / "audit.jsonl"

    log_tool_call("fill_missing", {"strategy": "bogus"}, "Unknown strategy: 'bogus'", is_error=True, path=path)

    entries = read_audit_log(path=path)
    assert entries[0]["is_error"] is True
