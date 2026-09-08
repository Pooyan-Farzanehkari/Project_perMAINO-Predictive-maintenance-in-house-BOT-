import json
from types import SimpleNamespace

import pandas as pd

from src.llm.agent import AgentSession
from src.llm.audit_log import read_audit_log


def _text(text):
    return SimpleNamespace(type="text", text=text)


def _tool_use(id_, name, input_):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=input_)


def _response(content, stop_reason):
    return SimpleNamespace(content=content, stop_reason=stop_reason)


class FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.create_calls = []

    def create(self, **kwargs):
        # AgentSession mutates self.messages in place after this call returns, so
        # snapshot the list now rather than storing a live reference to it.
        self.create_calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


def test_run_turn_single_tool_call_then_final_text(tmp_path):
    tool_use = _tool_use("toolu_1", "profile_dataset", {})
    first = _response([tool_use], "tool_use")
    second = _response([_text("Done, dataset has 3 rows.")], "end_turn")
    client = FakeClient([first, second])

    df = pd.DataFrame({"a": [1.0, 2.0, None]})
    session = AgentSession(df=df, client=client, audit_log_path=tmp_path / "audit.jsonl")
    reply = session.run_turn("profile the data")

    assert reply == "Done, dataset has 3 rows."
    sent_result = client.messages.create_calls[1]["messages"][-1]["content"][0]
    assert sent_result["tool_use_id"] == "toolu_1"
    assert json.loads(sent_result["content"])["n_rows"] == 3


def test_run_turn_applies_df_returning_tool(tmp_path):
    tool_use = _tool_use("toolu_1", "fill_missing", {"strategy": "mean", "columns": ["a"]})
    first = _response([tool_use], "tool_use")
    second = _response([_text("Filled missing values.")], "end_turn")
    client = FakeClient([first, second])

    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    audit_path = tmp_path / "audit.jsonl"
    session = AgentSession(df=df, client=client, audit_log_path=audit_path)
    session.run_turn("fill missing values")

    assert session.df["a"].isna().sum() == 0
    sent_result = client.messages.create_calls[1]["messages"][-1]["content"][0]
    assert "Applied. New shape:" in sent_result["content"]

    entries = read_audit_log(path=audit_path)
    assert len(entries) == 1
    assert entries[0]["tool"] == "fill_missing"
    assert entries[0]["result"] == {"new_shape": [3, 1]}


def test_run_turn_tool_error_reported_as_is_error(tmp_path):
    tool_use = _tool_use("toolu_1", "fill_missing", {"strategy": "bogus"})
    first = _response([tool_use], "tool_use")
    second = _response([_text("That strategy isn't supported.")], "end_turn")
    client = FakeClient([first, second])

    df = pd.DataFrame({"a": [1.0, None]})
    session = AgentSession(df=df, client=client, audit_log_path=tmp_path / "audit.jsonl")
    session.run_turn("fill with bogus strategy")

    sent_result = client.messages.create_calls[1]["messages"][-1]["content"][0]
    assert sent_result["is_error"] is True
    assert "bogus" in sent_result["content"] or "Unknown strategy" in sent_result["content"]


def test_run_turn_unknown_tool_reported_as_is_error(tmp_path):
    tool_use = _tool_use("toolu_1", "does_not_exist", {})
    first = _response([tool_use], "tool_use")
    second = _response([_text("Sorry, that tool doesn't exist.")], "end_turn")
    client = FakeClient([first, second])

    df = pd.DataFrame({"a": [1.0]})
    session = AgentSession(df=df, client=client, audit_log_path=tmp_path / "audit.jsonl")
    session.run_turn("call a bogus tool")

    sent_result = client.messages.create_calls[1]["messages"][-1]["content"][0]
    assert sent_result["is_error"] is True


def test_run_turn_retries_once_on_fabricated_tool_call(tmp_path):
    fabricated = _response(
        [_text('I\'ll check now.\n\nantml:invoke name="get_data_context">\n</invoke>')], "end_turn"
    )
    clean = _response([_text("No context is saved yet.")], "end_turn")
    client = FakeClient([fabricated, clean])

    df = pd.DataFrame({"a": [1.0]})
    audit_path = tmp_path / "audit.jsonl"
    session = AgentSession(df=df, client=client, audit_log_path=audit_path)
    reply = session.run_turn("check the data context")

    assert reply == "No context is saved yet."
    assert len(client.messages.create_calls) == 2
    retry_message = client.messages.create_calls[1]["messages"][-1]
    assert retry_message["role"] == "user"
    assert "working correctly" in retry_message["content"]

    entries = read_audit_log(path=audit_path)
    assert len(entries) == 1
    assert entries[0]["tool"] == "_unreliable_reply_detected"
    assert entries[0]["is_error"] is True

    # The failed attempt and the corrective nudge must never persist into history --
    # only the original question and the eventual clean answer.
    assert session.messages == [
        {"role": "user", "content": "check the data context"},
        {"role": "assistant", "content": clean.content},
    ]


def test_run_turn_detects_false_refusal_claim(tmp_path):
    refusal_text = "I don't have a tool-calling interface -- it's not wired up on my end."
    refusal = _response([_text(refusal_text)], "end_turn")
    clean = _response([_text("Got it, calling now.")], "end_turn")
    client = FakeClient([refusal, clean])

    df = pd.DataFrame({"a": [1.0]})
    audit_path = tmp_path / "audit.jsonl"
    session = AgentSession(df=df, client=client, audit_log_path=audit_path)
    reply = session.run_turn("check the data context")

    assert reply == "Got it, calling now."
    entries = read_audit_log(path=audit_path)
    assert len(entries) == 1
    assert entries[0]["tool"] == "_unreliable_reply_detected"
    assert session.messages == [
        {"role": "user", "content": "check the data context"},
        {"role": "assistant", "content": clean.content},
    ]


def test_run_turn_suppresses_content_after_repeated_unreliable_replies(tmp_path):
    fabricated_text = 'antml:invoke name="profile_dataset"></invoke>'
    fabricated = _response([_text(fabricated_text)], "end_turn")
    client = FakeClient([fabricated, fabricated, fabricated])

    df = pd.DataFrame({"a": [1.0]})
    audit_path = tmp_path / "audit.jsonl"
    session = AgentSession(df=df, client=client, audit_log_path=audit_path)
    reply = session.run_turn("profile it")

    assert reply.startswith("[warning:")
    assert fabricated_text not in reply
    assert len(client.messages.create_calls) == 3

    entries = read_audit_log(path=audit_path)
    assert len(entries) == 3
    assert all(e["tool"] == "_unreliable_reply_detected" for e in entries)

    # Fully suppressed: nothing from the 3 failed attempts should persist, only the
    # engineer's original question -- next turn starts from a clean slate.
    assert session.messages == [{"role": "user", "content": "profile it"}]


def test_unreliable_retry_within_a_turn_does_not_pollute_the_next_turn(tmp_path):
    fabricated = _response(
        [_text('antml:invoke name="profile_dataset"></invoke>')], "end_turn"
    )
    tool_use_response = _response([_tool_use("toolu_1", "profile_dataset", {})], "tool_use")
    turn1_final = _response([_text("Here is the profile.")], "end_turn")
    turn2_final = _response([_text("Sure, doing that now.")], "end_turn")
    client = FakeClient([fabricated, tool_use_response, turn1_final, turn2_final])

    df = pd.DataFrame({"a": [1.0]})
    audit_path = tmp_path / "audit.jsonl"
    session = AgentSession(df=df, client=client, audit_log_path=audit_path)

    reply1 = session.run_turn("profile it")
    assert reply1 == "Here is the profile."

    reply2 = session.run_turn("now clean it up")
    assert reply2 == "Sure, doing that now."

    sent_for_turn2 = client.messages.create_calls[-1]["messages"]
    flattened = json.dumps(sent_for_turn2, default=str)
    assert "antml:invoke" not in flattened
    assert "working correctly" not in flattened


def test_run_turn_handles_tool_use_stop_reason_with_no_tool_use_block(tmp_path):
    # A real (if malformed) response: stop_reason says tool_use but content has none.
    # This previously crashed by sending the API an empty tool-results message.
    malformed = _response([_text("I'll call a tool.")], "tool_use")
    client = FakeClient([malformed])

    df = pd.DataFrame({"a": [1.0]})
    session = AgentSession(df=df, client=client, audit_log_path=tmp_path / "audit.jsonl")
    reply = session.run_turn("do something")

    assert reply.startswith("[warning:")
    assert "didn't actually specify" in reply


def test_run_turn_hits_iteration_cap(tmp_path):
    responses = [_response([_tool_use(f"t{i}", "profile_dataset", {})], "tool_use") for i in range(20)]
    client = FakeClient(responses)

    df = pd.DataFrame({"a": [1.0]})
    session = AgentSession(df=df, client=client, audit_log_path=tmp_path / "audit.jsonl")
    reply = session.run_turn("keep calling tools forever")

    assert "tool-call limit" in reply
