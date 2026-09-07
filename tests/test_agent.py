import json
from types import SimpleNamespace

import pandas as pd

from src.llm.agent import AgentSession


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


def test_run_turn_single_tool_call_then_final_text():
    tool_use = _tool_use("toolu_1", "profile_dataset", {})
    first = _response([tool_use], "tool_use")
    second = _response([_text("Done, dataset has 3 rows.")], "end_turn")
    client = FakeClient([first, second])

    df = pd.DataFrame({"a": [1.0, 2.0, None]})
    session = AgentSession(df=df, client=client)
    reply = session.run_turn("profile the data")

    assert reply == "Done, dataset has 3 rows."
    sent_result = client.messages.create_calls[1]["messages"][-1]["content"][0]
    assert sent_result["tool_use_id"] == "toolu_1"
    assert json.loads(sent_result["content"])["n_rows"] == 3


def test_run_turn_applies_df_returning_tool():
    tool_use = _tool_use("toolu_1", "fill_missing", {"strategy": "mean", "columns": ["a"]})
    first = _response([tool_use], "tool_use")
    second = _response([_text("Filled missing values.")], "end_turn")
    client = FakeClient([first, second])

    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    session = AgentSession(df=df, client=client)
    session.run_turn("fill missing values")

    assert session.df["a"].isna().sum() == 0
    sent_result = client.messages.create_calls[1]["messages"][-1]["content"][0]
    assert "Applied. New shape:" in sent_result["content"]


def test_run_turn_tool_error_reported_as_is_error():
    tool_use = _tool_use("toolu_1", "fill_missing", {"strategy": "bogus"})
    first = _response([tool_use], "tool_use")
    second = _response([_text("That strategy isn't supported.")], "end_turn")
    client = FakeClient([first, second])

    df = pd.DataFrame({"a": [1.0, None]})
    session = AgentSession(df=df, client=client)
    session.run_turn("fill with bogus strategy")

    sent_result = client.messages.create_calls[1]["messages"][-1]["content"][0]
    assert sent_result["is_error"] is True
    assert "bogus" in sent_result["content"] or "Unknown strategy" in sent_result["content"]


def test_run_turn_unknown_tool_reported_as_is_error():
    tool_use = _tool_use("toolu_1", "does_not_exist", {})
    first = _response([tool_use], "tool_use")
    second = _response([_text("Sorry, that tool doesn't exist.")], "end_turn")
    client = FakeClient([first, second])

    df = pd.DataFrame({"a": [1.0]})
    session = AgentSession(df=df, client=client)
    session.run_turn("call a bogus tool")

    sent_result = client.messages.create_calls[1]["messages"][-1]["content"][0]
    assert sent_result["is_error"] is True


def test_run_turn_hits_iteration_cap():
    responses = [_response([_tool_use(f"t{i}", "profile_dataset", {})], "tool_use") for i in range(20)]
    client = FakeClient(responses)

    df = pd.DataFrame({"a": [1.0]})
    session = AgentSession(df=df, client=client)
    reply = session.run_turn("keep calling tools forever")

    assert "tool-call limit" in reply
