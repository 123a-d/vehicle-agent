import json
import multi_tool_agent
from conftest import assistant, tool_call

def run_fake(monkeypatch, responses, result=None, max_steps=10):
    calls, snapshots = [], []
    sequence = iter(responses)
    monkeypatch.setattr(multi_tool_agent, "create_client", lambda: object())
    def fake_llm(client, messages):
        snapshots.append(list(messages)); return next(sequence)
    def fake_tool(tool_name, arguments):
        calls.append((tool_name, arguments)); return result if result is not None else {"success": True}
    monkeypatch.setattr(multi_tool_agent, "call_llm", fake_llm)
    monkeypatch.setattr(multi_tool_agent, "execute_tool", fake_tool)
    return multi_tool_agent.run_agent_loop("request", max_steps), calls, snapshots

def test_single_tool_call(monkeypatch):
    answer, calls, _ = run_fake(monkeypatch, [assistant(tool_calls=[tool_call("c1", "recognize_vehicle", '{"image_path":"x.jpg"}')]), assistant("done")])
    assert answer == "done" and calls == [("recognize_vehicle", {"image_path": "x.jpg"})]

def test_multiple_tool_calls_same_round(monkeypatch):
    _, calls, _ = run_fake(monkeypatch, [assistant(tool_calls=[tool_call("c1", "a"), tool_call("c2", "b")]), assistant("done")])
    assert [x[0] for x in calls] == ["a", "b"]

def test_consecutive_tool_call_rounds(monkeypatch):
    answer, calls, _ = run_fake(monkeypatch, [assistant(tool_calls=[tool_call("c1", "a")]), assistant(tool_calls=[tool_call("c2", "b")]), assistant("done")])
    assert answer == "done" and [x[0] for x in calls] == ["a", "b"]

def test_tool_result_backfilled(monkeypatch):
    result = {"success": True, "vehicle_id": "0009"}
    _, _, snapshots = run_fake(monkeypatch, [assistant(tool_calls=[tool_call("c1", "a")]), assistant("done")], result)
    message = [x for x in snapshots[1] if x["role"] == "tool"][0]
    assert message["tool_call_id"] == "c1" and json.loads(message["content"]) == result

def test_tool_error_backfilled(monkeypatch):
    result = {"success": False, "error_type": "RuntimeError"}
    answer, _, snapshots = run_fake(monkeypatch, [assistant(tool_calls=[tool_call("c1", "bad")]), assistant("explained")], result)
    assert answer == "explained" and json.loads(snapshots[1][-1]["content"])["success"] is False

def test_invalid_tool_arguments(monkeypatch):
    answer, calls, snapshots = run_fake(monkeypatch, [assistant(tool_calls=[tool_call("c1", "bad", "{")]), assistant("invalid")])
    assert answer == "invalid" and calls == []
    assert json.loads(snapshots[1][-1]["content"])["error_type"] == "InvalidToolArguments"

def test_max_steps_ten(monkeypatch):
    answer, calls, snapshots = run_fake(monkeypatch, [assistant(tool_calls=[tool_call(str(i), "loop")]) for i in range(10)])
    assert "最大执行步数" in answer and len(calls) == len(snapshots) == 10
