import json
import session_agent
from conftest import assistant, tool_call

def new_session(monkeypatch, session_id):
    monkeypatch.setattr(session_agent, "create_client", lambda: object())
    return session_agent.AgentSession(session_id)

def test_independent_sessions(monkeypatch):
    a, b = new_session(monkeypatch, "A"), new_session(monkeypatch, "B")
    a.messages.append({"role": "user", "content": "only A"})
    assert a.messages is not b.messages and len(b.messages) == 1

def test_messages_history_maintained(monkeypatch):
    session = new_session(monkeypatch, "history")
    monkeypatch.setattr(session, "call_llm", lambda: assistant("hello"))
    assert session.chat("first") == "hello"
    assert [x["role"] for x in session.messages] == ["system", "user", "assistant"]

def test_session_id_auto_override(monkeypatch):
    session = new_session(monkeypatch, "trusted")
    responses = iter([assistant(tool_calls=[tool_call("c1", "query_recognition_history", '{"session_id":"invented"}')]), assistant("done")])
    captured = []
    monkeypatch.setattr(session, "call_llm", lambda: next(responses))
    monkeypatch.setattr(session_agent, "execute_tool", lambda tool_name, arguments: captured.append(arguments.copy()) or {"success": True})
    session.chat("history")
    assert captured[0]["session_id"] == "trusted"

def test_multiple_turn_history(monkeypatch):
    session = new_session(monkeypatch, "multi")
    responses = iter([assistant("one"), assistant("two")])
    monkeypatch.setattr(session, "call_llm", lambda: next(responses))
    session.chat("q1"); session.chat("q2")
    assert [x["content"] for x in session.messages[1:]] == ["q1", "one", "q2", "two"]

def test_session_tool_result_saved(monkeypatch):
    session = new_session(monkeypatch, "tools")
    responses = iter([assistant(tool_calls=[tool_call("c1", "query_recognition_history")]), assistant("done")])
    monkeypatch.setattr(session, "call_llm", lambda: next(responses))
    monkeypatch.setattr(session_agent, "execute_tool", lambda **kwargs: {"success": True, "count": 0})
    session.chat("history")
    message = next(x for x in session.messages if x["role"] == "tool")
    assert message["tool_call_id"] == "c1" and json.loads(message["content"])["count"] == 0
