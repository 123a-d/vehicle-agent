import logging

import pytest

import agent_core
import agent_tools
import config


def test_tool_definitions_generate_schema_and_registry():
    names = [definition.name for definition in agent_tools.TOOL_DEFINITIONS]
    assert names == [item["function"]["name"] for item in agent_tools.TOOLS]
    assert names == list(agent_tools.TOOL_REGISTRY)
    for definition in agent_tools.TOOL_DEFINITIONS:
        assert agent_tools.TOOL_REGISTRY[definition.name] is definition.handler


def test_tool_definition_contains_required_parts():
    for definition in agent_tools.TOOL_DEFINITIONS:
        assert definition.name
        assert definition.description
        assert definition.parameters["type"] == "object"
        assert callable(definition.handler)


def test_api_key_is_read_from_environment(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret")
    assert config.get_deepseek_api_key() == "test-secret"
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    with pytest.raises(RuntimeError):
        config.get_deepseek_api_key()


def test_unknown_tool_emits_warning_without_arguments(caplog):
    with caplog.at_level(logging.WARNING):
        agent_tools.execute_tool("not_registered", {"secret": "must-not-be-logged"})
    assert "unknown_tool" in caplog.text
    assert "must-not-be-logged" not in caplog.text


def test_invalid_json_emits_warning_without_raw_payload(monkeypatch, caplog):
    from conftest import assistant, tool_call

    responses = iter([
        assistant(tool_calls=[tool_call("c1", "broken", '{"secret":"not-closed"')]),
        assistant("done"),
    ])
    with caplog.at_level(logging.WARNING):
        result = agent_core.run_tool_call_loop(
            messages=[],
            call_llm=lambda _messages: next(responses),
            execute_tool=lambda **_kwargs: {"success": True},
        )
    assert result == "done"
    assert "invalid_tool_arguments_json" in caplog.text
    assert "not-closed" not in caplog.text
