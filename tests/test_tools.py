import sqlite3
import agent_tools
import vehicle_info_tool
import recognition_record_tool
import recognition_history_tool

EXPECTED = {"recognize_vehicle", "query_vehicle_info", "save_recognition_record", "query_recognition_history"}

def test_four_tools_registered_and_exposed():
    assert set(agent_tools.TOOL_REGISTRY) == EXPECTED
    assert {x["function"]["name"] for x in agent_tools.TOOLS} == EXPECTED

def test_execute_tool_normal(monkeypatch):
    monkeypatch.setitem(agent_tools.TOOL_REGISTRY, "fake", lambda value: {"success": True, "value": value})
    assert agent_tools.execute_tool("fake", {"value": 7}) == {"success": True, "value": 7}

def test_execute_tool_unknown():
    result = agent_tools.execute_tool("missing", {})
    assert result["success"] is False and result["error_type"] == "UnknownTool"

def test_execute_tool_argument_error(monkeypatch):
    monkeypatch.setitem(agent_tools.TOOL_REGISTRY, "needs_value", lambda value: {})
    result = agent_tools.execute_tool("needs_value", {})
    assert result["error_type"] == "ToolArgumentError"

def test_execute_tool_exception(monkeypatch):
    def broken(): raise RuntimeError("boom")
    monkeypatch.setitem(agent_tools.TOOL_REGISTRY, "broken", broken)
    result = agent_tools.execute_tool("broken", {})
    assert result["error_type"] == "RuntimeError" and result["error"] == "boom"

def test_database_tools_use_temporary_database(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    connection = sqlite3.connect(path)
    connection.executescript("""
      CREATE TABLE vehicle_info (vehicle_id TEXT PRIMARY KEY, vehicle_name TEXT NOT NULL, vehicle_type TEXT, vehicle_subtype TEXT);
      CREATE TABLE recognition_records (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, image_path TEXT NOT NULL, vehicle_id TEXT, vehicle_name TEXT, confidence REAL, status TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
      INSERT INTO vehicle_info VALUES ('0009','Volvo S90','Sedan','Upper Mid-size');
    """)
    connection.commit(); connection.close()
    monkeypatch.setattr(vehicle_info_tool, "DATABASE_PATH", path)
    monkeypatch.setattr(recognition_record_tool, "DATABASE_PATH", path)
    monkeypatch.setattr(recognition_history_tool, "DATABASE_PATH", path)
    assert vehicle_info_tool.query_vehicle_info("0009")["vehicle_name"] == "Volvo S90"
    saved = recognition_record_tool.save_recognition_record("session_test", "x.jpg", "0009", "Volvo S90", .91, "recognized")
    assert saved["success"] is True
    history = recognition_history_tool.query_recognition_history("session_test")
    assert history["success"] is True and history["count"] == 1
