"""统一 Tool Definition，并生成 DeepSeek Schema 与 Python Registry。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable

from logging_config import get_logger
from recognition_history_tool import query_recognition_history
from recognition_record_tool import save_recognition_record
from vehicle_info_tool import query_vehicle_info
from vehicle_tool import recognize_vehicle

logger = get_logger(__name__)

@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., dict]

    def deepseek_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

TOOL_DEFINITIONS = (
    ToolDefinition(
        name="recognize_vehicle",
        description=("识别本地车辆图片中的车型。当用户要求识别车辆图片、判断车型、查看模型置信度或候选车型时使用。"
                     "必须使用该工具进行图片车型识别，不能根据图片文件名直接猜测车型。"),
        parameters={"type": "object", "properties": {
            "image_path": {"type": "string", "description": "待识别车辆图片的本地文件路径。"},
            "top_k": {"type": "integer", "description": "返回概率最高的前K个候选车型，默认3。", "minimum": 1, "maximum": 10},
            "uncertainty_threshold": {"type": "number", "description": "业务层置信度阈值，默认0.40。", "minimum": 0.0, "maximum": 1.0},
        }, "required": ["image_path"]},
        handler=recognize_vehicle,
    ),
    ToolDefinition(
        name="query_vehicle_info",
        description="根据 vehicle_id 从 SQLite 数据库查询车型详细信息，包括车型名称、车辆类型和细分类型。当已经知道 vehicle_id，并且需要获得车型详细信息时使用。",
        parameters={"type": "object", "properties": {"vehicle_id": {"type": "string", "description": "车型ID，例如 0009。"}}, "required": ["vehicle_id"]},
        handler=query_vehicle_info,
    ),
    ToolDefinition(
        name="save_recognition_record",
        description="把一次车辆识别结果保存到 SQLite 数据库。当用户要求保存、记录或记住本次识别结果时使用。",
        parameters={"type": "object", "properties": {
            "session_id": {"type": "string", "description": "当前会话ID，例如 session_001。"},
            "image_path": {"type": "string", "description": "本次识别图片的本地路径。"},
            "vehicle_id": {"type": "string", "description": "视觉识别得到的车型ID。"},
            "vehicle_name": {"type": "string", "description": "车型名称。"},
            "confidence": {"type": "number", "description": "视觉模型给出的置信度。"},
            "status": {"type": "string", "description": "识别状态，例如 recognized、uncertain 或 unknown。"},
        }, "required": ["session_id", "image_path", "vehicle_id", "vehicle_name", "confidence", "status"]},
        handler=save_recognition_record,
    ),
    ToolDefinition(
        name="query_recognition_history",
        description="根据 session_id 查询当前会话之前保存的车辆识别历史。当用户询问刚才识别了什么、识别历史、最近识别记录时使用。",
        parameters={"type": "object", "properties": {
            "session_id": {"type": "string", "description": "需要查询的会话ID。"},
            "limit": {"type": "integer", "description": "最多返回多少条记录，默认10。", "minimum": 1, "maximum": 100},
        }, "required": ["session_id"]},
        handler=query_recognition_history,
    ),
)

TOOLS = [definition.deepseek_schema() for definition in TOOL_DEFINITIONS]
TOOL_REGISTRY: dict[str, Callable[..., dict]] = {
    definition.name: definition.handler for definition in TOOL_DEFINITIONS
}

def execute_tool(tool_name: str, arguments: dict[str, Any]) -> dict:
    tool_function = TOOL_REGISTRY.get(tool_name)
    if tool_function is None:
        logger.warning("unknown_tool tool=%s", tool_name)
        return {"tool_name": tool_name, "success": False, "error_type": "UnknownTool", "error": f"未注册工具：{tool_name}"}
    try:
        return tool_function(**arguments)
    except TypeError as exc:
        logger.warning("tool_argument_error tool=%s error=%s", tool_name, exc)
        return {"tool_name": tool_name, "success": False, "error_type": "ToolArgumentError", "error": str(exc)}
    except Exception as exc:
        logger.error("tool_execution_error tool=%s error_type=%s", tool_name, type(exc).__name__)
        return {"tool_name": tool_name, "success": False, "error_type": type(exc).__name__, "error": str(exc)}

def main():
    print("当前已注册 Tool：")
    for tool_name in TOOL_REGISTRY:
        print(f"  - {tool_name}")

if __name__ == "__main__":
    main()
