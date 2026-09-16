"""
Stage 7.5
多 Tool Schema + Tool Registry

本文件负责两件事：

1. TOOLS
   把有哪些工具、工具做什么、需要什么参数
   告诉 DeepSeek。

2. TOOL_REGISTRY
   把 DeepSeek 返回的工具名称
   映射到真正的 Python 函数。

注意：

Tool Schema ≠ Python函数

Tool Schema:
    给 LLM 看的“工具说明书”

Tool Registry:
    给 Python 程序用的“工具登记表”
"""

from __future__ import annotations

from typing import Any, Callable


# ============================================================
# 1. 导入四个真实 Tool
# ============================================================

from vehicle_tool import (
    recognize_vehicle,
)

from vehicle_info_tool import (
    query_vehicle_info,
)

from recognition_record_tool import (
    save_recognition_record,
)

from recognition_history_tool import (
    query_recognition_history,
)


# ============================================================
# 2. Tool Schema
# ============================================================
#
# TOOLS 会发送给 DeepSeek。
#
# DeepSeek 根据：
#
#     name
#     description
#     parameters
#
# 判断：
#
#     该不该调用工具？
#     调哪个？
#     应该传什么参数？
#
# ============================================================

TOOLS = [

    # --------------------------------------------------------
    # Tool 1：车辆视觉识别
    # --------------------------------------------------------
    {
        "type": "function",

        "function": {

            "name":
                "recognize_vehicle",

            "description": (
                "识别本地车辆图片中的车型。"
                "当用户要求识别车辆图片、判断车型、"
                "查看模型置信度或候选车型时使用。"
                "必须使用该工具进行图片车型识别，"
                "不能根据图片文件名直接猜测车型。"
            ),

            "parameters": {

                "type": "object",

                "properties": {

                    "image_path": {
                        "type": "string",
                        "description":
                            "待识别车辆图片的本地文件路径。",
                    },

                    "top_k": {
                        "type": "integer",
                        "description":
                            "返回概率最高的前K个候选车型，默认3。",
                        "minimum": 1,
                        "maximum": 10,
                    },

                    "uncertainty_threshold": {
                        "type": "number",
                        "description":
                            "业务层置信度阈值，默认0.40。",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },

                "required": [
                    "image_path"
                ],
            },
        },
    },


    # --------------------------------------------------------
    # Tool 2：车型信息查询
    # --------------------------------------------------------
    {
        "type": "function",

        "function": {

            "name":
                "query_vehicle_info",

            "description": (
                "根据 vehicle_id 从 SQLite 数据库查询车型详细信息，"
                "包括车型名称、车辆类型和细分类型。"
                "当已经知道 vehicle_id，"
                "并且需要获得车型详细信息时使用。"
            ),

            "parameters": {

                "type": "object",

                "properties": {

                    "vehicle_id": {
                        "type": "string",
                        "description":
                            "车型ID，例如 0009。",
                    },
                },

                "required": [
                    "vehicle_id"
                ],
            },
        },
    },


    # --------------------------------------------------------
    # Tool 3：保存车辆识别记录
    # --------------------------------------------------------
    {
        "type": "function",

        "function": {

            "name":
                "save_recognition_record",

            "description": (
                "把一次车辆识别结果保存到 SQLite 数据库。"
                "当用户要求保存、记录或记住本次识别结果时使用。"
            ),

            "parameters": {

                "type": "object",

                "properties": {

                    "session_id": {
                        "type": "string",
                        "description":
                            "当前会话ID，例如 session_001。",
                    },

                    "image_path": {
                        "type": "string",
                        "description":
                            "本次识别图片的本地路径。",
                    },

                    "vehicle_id": {
                        "type": "string",
                        "description":
                            "视觉识别得到的车型ID。",
                    },

                    "vehicle_name": {
                        "type": "string",
                        "description":
                            "车型名称。",
                    },

                    "confidence": {
                        "type": "number",
                        "description":
                            "视觉模型给出的置信度。",
                    },

                    "status": {
                        "type": "string",
                        "description":
                            "识别状态，例如 recognized、uncertain 或 unknown。",
                    },
                },

                "required": [
                    "session_id",
                    "image_path",
                    "vehicle_id",
                    "vehicle_name",
                    "confidence",
                    "status",
                ],
            },
        },
    },


    # --------------------------------------------------------
    # Tool 4：查询车辆识别历史
    # --------------------------------------------------------
    {
        "type": "function",

        "function": {

            "name":
                "query_recognition_history",

            "description": (
                "根据 session_id 查询当前会话之前保存的车辆识别历史。"
                "当用户询问刚才识别了什么、"
                "识别历史、最近识别记录时使用。"
            ),

            "parameters": {

                "type": "object",

                "properties": {

                    "session_id": {
                        "type": "string",
                        "description":
                            "需要查询的会话ID。",
                    },

                    "limit": {
                        "type": "integer",
                        "description":
                            "最多返回多少条记录，默认10。",
                        "minimum": 1,
                        "maximum": 100,
                    },
                },

                "required": [
                    "session_id"
                ],
            },
        },
    },
]


# ============================================================
# 3. Tool Registry
# ============================================================
#
# 这是 Stage 7.5 最重要的新概念。
#
# 它本质就是一个 Python dict：
#
#     工具名字符串
#         ↓
#     真正的 Python 函数
#
# 例如：
#
#     "recognize_vehicle"
#             ↓
#     recognize_vehicle
#
# 注意右边没有括号：
#
#     recognize_vehicle
#
# 表示“函数本身”
#
# 而不是：
#
#     recognize_vehicle()
#
# 后者代表“现在立刻执行函数”。
#
# ============================================================

TOOL_REGISTRY: dict[
    str,
    Callable[..., dict],
] = {

    "recognize_vehicle":
        recognize_vehicle,

    "query_vehicle_info":
        query_vehicle_info,

    "save_recognition_record":
        save_recognition_record,

    "query_recognition_history":
        query_recognition_history,
}


# ============================================================
# 4. 通用 Tool Dispatcher
# ============================================================

def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
) -> dict:
    """
    根据 DeepSeek 返回的 tool_name，
    自动查找并执行真正的 Python Tool。

    例如：

        tool_name =
            "query_vehicle_info"

        arguments =
            {
                "vehicle_id": "0009"
            }

    第一步：

        TOOL_REGISTRY.get(
            "query_vehicle_info"
        )

    得到：

        query_vehicle_info

    第二步：

        tool_function(**arguments)

    实际相当于：

        query_vehicle_info(
            vehicle_id="0009"
        )
    """

    # --------------------------------------------------------
    # 根据字符串名称找到函数
    # --------------------------------------------------------

    tool_function = (
        TOOL_REGISTRY.get(
            tool_name
        )
    )


    # --------------------------------------------------------
    # 如果没有注册这个工具
    # --------------------------------------------------------

    if tool_function is None:

        return {
            "tool_name":
                tool_name,

            "success":
                False,

            "error_type":
                "UnknownTool",

            "error":
                f"未注册工具：{tool_name}",
        }


    # --------------------------------------------------------
    # 调用真正的函数
    # --------------------------------------------------------
    #
    # **arguments
    #
    # 是一个非常重要的 Python 语法。
    #
    # 例如：
    #
    # arguments =
    # {
    #     "vehicle_id": "0009"
    # }
    #
    # 那么：
    #
    # tool_function(**arguments)
    #
    # 相当于：
    #
    # tool_function(
    #     vehicle_id="0009"
    # )
    #
    # --------------------------------------------------------

    try:

        result = tool_function(
            **arguments
        )

        return result


    except TypeError as exc:

        # 最常见情况：
        #
        # DeepSeek 参数名称或数量
        # 与 Python 函数不匹配。

        return {
            "tool_name":
                tool_name,

            "success":
                False,

            "error_type":
                "ToolArgumentError",

            "error":
                str(exc),
        }


    except Exception as exc:

        # 最后一层保险。
        #
        # 避免某个 Tool 出错后
        # 直接把整个 Agent 程序搞崩。

        return {
            "tool_name":
                tool_name,

            "success":
                False,

            "error_type":
                type(exc).__name__,

            "error":
                str(exc),
        }


# ============================================================
# 5. 简单自检
# ============================================================

def main():
    """
    暂时不用 DeepSeek，
    直接测试 Registry 是否工作。
    """

    print(
        "当前已注册 Tool："
    )

    for tool_name in TOOL_REGISTRY:

        print(
            f"  - {tool_name}"
        )


    print(
        "\n测试 query_vehicle_info："
    )

    result = execute_tool(
        tool_name="query_vehicle_info",

        arguments={
            "vehicle_id":
                "0009"
        },
    )

    print(
        result
    )


if __name__ == "__main__":
    main()