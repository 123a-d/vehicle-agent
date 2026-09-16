"""
Stage 6 · Step 2
文件名：deepseek_vehicle_agent.py

目标：
完成第一次真正的 Tool Calling：

用户
↓
DeepSeek 判断是否调用工具
↓
返回 tool_call
↓
Python 执行 recognize_vehicle()
↓
Tool Result 回填 messages
↓
再次请求 DeepSeek
↓
DeepSeek 生成最终回答
"""

from __future__ import annotations

import argparse
import json
import os

from openai import OpenAI

from vehicle_tool import recognize_vehicle


BASE_URL = os.environ.get(
    "DEEPSEEK_BASE_URL",
    "https://api.deepseek.com",
)
MODEL_NAME = os.environ.get(
    "DEEPSEEK_MODEL",
    "deepseek-v4-flash",
)


# ============================================================
# 1. 创建 DeepSeek Client
# ============================================================

def create_client() -> OpenAI:
    """
    从环境变量读取 DeepSeek API Key。
    """

    api_key = os.environ.get(
        "DEEPSEEK_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "没有检测到环境变量 DEEPSEEK_API_KEY。"
        )

    return OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
    )


# ============================================================
# 2. Tool Schema
# ============================================================
#
# 这不是 Python 函数本身。
#
# 它只是“告诉 DeepSeek：
# 我们有一个什么工具，它需要什么参数”。
#
# DeepSeek 会看到 schema，
# 但不会直接执行本地函数。
# ============================================================

TOOLS = [
    {
        "type": "function",

        "function": {
            "name": "recognize_vehicle",

            "description": (
                "识别本地车辆图片。"
                "当用户要求判断车辆车型、"
                "查看车型候选或模型置信度时使用。"
                "不要根据文件名猜车型。"
            ),

            "parameters": {
                "type": "object",

                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": (
                            "待识别图片的本地路径"
                        ),
                    },

                    "top_k": {
                        "type": "integer",
                        "description": (
                            "返回概率最高的前K个候选"
                        ),
                        "minimum": 1,
                        "maximum": 10,
                    },

                    "uncertainty_threshold": {
                        "type": "number",
                        "description": (
                            "业务层置信度阈值，"
                            "建议使用0.40"
                        ),
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },

                "required": [
                    "image_path"
                ],
            },
        },
    }
]


# ============================================================
# 3. 请求 DeepSeek
# ============================================================

def send_messages(
    client: OpenAI,
    messages: list,
):
    """
    和 Stage 6 第一步相比，
    最重要的新参数是 tools=TOOLS。
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=TOOLS,
        stream=False,
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        },
    )

    return (
        response
        .choices[0]
        .message
    )


# ============================================================
# 4. Tool Dispatcher
# ============================================================
#
# Dispatcher = 工具调度器。
#
# DeepSeek 只会返回：
#   tool name
#   arguments
#
# 真正决定调用哪个 Python 函数，
# 是我们自己的代码。
# ============================================================

def execute_tool(
    tool_name: str,
    arguments: dict,
) -> dict:
    """
    根据工具名调用真实 Python 函数。
    """

    if tool_name == "recognize_vehicle":

        return recognize_vehicle(
            image_path=arguments.get(
                "image_path"
            ),

            top_k=arguments.get(
                "top_k",
                3,
            ),

            uncertainty_threshold=(
                arguments.get(
                    "uncertainty_threshold",
                    0.40,
                )
            ),
        )

    return {
        "success": False,
        "error_type": "UnknownTool",
        "error": (
            f"未注册工具：{tool_name}"
        ),
    }


# ============================================================
# 5. Agent 单轮流程
# ============================================================

def run_agent(
    user_message: str,
) -> str:
    """
    最基础的 Agent + Tool Calling 循环。

    1. user -> DeepSeek
    2. DeepSeek 判断是否调用 Tool
    3. Python 执行 Tool
    4. Tool Result 回填 messages
    5. 再请求 DeepSeek
    6. 得到最终回答
    """

    client = create_client()

    messages = [
        {
            "role": "system",
            "content": (
                "你是车辆智能助手。"
                "当用户要求识别车辆图片时，"
                "必须调用 recognize_vehicle 工具，"
                "不能根据文件名直接猜车型。"
                "如果工具状态为 uncertain，"
                "要明确表达不确定性。"
                "如果工具执行失败，"
                "要说明失败原因。"
            ),
        },

        {
            "role": "user",
            "content": user_message,
        },
    ]

    # --------------------------------------------------------
    # 第一次请求：
    # 让 DeepSeek 决定是否需要工具。
    # --------------------------------------------------------

    assistant_message = send_messages(
        client=client,
        messages=messages,
    )

    # --------------------------------------------------------
    # 如果模型没有要求调用工具，
    # 直接返回普通回答。
    # --------------------------------------------------------

    if not assistant_message.tool_calls:

        return (
            assistant_message.content
            or ""
        )

    # --------------------------------------------------------
    # 把带 tool_calls 的 assistant 消息
    # 放回对话历史。
    # --------------------------------------------------------

    messages.append(
        assistant_message
    )

    # --------------------------------------------------------
    # 一个 assistant 消息可能包含多个 tool call。
    # 当前只有一个工具，
    # 但仍按列表处理，
    # 为后续多工具阶段做准备。
    # --------------------------------------------------------

    for tool_call in (
        assistant_message.tool_calls
    ):

        tool_name = (
            tool_call
            .function
            .name
        )

        # arguments 是 JSON 字符串，
        # 先转换成 Python dict。
        try:

            arguments = json.loads(
                tool_call
                .function
                .arguments
            )

        except json.JSONDecodeError as exc:

            tool_result = {
                "success": False,
                "error_type": (
                    "InvalidToolArguments"
                ),
                "error": str(exc),
            }

        else:

            print(
                "\n[Agent] DeepSeek 请求调用工具："
                f"{tool_name}"
            )

            print(
                "[Agent] Tool arguments："
            )

            print(
                json.dumps(
                    arguments,
                    ensure_ascii=False,
                    indent=2,
                )
            )

            # 真正执行本地 Python 函数。
            tool_result = execute_tool(
                tool_name=tool_name,
                arguments=arguments,
            )

        print(
            "\n[Tool] 执行结果："
        )

        print(
            json.dumps(
                tool_result,
                ensure_ascii=False,
                indent=2,
            )
        )

        # ----------------------------------------------------
        # 把工具结果以 role="tool" 加回 messages。
        #
        # tool_call_id 必须对应刚才那个 tool call。
        # ----------------------------------------------------

        messages.append(
            {
                "role": "tool",
                "tool_call_id": (
                    tool_call.id
                ),
                "content": json.dumps(
                    tool_result,
                    ensure_ascii=False,
                ),
            }
        )

    # --------------------------------------------------------
    # 第二次请求：
    # 此时 DeepSeek 已经看到真实 Tool Result，
    # 才能生成最终自然语言回答。
    # --------------------------------------------------------

    final_message = send_messages(
        client=client,
        messages=messages,
    )

    return (
        final_message.content
        or ""
    )


# ============================================================
# 6. 命令行测试
# ============================================================

def main():
    """
    示例：

        python deepseek_vehicle_agent.py "D:\\car_project\\0009_0030.jpg"
    """

    parser = argparse.ArgumentParser(
        description=(
            "Stage 6 · DeepSeek + Visual Tool Calling"
        )
    )

    parser.add_argument(
        "image_path",
        help="待识别车辆图片路径",
    )

    args = parser.parse_args()

    user_message = (
        "请识别这张车辆图片是什么车型，"
        "并告诉我模型置信度："
        f"{args.image_path}"
    )

    print("=" * 76)
    print(
        "Stage 6 · DeepSeek + Visual Tool Calling"
    )
    print("=" * 76)

    print(
        f"\nUser: {user_message}"
    )

    try:

        final_answer = run_agent(
            user_message
        )

        print(
            "\n[Agent Final Answer]"
        )

        print(
            final_answer
        )

    except Exception as exc:

        print(
            "\nAgent运行失败："
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )


if __name__ == "__main__":
    main()
