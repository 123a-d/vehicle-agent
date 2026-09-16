
from __future__ import annotations
import json
import os

from openai import OpenAI

# ============================================================
# 1. 导入 Stage 7.5 已经完成的多工具系统
# ============================================================
#
# TOOLS
#     → 给 DeepSeek 看的 Tool Schema
#
# execute_tool()
#     → Python 统一 Tool Dispatcher
#
from agent_tools import (
    TOOLS,
    execute_tool,
)


# ============================================================
# 2. DeepSeek 配置
# ============================================================

BASE_URL = os.environ.get(
    "DEEPSEEK_BASE_URL",
    "https://api.deepseek.com",
)

MODEL_NAME = os.environ.get(
    "DEEPSEEK_MODEL",
    "deepseek-v4-flash",
)


# ============================================================
# 3. 创建 DeepSeek Client
# ============================================================

def create_client() -> OpenAI:
    """
    从环境变量读取 DeepSeek API Key。

    PowerShell：

        $env:DEEPSEEK_API_KEY="你的真实API Key"

    不要把 Key 直接写进源码。
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
# 4. 单次调用 DeepSeek
# ============================================================

def call_llm(
    client: OpenAI,
    messages: list,
):
    """
    每调用一次：

        messages
            ↓
        DeepSeek
            ↓
        assistant_message

    assistant_message 可能：

    情况 A：
        直接给最终答案

    情况 B：
        返回 tool_calls
    """

    response = (
        client
        .chat
        .completions
        .create(
            model=MODEL_NAME,

            messages=messages,

            # 把所有 Tool Schema 提供给 DeepSeek
            tools=TOOLS,

            stream=False,

            extra_body={
                "thinking": {
                    "type": "disabled"
                }
            },
        )
    )

    return (
        response
        .choices[0]
        .message
    )


# ============================================================
# 5. Agent Loop
# ============================================================

def run_agent_loop(
    user_message: str,
    max_steps: int = 10,
) -> str:
    """
    真正的 Agent Loop。

    工作过程：

        用户消息
            ↓
        LLM 判断
            ↓
        有 Tool Call？
         /       \
       有         没有
       ↓           ↓
    执行 Tool     返回答案
       ↓
    Tool Result
       ↓
    加回 messages
       ↓
    下一轮 LLM

    一直循环，
    直到 LLM 不再请求 Tool。
    """

    client = create_client()


    # ========================================================
    # 6. 初始化 messages
    # ========================================================

    messages = [

        {
            "role":
                "system",

            "content": (
                "你是一个车辆智能助手。"

                "你拥有以下工具："

                "1. recognize_vehicle："
                "识别车辆图片；"

                "2. query_vehicle_info："
                "根据 vehicle_id 查询车型详细信息；"

                "3. save_recognition_record："
                "把识别结果保存到数据库；"

                "4. query_recognition_history："
                "查询某个会话的识别历史。"

                "当用户要求识别车辆图片时，"
                "必须调用 recognize_vehicle，"
                "不能根据图片文件名猜测车型。"

                "如果用户同时要求："
                "识别车辆、查询详细信息、保存识别结果，"
                "必须按照工具之间的数据依赖顺序执行。"

                "第一步必须先调用 recognize_vehicle。"

                "只有在 recognize_vehicle 返回真实 vehicle_id 后，"
                "才能调用 query_vehicle_info。"

                "只有在获得真实识别结果和车型信息后，"
                "才能调用 save_recognition_record。"

                "如果后一个工具需要前一个工具的输出，"
                "不要提前猜测参数，"
                "必须先等待前一个工具返回结果。"

                "调用 save_recognition_record 时，"
                "vehicle_id、vehicle_name、confidence、status "
                "必须来自前面工具的真实返回结果，"
                "不能自己编造。"

                "session_id 使用用户提供的当前会话ID。"

                "如果某个工具返回 success=false，"
                "不要编造后续结果。"

                "如果错误导致后续任务无法继续，"
                "应该停止后续依赖操作，"
                "并在最终回答中说明失败原因。"

                "只有当用户要求的任务全部完成，"
                "并且不再需要调用任何工具时，"
                "才生成最终自然语言回答。"

                "对于视觉识别结果，"
                "使用“模型识别为”这样的表达，"
                "不要把模型预测描述成绝对事实。"
            ),
        },

        {
            "role":
                "user",

            "content":
                user_message,
        },
    ]


    # ========================================================
    # 7. Agent Loop
    # ========================================================
    #
    # 这里虽然用的是 for，
    # 本质上就是带上限的 while Agent Loop。
    #
    # max_steps 是保险，
    # 防止模型不停 Tool → Tool → Tool。
    # ========================================================

    for step in range(
        1,
        max_steps + 1,
    ):

        print(
            "\n"
            + "=" * 10
            + f" Agent Step {step} "
            + "=" * 10
        )


        # ----------------------------------------------------
        # 让 LLM 根据当前完整上下文
        # 决定下一步。
        # ----------------------------------------------------

        assistant_message = call_llm(
            client=client,
            messages=messages,
        )


        tool_calls = (
            assistant_message
            .tool_calls
        )


        # ====================================================
        # 8. 如果没有新的 Tool Call
        # ====================================================
        #
        # 说明 LLM 认为任务已经完成。
        # ====================================================

        if not tool_calls:

            print(
                "\n[Agent] 没有新的 Tool Call，"
                "任务结束。"
            )

            return (
                assistant_message.content
                or ""
            )


        # ====================================================
        # 9. 把 assistant 的 Tool Call 加入历史
        # ====================================================
        #
        # 第二轮 LLM 需要知道：
        #
        # “上一轮是我请求调用了这些工具。”
        #
        # ====================================================

        assistant_history = {

            "role":
                "assistant",

            "content":
                assistant_message.content,

            "tool_calls": [
                tool_call.model_dump()
                for tool_call
                in tool_calls
            ],
        }


        messages.append(
            assistant_history
        )


        # ====================================================
        # 10. 执行这一轮所有 Tool Calls
        # ====================================================

        for tool_call in tool_calls:

            tool_name = (
                tool_call
                .function
                .name
            )


            # ------------------------------------------------
            # DeepSeek 返回的 arguments
            # 通常是 JSON 字符串。
            #
            # 例如：
            #
            # '{"vehicle_id":"0009"}'
            #
            # json.loads() 后：
            #
            # {
            #     "vehicle_id": "0009"
            # }
            #
            # ------------------------------------------------

            try:

                arguments = json.loads(
                    tool_call
                    .function
                    .arguments
                )


            except json.JSONDecodeError as exc:

                tool_result = {

                    "tool_name":
                        tool_name,

                    "success":
                        False,

                    "error_type":
                        "InvalidToolArguments",

                    "error":
                        str(exc),
                }


            else:

                print(
                    f"\n[Agent] 调用工具："
                    f"{tool_name}"
                )


                print(
                    "[Agent] 参数："
                )


                print(
                    json.dumps(
                        arguments,
                        ensure_ascii=False,
                        indent=2,
                    )
                )


                # ============================================
                # 11. 统一 Tool Dispatcher
                # ============================================
                #
                # execute_tool 内部：
                #
                # tool_name
                #     ↓
                # TOOL_REGISTRY
                #     ↓
                # 找到真正 Python 函数
                #     ↓
                # **arguments
                #     ↓
                # 执行
                #
                # ============================================

                tool_result = execute_tool(
                    tool_name=tool_name,
                    arguments=arguments,
                )


            print(
                "\n[Tool Result]"
            )


            print(
                json.dumps(
                    tool_result,
                    ensure_ascii=False,
                    indent=2,
                )
            )


            # =================================================
            # 12. 把 Tool Result 回填给 LLM
            # =================================================
            #
            # tool_call_id 必须与前面的 Tool Call 对应。
            #
            # =================================================

            messages.append(
                {
                    "role":
                        "tool",

                    "tool_call_id":
                        tool_call.id,

                    "content":
                        json.dumps(
                            tool_result,
                            ensure_ascii=False,
                        ),
                }
            )


        # ====================================================
        # 13. 这里故意不 return
        # ====================================================
        #
        # 执行 Tool 后，
        # 回到循环顶部。
        #
        # 下一轮 DeepSeek 会看到：
        #
        # user
        # assistant(tool call)
        # tool(result)
        #
        # 然后决定：
        #
        # 还要继续调用 Tool？
        # 还是已经可以回答用户？
        #
        # ====================================================


    # ========================================================
    # 14. 最大步数保护
    # ========================================================

    return (
        "Agent 已达到最大执行步数，"
        "为防止无限工具循环，"
        "本次任务已停止。"
    )


# ============================================================
# 15. Stage 7.7 三工具连续调用测试
# ============================================================

def main():

    print("=" * 76)

    print(
        "Stage 7.7 · "
        "Multi-Tool Chaining Test"
    )

    print("=" * 76)


    # ========================================================
    # 测试任务
    # ========================================================
    #
    # 这个请求同时包含三个目标：
    #
    # 1. 识别车辆图片
    # 2. 查询详细车型信息
    # 3. 保存识别结果
    #
    # 理想 Tool Chain：
    #
    # recognize_vehicle
    #       ↓
    # query_vehicle_info
    #       ↓
    # save_recognition_record
    #       ↓
    # Final Answer
    #
    # ========================================================

    user_message = (
        "当前会话ID是 session_001。"

        "请识别这张车辆图片："
        "D:\\car_project\\0009_0030.jpg。"

        "识别完成后，"
        "请查询该车型的详细信息，"

        "然后把这次识别结果"
        "保存到当前会话中。"
    )


    print(
        f"\nUser: {user_message}"
    )


    try:

        answer = run_agent_loop(
            user_message=user_message,
            max_steps=10,
        )


        print(
            "\n[Final Answer]"
        )


        print(
            answer
        )


    except Exception as exc:

        print(
            "\nAgent运行失败："
        )


        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )


if __name__ == "__main__":
    main()
