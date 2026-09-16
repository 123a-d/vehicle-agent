"""
Stage 7.8
Session 多轮会话 Agent

目标：
1. 同一个 session_id 下连续多轮聊天
2. 保留 messages 对话上下文
3. 继续支持多工具 Agent Loop
4. 可以查询当前 Session 的历史识别记录
"""

from __future__ import annotations

import json
import os

from openai import OpenAI

from agent_tools import (
    TOOLS,
    execute_tool,
)


# ============================================================
# 1. DeepSeek 配置
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
# 2. 创建 Client
# ============================================================

def create_client() -> OpenAI:

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
# 3. Session 类
# ============================================================
#
# 一个 Session 最核心的两样东西：
#
# 1. session_id
# 2. messages
#
# session_id：
#     标识“这是哪一次会话”
#
# messages：
#     保存整个对话历史
#
# ============================================================

class AgentSession:

    def __init__(
        self,
        session_id: str,
    ):
        """
        创建一个新的 Agent Session。
        """

        self.session_id = (
            str(session_id)
            .strip()
        )

        self.client = create_client()

        # ----------------------------------------
        # messages 只在创建 Session 时初始化一次
        #
        # 后续每轮用户消息都追加进去，
        # 不会重新创建。
        # ----------------------------------------

        self.messages = [

            {
                "role":
                    "system",

                "content": (
                    "你是一个车辆智能助手。"

                    "当前会话ID为："
                    f"{self.session_id}。"

                    "你拥有以下工具："

                    "1. recognize_vehicle："
                    "识别车辆图片；"

                    "2. query_vehicle_info："
                    "根据 vehicle_id 查询车型详细信息；"

                    "3. save_recognition_record："
                    "保存识别记录；"

                    "4. query_recognition_history："
                    "查询当前会话识别历史。"

                    "当用户要求识别车辆图片时，"
                    "必须使用 recognize_vehicle，"
                    "不能根据文件名猜车型。"

                    "如果用户要求识别、查询详情并保存，"
                    "必须按照："
                    "recognize_vehicle → "
                    "query_vehicle_info → "
                    "save_recognition_record "
                    "的依赖顺序完成。"

                    "调用 save_recognition_record 或 "
                    "query_recognition_history 时，"
                    f"当前 session_id 固定使用 "
                    f"{self.session_id}。"

                    "不要编造工具结果。"

                    "如果用户说“刚才”“之前”“最近识别的”，"
                    "应优先结合当前会话上下文，"
                    "必要时调用 query_recognition_history。"

                    "只有不再需要工具时，"
                    "才给最终自然语言回答。"
                ),
            }
        ]


    # ========================================================
    # 4. 单次 LLM 请求
    # ========================================================

    def call_llm(self):

        response = (
            self.client
            .chat
            .completions
            .create(
                model=MODEL_NAME,

                messages=self.messages,

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


    # ========================================================
    # 5. 处理一轮用户消息
    # ========================================================

    def chat(
        self,
        user_message: str,
        max_steps: int = 10,
    ) -> str:
        """
        处理用户的一轮消息。

        注意：

        和之前最大区别是：

            self.messages

        不会在每轮重新初始化。

        所以历史会一直保留。
        """

        # ----------------------------------------
        # 把当前用户消息加入 Session 历史
        # ----------------------------------------

        self.messages.append(
            {
                "role":
                    "user",

                "content":
                    user_message,
            }
        )


        # ====================================================
        # 6. Agent Loop
        # ====================================================

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


            assistant_message = (
                self.call_llm()
            )


            tool_calls = (
                assistant_message
                .tool_calls
            )


            # ------------------------------------------------
            # 没有 Tool Call
            # ------------------------------------------------

            if not tool_calls:

                final_answer = (
                    assistant_message.content
                    or ""
                )

                # --------------------------------------------
                # 非常重要：
                #
                # 最终 assistant 回答也必须加入 messages。
                #
                # 这样下一轮用户说：
                #
                # “你刚才说的那个车型呢？”
                #
                # LLM 才能看到自己上一轮说过什么。
                # --------------------------------------------

                self.messages.append(
                    {
                        "role":
                            "assistant",

                        "content":
                            final_answer,
                    }
                )


                print(
                    "\n[Agent] 没有新的 Tool Call，"
                    "本轮结束。"
                )


                return final_answer


            # =================================================
            # 7. 保存 assistant Tool Call
            # =================================================

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


            self.messages.append(
                assistant_history
            )


            # =================================================
            # 8. 执行 Tool
            # =================================================

            for tool_call in tool_calls:

                tool_name = (
                    tool_call
                    .function
                    .name
                )


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

                    # =========================================
                    # 9. 自动补 Session ID
                    # =========================================
                    #
                    # 为了避免 LLM 自己乱生成 session_id，
                    # 对需要 Session 的工具，
                    # Python 侧直接覆盖。
                    #
                    # 这比完全相信 LLM 更可靠。
                    # =========================================

                    if tool_name in {
                        "save_recognition_record",
                        "query_recognition_history",
                    }:

                        arguments[
                            "session_id"
                        ] = self.session_id


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


                # --------------------------------------------
                # Tool Result 加回 Session messages
                # --------------------------------------------

                self.messages.append(
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


        return (
            "Agent 达到最大执行步数，"
            "本轮已停止。"
        )


# ============================================================
# 10. 命令行多轮聊天
# ============================================================

def main():

    print("=" * 76)

    print(
        "Stage 7.8 · Session Multi-Turn Agent"
    )

    print("=" * 76)


    # ----------------------------------------
    # 现在先手工固定一个 Session ID。
    #
    # 后面 Web 阶段会自动生成。
    # ----------------------------------------

    session_id = (
        "session_001"
    )


    session = AgentSession(
        session_id=session_id
    )


    print(
        f"\n当前 Session：{session_id}"
    )


    print(
        "输入 exit / quit 退出程序。"
    )


    # ========================================================
    # 11. 真正多轮交互
    # ========================================================

    while True:

        user_message = input(
            "\nUser: "
        ).strip()


        if not user_message:
            continue


        if user_message.lower() in {
            "exit",
            "quit",
        }:

            print(
                "\nSession 已结束。"
            )

            break


        try:

            answer = session.chat(
                user_message=user_message,
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
