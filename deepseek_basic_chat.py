"""
Stage 6 · Step 1
文件名：deepseek_basic_chat.py

目标：
先验证 Python -> DeepSeek API -> LLM 返回回答。
等这一步跑通后，再接 Stage 5 的 recognize_vehicle() Tool。
"""

from __future__ import annotations

import os
from openai import OpenAI


BASE_URL = os.environ.get(
    "DEEPSEEK_BASE_URL",
    "https://api.deepseek.com",
)
MODEL_NAME = os.environ.get(
    "DEEPSEEK_MODEL",
    "deepseek-v4-flash",
)


def create_client() -> OpenAI:
    """
    创建 DeepSeek Client。

    API Key 不写死在代码里，
    而是从环境变量 DEEPSEEK_API_KEY 读取。
    """

    api_key = os.environ.get("DEEPSEEK_API_KEY")

    if not api_key:
        raise RuntimeError(
            "没有检测到环境变量 DEEPSEEK_API_KEY。\n"
            "请先在 PowerShell 中运行：\n"
            '$env:DEEPSEEK_API_KEY="你的DeepSeek API Key"'
        )

    return OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
    )


def ask_deepseek(user_message: str) -> str:
    """
    发送最基础的聊天请求。

    messages 目前只有两种角色：
    - system：规定助手角色和规则
    - user：用户真正提出的问题

    Stage 6 后面做 Tool Calling 时，
    还会继续出现 assistant 和 tool。
    """

    client = create_client()

    messages = [
        {
            "role": "system",
            "content": (
                "你是一个车辆智能助手。"
                "请使用简洁、准确的中文回答。"
            ),
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        stream=False,
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        },
    )

    message = response.choices[0].message

    return message.content or ""


def main():
    """
    运行：
        python deepseek_basic_chat.py
    """

    print("=" * 70)
    print("Stage 6 · DeepSeek Basic Chat Test")
    print("=" * 70)

    question = "请用一句话解释什么是 Tool Calling。"

    print(f"\nUser: {question}")

    try:
        answer = ask_deepseek(question)
        print(f"\nDeepSeek: {answer}")

    except Exception as exc:
        print("\n调用失败：")
        print(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
