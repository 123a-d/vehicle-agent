"""Vehicle Agent 共用的 Tool Calling 执行循环。"""
from __future__ import annotations
import json
from collections.abc import Callable
from typing import Any
from logging_config import get_logger

logger = get_logger(__name__)

def run_tool_call_loop(*, messages: list[dict[str, Any]], call_llm: Callable[[list[dict[str, Any]]], Any], execute_tool: Callable[..., dict], max_steps: int = 10, prepare_arguments: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None, persist_final_answer: bool = False, completion_message: str = "任务结束。", max_steps_message: str = "Agent 已达到最大执行步数，为防止无限工具循环，本次任务已停止。") -> str:
    """执行公共的 LLM → Tool → Result 回填循环。"""
    for step in range(1, max_steps + 1):
        if step == 1:
            logger.info("agent_loop_started max_steps=%s", max_steps)
        try:
            assistant_message = call_llm(messages)
        except Exception as exc:
            logger.error("llm_call_failed step=%s error_type=%s", step, type(exc).__name__)
            raise
        tool_calls = assistant_message.tool_calls
        if not tool_calls:
            final_answer = assistant_message.content or ""
            if persist_final_answer:
                messages.append({"role": "assistant", "content": final_answer})
            logger.info("agent_loop_finished step=%s", step)
            return final_answer
        messages.append({"role": "assistant", "content": assistant_message.content, "tool_calls": [call.model_dump() for call in tool_calls]})
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as exc:
                logger.warning("invalid_tool_arguments_json tool=%s", tool_name)
                tool_result = {"tool_name": tool_name, "success": False, "error_type": "InvalidToolArguments", "error": str(exc)}
            else:
                if prepare_arguments is not None:
                    arguments = prepare_arguments(tool_name, arguments)
                logger.info("tool_call tool=%s step=%s", tool_name, step)
                tool_result = execute_tool(tool_name=tool_name, arguments=arguments)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(tool_result, ensure_ascii=False)})
    logger.warning("agent_loop_max_steps_reached max_steps=%s", max_steps)
    return max_steps_message
