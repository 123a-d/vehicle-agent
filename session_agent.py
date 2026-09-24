"""带持久消息历史的 Vehicle Agent Session。"""
from __future__ import annotations
from openai import OpenAI
from agent_core import run_tool_call_loop
from agent_tools import TOOLS, execute_tool
from config import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, get_deepseek_api_key
from logging_config import get_logger

BASE_URL = DEEPSEEK_BASE_URL
MODEL_NAME = DEEPSEEK_MODEL
logger = get_logger(__name__)

def create_client() -> OpenAI:
    return OpenAI(api_key=get_deepseek_api_key(), base_url=BASE_URL)

class AgentSession:
    def __init__(self, session_id: str):
        self.session_id = str(session_id).strip()
        self.client = create_client()
        self.messages = [{"role": "system", "content": self._system_prompt()}]
        logger.info("session_created session_id=%s", self.session_id)

    def _system_prompt(self) -> str:
        return (
            "你是一个车辆智能助手。"
            f"当前会话ID为：{self.session_id}。"
            "你拥有以下工具："
            "1. recognize_vehicle：识别车辆图片；"
            "2. query_vehicle_info：根据 vehicle_id 查询车型详细信息；"
            "3. save_recognition_record：保存识别记录；"
            "4. query_recognition_history：查询当前会话识别历史。"
            "当用户要求识别车辆图片时，必须使用 recognize_vehicle，不能根据文件名猜车型。"
            "如果用户要求识别、查询详情并保存，必须按照：recognize_vehicle → query_vehicle_info → save_recognition_record 的依赖顺序完成。"
            "调用 save_recognition_record 或 query_recognition_history 时，"
            f"当前 session_id 固定使用 {self.session_id}。"
            "不要编造工具结果。"
            "如果用户说“刚才”“之前”“最近识别的”，应优先结合当前会话上下文，必要时调用 query_recognition_history。"
            "只有不再需要工具时，才给最终自然语言回答。"
        )

    def call_llm(self):
        try:
            response = self.client.chat.completions.create(model=MODEL_NAME, messages=self.messages, tools=TOOLS, stream=False, extra_body={"thinking": {"type": "disabled"}})
            return response.choices[0].message
        except Exception as exc:
            logger.error("deepseek_request_failed model=%s session_id=%s error_type=%s", MODEL_NAME, self.session_id, type(exc).__name__)
            raise

    def _prepare_arguments(self, tool_name: str, arguments: dict) -> dict:
        if tool_name in {"save_recognition_record", "query_recognition_history"}:
            arguments["session_id"] = self.session_id
        return arguments

    def chat(self, user_message: str, max_steps: int = 10) -> str:
        logger.info("agent_started mode=session session_id=%s", self.session_id)
        self.messages.append({"role": "user", "content": user_message})
        answer = run_tool_call_loop(
            messages=self.messages,
            call_llm=lambda _messages: self.call_llm(),
            execute_tool=execute_tool,
            max_steps=max_steps,
            prepare_arguments=self._prepare_arguments,
            persist_final_answer=True,
            completion_message="本轮结束。",
            max_steps_message="Agent 达到最大执行步数，本轮已停止。",
        )
        logger.info("agent_finished mode=session session_id=%s", self.session_id)
        return answer

def main():
    session = AgentSession("session_001")
    print(f"\n当前 Session：{session.session_id}\n输入 exit / quit 退出程序。")
    while True:
        user_message = input("\nUser: ").strip()
        if not user_message:
            continue
        if user_message.lower() in {"exit", "quit"}:
            print("\nSession 已结束。")
            break
        try:
            print("\n[Final Answer]\n" + session.chat(user_message))
        except Exception as exc:
            print(f"\nAgent运行失败：\n{type(exc).__name__}: {exc}")

if __name__ == "__main__":
    main()
