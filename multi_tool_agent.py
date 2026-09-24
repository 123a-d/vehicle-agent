"""无持久 Session 的多工具 Vehicle Agent。"""
from __future__ import annotations
from openai import OpenAI
from agent_core import run_tool_call_loop
from agent_tools import TOOLS, execute_tool
from config import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, get_deepseek_api_key
from logging_config import get_logger

BASE_URL = DEEPSEEK_BASE_URL
MODEL_NAME = DEEPSEEK_MODEL
logger = get_logger(__name__)
SYSTEM_PROMPT = (
    "你是一个车辆智能助手。"
    "你拥有以下工具："
    "1. recognize_vehicle：识别车辆图片；"
    "2. query_vehicle_info：根据 vehicle_id 查询车型详细信息；"
    "3. save_recognition_record：把识别结果保存到数据库；"
    "4. query_recognition_history：查询某个会话的识别历史。"
    "当用户要求识别车辆图片时，必须调用 recognize_vehicle，不能根据图片文件名猜测车型。"
    "如果用户同时要求：识别车辆、查询详细信息、保存识别结果，必须按照工具之间的数据依赖顺序执行。"
    "第一步必须先调用 recognize_vehicle。"
    "只有在 recognize_vehicle 返回真实 vehicle_id 后，才能调用 query_vehicle_info。"
    "只有在获得真实识别结果和车型信息后，才能调用 save_recognition_record。"
    "如果后一个工具需要前一个工具的输出，不要提前猜测参数，必须先等待前一个工具返回结果。"
    "调用 save_recognition_record 时，vehicle_id、vehicle_name、confidence、status 必须来自前面工具的真实返回结果，不能自己编造。"
    "session_id 使用用户提供的当前会话ID。"
    "如果某个工具返回 success=false，不要编造后续结果。"
    "如果错误导致后续任务无法继续，应该停止后续依赖操作，并在最终回答中说明失败原因。"
    "只有当用户要求的任务全部完成，并且不再需要调用任何工具时，才生成最终自然语言回答。"
    "对于视觉识别结果，使用“模型识别为”这样的表达，不要把模型预测描述成绝对事实。"
)

def create_client() -> OpenAI:
    return OpenAI(api_key=get_deepseek_api_key(), base_url=BASE_URL)

def call_llm(client: OpenAI, messages: list):
    try:
        response = client.chat.completions.create(model=MODEL_NAME, messages=messages, tools=TOOLS, stream=False, extra_body={"thinking": {"type": "disabled"}})
        return response.choices[0].message
    except Exception as exc:
        logger.error("deepseek_request_failed model=%s error_type=%s", MODEL_NAME, type(exc).__name__)
        raise

def run_agent_loop(user_message: str, max_steps: int = 10) -> str:
    logger.info("agent_started mode=single_turn")
    client = create_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}]
    answer = run_tool_call_loop(
        messages=messages,
        call_llm=lambda current: call_llm(client=client, messages=current),
        execute_tool=execute_tool,
        max_steps=max_steps,
    )
    logger.info("agent_finished mode=single_turn")
    return answer

def main():
    message = "当前会话ID是 session_001。请识别 D:\\car_project\\0009_0030.jpg，查询详情并保存结果。"
    try:
        print("\n[Final Answer]\n" + run_agent_loop(message))
    except Exception as exc:
        print(f"\nAgent运行失败：\n{type(exc).__name__}: {exc}")

if __name__ == "__main__":
    main()
