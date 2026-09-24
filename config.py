"""Vehicle Agent 可变运行配置的单一来源。"""
from __future__ import annotations
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
DATABASE_PATH = Path(os.environ.get("VEHICLE_DATABASE_PATH", PROJECT_ROOT / "database" / "vehicle_agent.db"))
CHECKPOINT_PATH = Path(os.environ.get("VEHICLE_CHECKPOINT_PATH", PROJECT_ROOT / "outputs" / "unknown_weighted" / "best_macro_f1.pth"))
UPLOAD_ROOT = Path(os.environ.get("VEHICLE_UPLOAD_ROOT", PROJECT_ROOT / "uploads"))
WEB_INDEX_PATH = Path(os.environ.get("VEHICLE_WEB_INDEX_PATH", PROJECT_ROOT / "web" / "index.html"))
MAX_IMAGE_BYTES = int(os.environ.get("VEHICLE_MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
LOG_LEVEL = os.environ.get("VEHICLE_LOG_LEVEL", "INFO").upper()

def get_deepseek_api_key() -> str:
    """按需读取密钥，避免导入时缓存或输出敏感值。"""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("没有检测到环境变量 DEEPSEEK_API_KEY。")
    return api_key
