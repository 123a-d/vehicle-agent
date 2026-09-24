"""项目统一日志入口；日志不包含密钥、Authorization 或完整 Prompt。"""
from __future__ import annotations
import logging
from config import LOG_LEVEL

def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return logging.getLogger(name)
