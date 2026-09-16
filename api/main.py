"""
Stage 8.7
Vehicle Agent FastAPI Backend

功能：
- GET /                     基本状态
- GET /health               健康检查
- GET /app                  Web 前端
- POST /sessions            创建 Session
- GET /sessions             查看 Session
- DELETE /sessions/{id}     删除 Session
- POST /sessions/{id}/upload-image
                            上传并验证图片
- POST /chat                指定 Session 聊天
"""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

from session_agent import AgentSession


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_INDEX_PATH = PROJECT_ROOT / "web" / "index.html"
UPLOAD_ROOT = PROJECT_ROOT / "uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

MAX_IMAGE_BYTES = 10 * 1024 * 1024


app = FastAPI(
    title="Vehicle Agent API",
    description=(
        "ResNet18 Visual Tool + "
        "DeepSeek Multi-Tool Agent + "
        "SQLite"
    ),
    version="0.7.0",
)


SESSIONS: dict[str, AgentSession] = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str


def generate_session_id() -> str:
    """生成一个基本不会重复的 Session ID。"""
    return f"session_{uuid4().hex[:12]}"


def create_session() -> AgentSession:
    """创建 AgentSession 并注册到内存字典。"""
    session_id = generate_session_id()

    session = AgentSession(
        session_id=session_id
    )

    SESSIONS[session_id] = session

    return session


def get_session(
    session_id: str,
) -> AgentSession:
    """根据 session_id 获取 AgentSession。"""

    session_id = str(
        session_id
    ).strip()

    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="session_id 不能为空",
        )

    session = SESSIONS.get(
        session_id
    )

    if session is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Session 不存在或已经失效："
                f"{session_id}"
            ),
        )

    return session


@app.get("/")
def root() -> dict:
    return {
        "success": True,
        "message":
            "Vehicle Agent API is running.",
        "stage": "8.7",
        "web": "/app",
        "docs": "/docs",
    }


@app.get("/health")
def health_check() -> dict:
    return {
        "success": True,
        "status": "healthy",
        "service":
            "vehicle-agent-api",
        "stage": "8.7",
        "active_sessions":
            len(SESSIONS),
        "upload_directory":
            str(UPLOAD_ROOT),
        "web_index_exists":
            WEB_INDEX_PATH.exists(),
    }


@app.get("/app")
def web_app():
    if not WEB_INDEX_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Web前端文件不存在："
                f"{WEB_INDEX_PATH}"
            ),
        )

    return FileResponse(
        WEB_INDEX_PATH
    )


@app.post("/sessions")
def create_session_api() -> dict:
    try:
        session = create_session()

        return {
            "success": True,
            "session_id":
                session.session_id,
            "message":
                "Session 创建成功",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Session 创建失败："
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )


@app.get("/sessions")
def list_sessions() -> dict:
    return {
        "success": True,
        "count": len(SESSIONS),
        "sessions":
            list(SESSIONS.keys()),
    }


@app.delete(
    "/sessions/{session_id}"
)
def delete_session(
    session_id: str,
) -> dict:
    session_id = str(
        session_id
    ).strip()

    if session_id not in SESSIONS:
        raise HTTPException(
            status_code=404,
            detail=(
                "Session 不存在："
                f"{session_id}"
            ),
        )

    del SESSIONS[
        session_id
    ]

    return {
        "success": True,
        "session_id":
            session_id,
        "message":
            "Session 已删除",
    }


@app.post(
    "/sessions/{session_id}/upload-image"
)
def upload_image(
    session_id: str,
    file: UploadFile = File(...),
) -> dict:
    """
    上传并验证车辆图片。

    流程：
    Session校验
        ↓
    文件名 / 扩展名 / MIME / 大小检查
        ↓
    保存
        ↓
    PIL.Image.verify()
        ↓
    返回服务器本地 image_path
    """

    get_session(
        session_id
    )

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="上传文件没有文件名",
        )

    suffix = (
        Path(file.filename)
        .suffix
        .lower()
    )

    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=(
                "不支持的图片格式："
                f"{suffix}。"
                "目前支持 jpg、jpeg、png、webp。"
            ),
        )

    if (
        file.content_type
        and
        not file.content_type.startswith(
            "image/"
        )
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "上传文件不是图片类型："
                f"{file.content_type}"
            ),
        )

    file.file.seek(
        0,
        2,
    )

    file_size = (
        file.file.tell()
    )

    file.file.seek(0)

    if file_size <= 0:
        raise HTTPException(
            status_code=400,
            detail="上传文件为空",
        )

    if file_size > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                "图片过大，"
                "当前最大允许 10MB。"
            ),
        )

    session_upload_dir = (
        UPLOAD_ROOT
        / session_id
    )

    session_upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_filename = (
        f"{uuid4().hex}"
        f"{suffix}"
    )

    saved_path = (
        session_upload_dir
        / saved_filename
    )

    try:
        with saved_path.open(
            "wb"
        ) as output_file:
            shutil.copyfileobj(
                file.file,
                output_file,
            )

    except Exception as exc:
        if saved_path.exists():
            saved_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=(
                "图片保存失败："
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )

    try:
        with Image.open(
            saved_path
        ) as image:
            image.verify()

    except (
        UnidentifiedImageError,
        OSError,
    ):
        if saved_path.exists():
            saved_path.unlink()

        raise HTTPException(
            status_code=400,
            detail=(
                "上传文件无法解析为有效图片。"
            ),
        )

    return {
        "success": True,
        "session_id":
            session_id,
        "original_filename":
            file.filename,
        "saved_filename":
            saved_filename,
        "image_path":
            str(saved_path),
        "content_type":
            file.content_type,
        "size_bytes":
            file_size,
    }


@app.post("/chat")
def chat(
    request: ChatRequest,
) -> dict:
    """把用户消息发送给指定 Agent Session。"""

    session_id = (
        request
        .session_id
        .strip()
    )

    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="session_id 不能为空",
        )

    user_message = (
        request
        .message
        .strip()
    )

    if not user_message:
        raise HTTPException(
            status_code=400,
            detail="message 不能为空",
        )

    session = get_session(
        session_id
    )

    try:
        answer = session.chat(
            user_message=user_message
        )

        return {
            "success": True,
            "session_id":
                session.session_id,
            "message":
                user_message,
            "answer":
                answer,
        }

    except HTTPException:
        raise

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=(
                "Agent 请求超时，"
                "请稍后重新尝试。"
            ),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Agent 运行失败："
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )
