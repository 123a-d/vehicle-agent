import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import api.main as api_main

class FakeSession:
    def __init__(self, session_id): self.session_id, self.messages = session_id, []
    def chat(self, user_message): self.messages.append(user_message); return "mock: " + user_message

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "AgentSession", FakeSession)
    monkeypatch.setattr(api_main, "UPLOAD_ROOT", tmp_path / "uploads")
    api_main.SESSIONS.clear()
    with TestClient(api_main.app) as value: yield value
    api_main.SESSIONS.clear()

def create_session(client):
    response = client.post("/sessions"); assert response.status_code == 200
    return response.json()["session_id"]

def png_bytes():
    value = io.BytesIO(); Image.new("RGB", (2, 2), "red").save(value, "PNG"); return value.getvalue()

def test_health(client):
    response = client.get("/health"); assert response.status_code == 200 and response.json()["status"] == "healthy"

def test_create_and_list_sessions(client):
    session_id = create_session(client); assert client.get("/sessions").json()["sessions"] == [session_id]

def test_delete_session(client):
    session_id = create_session(client)
    assert client.delete(f"/sessions/{session_id}").status_code == 200
    assert client.delete(f"/sessions/{session_id}").status_code == 404

def test_upload_valid_image(client):
    session_id = create_session(client)
    response = client.post(f"/sessions/{session_id}/upload-image", files={"file": ("car.png", png_bytes(), "image/png")})
    assert response.status_code == 200 and response.json()["success"] is True

def test_chat(client):
    session_id = create_session(client)
    response = client.post("/chat", json={"session_id": session_id, "message": "hello"})
    assert response.status_code == 200 and response.json()["answer"] == "mock: hello"

def test_invalid_session_upload(client):
    response = client.post("/sessions/missing/upload-image", files={"file": ("car.png", png_bytes(), "image/png")})
    assert response.status_code == 404

def test_invalid_session_chat(client):
    assert client.post("/chat", json={"session_id": "missing", "message": "hello"}).status_code == 404

def test_invalid_image(client):
    session_id = create_session(client)
    response = client.post(f"/sessions/{session_id}/upload-image", files={"file": ("bad.png", b"bad", "image/png")})
    assert response.status_code == 400

def test_invalid_extension(client):
    session_id = create_session(client)
    response = client.post(f"/sessions/{session_id}/upload-image", files={"file": ("bad.txt", b"bad", "text/plain")})
    assert response.status_code == 400

def test_oversized_image(client, monkeypatch):
    session_id = create_session(client); monkeypatch.setattr(api_main, "MAX_IMAGE_BYTES", 8)
    response = client.post(f"/sessions/{session_id}/upload-image", files={"file": ("big.png", png_bytes(), "image/png")})
    assert response.status_code == 413

def test_empty_chat(client):
    session_id = create_session(client)
    assert client.post("/chat", json={"session_id": session_id, "message": "  "}).status_code == 400
