from fastapi.testclient import TestClient

from app.main import _resolve_engine, _resolve_engine_plan, app
from app.ocr_engines import EngineStatus


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ok", "degraded"}
    assert payload["version"]
    assert payload["worker_count"] >= 1


def test_engines_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/api/engines")

    assert response.status_code == 200
    payload = response.json()
    assert {engine["id"] for engine in payload} == {
        "paddle",
        "openvino",
        "onnxruntime",
        "easyocr",
        "tesseract",
    }
    assert any(engine["default"] for engine in payload)


def test_auto_engine_fast_prefers_openvino(monkeypatch) -> None:
    def fake_status(engine_id: str) -> EngineStatus:
        return EngineStatus(
            id=engine_id,
            name=engine_id,
            available=engine_id in {"openvino", "paddle"},
        )

    monkeypatch.setattr("app.main.get_engine_status", fake_status)

    assert _resolve_engine("auto", "fast", "ch") == "openvino"


def test_auto_engine_accurate_uses_accelerated_primary_with_paddle_review(monkeypatch) -> None:
    def fake_status(engine_id: str) -> EngineStatus:
        return EngineStatus(id=engine_id, name=engine_id, available=True)

    monkeypatch.setattr("app.main.get_engine_status", fake_status)

    assert _resolve_engine_plan("auto", "accurate", "ch") == ("openvino", "paddle")


def test_auto_engine_accurate_falls_back_to_paddle_without_accelerator(monkeypatch) -> None:
    def fake_status(engine_id: str) -> EngineStatus:
        return EngineStatus(id=engine_id, name=engine_id, available=engine_id == "paddle")

    monkeypatch.setattr("app.main.get_engine_status", fake_status)

    assert _resolve_engine_plan("auto", "accurate", "ch") == ("paddle", "")


def test_auto_engine_falls_back_to_onnxruntime(monkeypatch) -> None:
    def fake_status(engine_id: str) -> EngineStatus:
        return EngineStatus(id=engine_id, name=engine_id, available=engine_id == "onnxruntime")

    monkeypatch.setattr("app.main.get_engine_status", fake_status)

    assert _resolve_engine("auto", "balanced", "en") == "onnxruntime"


def test_install_profiles_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/api/install/profiles")

    assert response.status_code == 200
    payload = response.json()
    assert {
        "recommended",
        "openvino",
        "onnxruntime",
        "easyocr",
        "tesseract",
        "full",
        "dev",
        "core",
    } <= {profile["id"] for profile in payload}


def test_install_current_endpoint_is_empty_before_install() -> None:
    with TestClient(app) as client:
        response = client.get("/api/install/current")

    assert response.status_code == 200


def test_install_endpoint_requires_action_header() -> None:
    with TestClient(app) as client:
        response = client.post("/api/install/easyocr")

    assert response.status_code == 403


def test_missing_job_cancel_returns_404() -> None:
    with TestClient(app) as client:
        response = client.post("/api/jobs/not-found/cancel")

    assert response.status_code == 404


def test_missing_job_ocr_log_returns_404() -> None:
    with TestClient(app) as client:
        response = client.get("/api/jobs/not-found/ocr-log")

    assert response.status_code == 404


def test_llm_config_endpoints_do_not_echo_api_key(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("app.user_config.CONFIG_PATH", tmp_path / "config.json")

    with TestClient(app) as client:
        save_response = client.put(
            "/api/config/llm",
            json={
                "enabled": True,
                "provider": "openai-compatible",
                "model": "gpt-test",
                "base_url": "http://localhost/v1",
                "api_key": "secret",
            },
        )
        read_response = client.get("/api/config/llm")
        reset_response = client.post("/api/config/llm/reset")

    assert save_response.status_code == 200
    assert save_response.json()["has_api_key"] is True
    assert "secret" not in save_response.text
    assert read_response.json()["has_api_key"] is True
    assert reset_response.json()["has_api_key"] is False


def test_llm_test_endpoint_uses_submitted_config(monkeypatch) -> None:
    def fake_test(provider: str, model: str, base_url: str, api_key: str):
        assert provider == "ollama"
        assert model == "qwen-test"
        assert base_url == "http://127.0.0.1:11434"
        assert api_key == ""
        return {
            "provider": provider,
            "model": model,
            "corrected_text": "For MG, its where our journey begins.",
            "elapsed_ms": 12,
        }

    monkeypatch.setattr("app.main.test_llm_connection", fake_test)

    with TestClient(app) as client:
        response = client.post(
            "/api/config/llm/test",
            json={
                "enabled": True,
                "provider": "ollama",
                "model": "qwen-test",
                "base_url": "http://127.0.0.1:11434",
                "api_key": "",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["corrected_text"].startswith("For MG")
