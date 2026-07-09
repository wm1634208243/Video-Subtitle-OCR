from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from app.config import DATA_DIR


CONFIG_PATH = DATA_DIR / "config.json"
_LOCK = threading.Lock()
_DEFAULT_LLM = {
    "enabled": False,
    "provider": "openai",
    "model": "",
    "base_url": "",
    "api_key": "",
}


def get_llm_config(include_secret: bool = False) -> dict[str, Any]:
    config = _read_config().get("llm", {})
    merged = {**_DEFAULT_LLM, **config}
    result = {
        "enabled": bool(merged.get("enabled")),
        "provider": _normalize_provider(str(merged.get("provider") or "openai")),
        "model": str(merged.get("model") or ""),
        "base_url": str(merged.get("base_url") or ""),
        "has_api_key": bool(str(merged.get("api_key") or "").strip()),
    }
    if include_secret:
        result["api_key"] = str(merged.get("api_key") or "")
    return result


def save_llm_config(payload: dict[str, Any]) -> dict[str, Any]:
    with _LOCK:
        config = _read_config_unlocked()
        current = {**_DEFAULT_LLM, **config.get("llm", {})}
        api_key = str(payload.get("api_key") or "").strip()
        if not api_key:
            api_key = str(current.get("api_key") or "")
        config["llm"] = {
            "enabled": bool(payload.get("enabled")),
            "provider": _normalize_provider(str(payload.get("provider") or current.get("provider") or "openai")),
            "model": str(payload.get("model") or "").strip(),
            "base_url": str(payload.get("base_url") or "").strip(),
            "api_key": api_key,
        }
        _write_config_unlocked(config)
    return get_llm_config(include_secret=False)


def reset_llm_config() -> dict[str, Any]:
    with _LOCK:
        config = _read_config_unlocked()
        config.pop("llm", None)
        if config:
            _write_config_unlocked(config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()
    return get_llm_config(include_secret=False)


def _read_config() -> dict[str, Any]:
    with _LOCK:
        return _read_config_unlocked()


def _read_config_unlocked() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_config_unlocked(config: dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_provider(provider: str) -> str:
    value = provider.strip().lower()
    if value in {"openai", "anthropic", "ollama", "openai-compatible"}:
        return value
    return "openai"
