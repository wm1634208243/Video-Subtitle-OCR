from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import replace
from typing import Any

from app.models import JobOptions
from app.subtitles import SubtitleEntry


class LlmCorrectionError(RuntimeError):
    pass


def test_llm_connection(
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
) -> dict[str, object]:
    started = time.perf_counter()
    normalized_provider = (provider or "openai").strip().lower()
    resolved_model = _default_model(normalized_provider, model or "")
    resolved_base_url = _default_base_url(normalized_provider, base_url or "")
    if normalized_provider != "ollama" and not (api_key or "").strip():
        raise LlmCorrectionError("API Key is required for this provider.")

    sample = [SubtitleEntry(text="ForMG, itswhereourjourneybegins.", start=0.0, end=1.0)]
    text = _request_correction_text(
        normalized_provider,
        resolved_base_url,
        resolved_model,
        (api_key or "").strip(),
        sample,
    )
    mapping = _parse_correction_mapping(text)
    corrected = mapping.get(1, "")
    if not corrected:
        raise LlmCorrectionError(
            "The model responded, but no correction item was returned. "
            f"Response preview: {_preview_text(text)}"
        )
    return {
        "provider": normalized_provider,
        "model": resolved_model,
        "corrected_text": corrected,
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
    }


def correct_subtitles_with_llm(
    entries: list[SubtitleEntry],
    options: JobOptions,
    secrets: dict[str, str] | None = None,
) -> list[SubtitleEntry]:
    if not options.llm_correction or not entries:
        return entries

    provider = (options.llm_provider or "openai").strip().lower()
    model = _default_model(provider, options.llm_model)
    base_url = _default_base_url(provider, options.llm_base_url)
    api_key = (secrets or {}).get("llm_api_key", "").strip()
    if provider != "ollama" and not api_key:
        return entries

    corrected: list[SubtitleEntry] = []
    for chunk_start in range(0, len(entries), 40):
        chunk = entries[chunk_start : chunk_start + 40]
        try:
            mapping = _request_corrections(provider, base_url, model, api_key, chunk)
        except Exception:
            corrected.extend(chunk)
            continue
        corrected.extend(_apply_corrections(chunk, mapping))
    return corrected


def _request_corrections(
    provider: str,
    base_url: str,
    model: str,
    api_key: str,
    entries: list[SubtitleEntry],
) -> dict[int, str]:
    text = _request_correction_text(provider, base_url, model, api_key, entries)
    return _parse_correction_mapping(text)


def _request_correction_text(
    provider: str,
    base_url: str,
    model: str,
    api_key: str,
    entries: list[SubtitleEntry],
) -> str:
    user_payload = {
        "items": [
            {"i": index + 1, "text": entry.text}
            for index, entry in enumerate(entries)
        ]
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You correct OCR subtitles. Fix spelling, missing spaces, obvious punctuation, "
                "and OCR artifacts. Preserve language, meaning, line order, names, and casing when reasonable. "
                "Do not translate. Do not add explanations. Return only JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "Correct only the text values. Keep every i value. "
                "Return JSON shaped as {\"items\":[{\"i\":1,\"text\":\"...\"}]}.\n"
                + json.dumps(user_payload, ensure_ascii=False)
            ),
        },
    ]
    if provider == "anthropic":
        text = _anthropic_request(base_url, model, api_key, messages)
    elif provider == "ollama":
        text = _ollama_request(base_url, model, messages)
    elif provider in {"openai", "openai-compatible"}:
        text = _openai_request(base_url, model, api_key, messages)
    else:
        raise LlmCorrectionError(f"Unsupported LLM provider: {provider}")
    return text


def _openai_request(base_url: str, model: str, api_key: str, messages: list[dict[str, str]]) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = _join_url(base_url, "/chat/completions")
    strict_payload = {**payload, "response_format": {"type": "json_object"}}
    try:
        data = _post_json(url, strict_payload, headers)
        _raise_api_error_if_present(data)
        text = _extract_openai_text(data)
        if text:
            return text
    except LlmCorrectionError as exc:
        if not _should_retry_openai_without_response_format(str(exc)):
            raise

    data = _post_json(url, payload, headers)
    _raise_api_error_if_present(data)
    return _extract_openai_text(data)


def _anthropic_request(base_url: str, model: str, api_key: str, messages: list[dict[str, str]]) -> str:
    system = messages[0]["content"]
    user = messages[1]["content"]
    payload = {
        "model": model,
        "max_tokens": 4096,
        "temperature": 0.0,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    data = _post_json(_join_url(base_url, "/v1/messages"), payload, headers)
    _raise_api_error_if_present(data)
    blocks = data.get("content", [])
    if not isinstance(blocks, list):
        return ""
    return "\n".join(str(block.get("text", "")) for block in blocks if isinstance(block, dict))


def _ollama_request(base_url: str, model: str, messages: list[dict[str, str]]) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    data = _post_json(_join_url(base_url, "/api/chat"), payload, {"Content-Type": "application/json"})
    _raise_api_error_if_present(data)
    return str(data.get("message", {}).get("content", ""))


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        message = _api_error_message(_load_json_object(body)) or body.strip() or str(exc)
        raise LlmCorrectionError(f"HTTP {exc.code}: {_preview_text(message)}") from exc
    except urllib.error.URLError as exc:
        raise LlmCorrectionError(str(exc)) from exc
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise LlmCorrectionError("LLM response was not valid JSON") from exc
    if not isinstance(data, dict):
        raise LlmCorrectionError("LLM response JSON was not an object")
    return data


def _parse_correction_mapping(text: str) -> dict[int, str]:
    payload = _load_json_object(text)
    if not payload and _is_safe_corrected_text(text):
        return {1: " ".join(text.split())}

    direct_text = payload.get("text") or payload.get("corrected_text") or payload.get("correction")
    if isinstance(direct_text, str) and _is_safe_corrected_text(direct_text):
        return {1: " ".join(direct_text.split())}

    items = payload.get("items") or payload.get("corrections") or payload.get("results") or []
    if isinstance(items, dict):
        items = [
            {"i": key, "text": value}
            for key, value in items.items()
        ]
    if not isinstance(items, list):
        return {}
    mapping: dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("i") or item.get("index") or item.get("id"))
        except (TypeError, ValueError):
            continue
        value = item.get("text") or item.get("corrected_text") or item.get("correction")
        if isinstance(value, str) and _is_safe_corrected_text(value):
            mapping[index] = " ".join(value.split())
    return mapping


def _load_json_object(text: str) -> dict[str, Any]:
    value = (text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, flags=re.S)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return data if isinstance(data, dict) else {}


def _apply_corrections(entries: list[SubtitleEntry], mapping: dict[int, str]) -> list[SubtitleEntry]:
    corrected: list[SubtitleEntry] = []
    for index, entry in enumerate(entries, start=1):
        text = mapping.get(index, entry.text)
        corrected.append(replace(entry, text=text))
    return corrected


def _is_safe_corrected_text(text: str) -> bool:
    value = text.strip()
    if not value:
        return False
    if len(value) > 500:
        return False
    if "\n" in value or "\r" in value:
        return False
    return True


def _extract_openai_text(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return "".join(
                        str(item.get("text", ""))
                        for item in content
                        if isinstance(item, dict)
                    )
            text = choice.get("text")
            if isinstance(text, str):
                return text
    output_text = data.get("output_text")
    if isinstance(output_text, str):
        return output_text
    return ""


def _raise_api_error_if_present(data: dict[str, Any]) -> None:
    message = _api_error_message(data)
    if message:
        raise LlmCorrectionError(message)


def _api_error_message(data: dict[str, Any]) -> str:
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, str):
        return error
    if isinstance(error, dict):
        message = error.get("message") or error.get("error") or error.get("detail")
        if message:
            return str(message)
    message = data.get("message") if isinstance(data, dict) else None
    if isinstance(message, str) and data.get("ok") is False:
        return message
    return ""


def _should_retry_openai_without_response_format(message: str) -> bool:
    value = message.lower()
    retry_markers = (
        "response_format",
        "json_object",
        "json mode",
        "schema",
        "unsupported",
        "not supported",
        "invalid parameter",
        "invalid_request",
    )
    return any(marker in value for marker in retry_markers)


def _preview_text(text: str, limit: int = 220) -> str:
    value = " ".join((text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "..."


def _default_model(provider: str, model: str) -> str:
    value = model.strip()
    if value:
        return value
    return {
        "openai": "gpt-4.1-mini",
        "openai-compatible": "gpt-4.1-mini",
        "anthropic": "claude-3-5-haiku-latest",
        "ollama": "qwen2.5:7b",
    }.get(provider, "gpt-4.1-mini")


def _default_base_url(provider: str, base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    if value:
        return value
    return {
        "openai": "https://api.openai.com/v1",
        "openai-compatible": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com",
        "ollama": "http://127.0.0.1:11434",
    }.get(provider, "")


def _join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")
