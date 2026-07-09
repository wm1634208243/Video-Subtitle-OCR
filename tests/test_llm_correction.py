import json

from app.llm_correction import LlmCorrectionError, _apply_corrections, _openai_request, _parse_correction_mapping
from app.subtitles import SubtitleEntry


def test_parse_llm_correction_mapping_from_json_block() -> None:
    mapping = _parse_correction_mapping(
        '```json\n{"items":[{"i":1,"text":"For MG, it is where our journey begins."}]}\n```'
    )

    assert mapping == {1: "For MG, it is where our journey begins."}


def test_apply_llm_corrections_preserves_timing() -> None:
    entries = [SubtitleEntry(text="ForMG, itswhereourjourneybegins.", start=1.0, end=3.0)]

    corrected = _apply_corrections(entries, {1: "For MG, it's where our journey begins."})

    assert corrected[0].text == "For MG, it's where our journey begins."
    assert corrected[0].start == 1.0
    assert corrected[0].end == 3.0


def test_rejects_multiline_llm_corrections() -> None:
    mapping = _parse_correction_mapping(json.dumps({"items": [{"i": 1, "text": "bad\nline"}]}))

    assert mapping == {}


def test_parse_llm_correction_mapping_from_compatible_shapes() -> None:
    assert _parse_correction_mapping('{"corrections":[{"index":1,"corrected_text":"Fixed text"}]}') == {
        1: "Fixed text"
    }
    assert _parse_correction_mapping('{"corrected_text":"Fixed text"}') == {1: "Fixed text"}
    assert _parse_correction_mapping("Fixed text") == {1: "Fixed text"}


def test_openai_request_surfaces_api_error(monkeypatch) -> None:
    def fake_post_json(url, payload, headers):
        return {"error": {"message": "model not found"}}

    monkeypatch.setattr("app.llm_correction._post_json", fake_post_json)

    try:
        _openai_request("http://example.test/v1", "bad-model", "key", [])
    except LlmCorrectionError as exc:
        assert "model not found" in str(exc)
    else:
        raise AssertionError("expected LlmCorrectionError")


def test_openai_request_retries_without_response_format(monkeypatch) -> None:
    calls = []

    def fake_post_json(url, payload, headers):
        calls.append(payload)
        if "response_format" in payload:
            return {"error": {"message": "response_format is not supported"}}
        return {"choices": [{"message": {"content": '{"items":[{"i":1,"text":"Fixed text"}]}'}}]}

    monkeypatch.setattr("app.llm_correction._post_json", fake_post_json)

    text = _openai_request("http://example.test/v1", "custom-model", "key", [])

    assert text == '{"items":[{"i":1,"text":"Fixed text"}]}'
    assert len(calls) == 2
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]


def test_openai_request_retries_after_empty_strict_response(monkeypatch) -> None:
    calls = []

    def fake_post_json(url, payload, headers):
        calls.append(payload)
        if "response_format" in payload:
            return {"choices": [{"message": {"content": ""}}]}
        return {"choices": [{"message": {"content": "Fixed text"}}]}

    monkeypatch.setattr("app.llm_correction._post_json", fake_post_json)

    assert _openai_request("http://example.test/v1", "custom-model", "key", []) == "Fixed text"
    assert len(calls) == 2
