from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher


TRIM_CHARS = "•·_|~`^ "
VALID_PUNCTUATION = "，。！？；：、,.!?;:'\"-()[]（）【】《》“”‘’…"
NORMALIZE_DELETE = str.maketrans("", "", " \t\r\n" + VALID_PUNCTUATION)


@dataclass
class SubtitleEntry:
    text: str
    start: float
    end: float


def clean_text(raw: str | None, min_length: int) -> str | None:
    if not raw:
        return None
    text = " ".join(raw.split()).strip(TRIM_CHARS)
    if len(text) < min_length:
        return None

    real_chars = 0
    for char in text:
        if char.isalnum() or "\u4e00" <= char <= "\u9fff" or char in VALID_PUNCTUATION:
            real_chars += 1
    if real_chars / max(len(text), 1) < 0.55:
        return None
    return text


def normalize(text: str) -> str:
    return text.lower().translate(NORMALIZE_DELETE)


def is_similar(a: str, b: str, threshold: float) -> bool:
    left = normalize(a)
    right = normalize(b)
    if not left or not right:
        return False
    return SequenceMatcher(None, left, right).ratio() >= threshold


def merge_ocr_results(
    samples: list[tuple[float, str | None]],
    fps: float,
    min_length: int,
    similarity: float,
) -> list[SubtitleEntry]:
    entries: list[SubtitleEntry] = []
    current: SubtitleEntry | None = None
    current_best = ""
    frame_span = 1.0 / fps

    for timestamp, raw_text in samples:
        text = clean_text(raw_text, min_length)
        if text is None:
            if current:
                current.end = max(current.end, timestamp)
                entries.append(current)
                current = None
                current_best = ""
            continue

        end_time = timestamp + frame_span
        if current is None:
            current = SubtitleEntry(text=text, start=timestamp, end=end_time)
            current_best = text
            continue

        if is_similar(current_best, text, similarity):
            if len(normalize(text)) > len(normalize(current_best)):
                current.text = text
                current_best = text
            current.end = end_time
        else:
            entries.append(current)
            current = SubtitleEntry(text=text, start=timestamp, end=end_time)
            current_best = text

    if current:
        entries.append(current)

    return _drop_contained_fragments(entries)


def _drop_contained_fragments(entries: list[SubtitleEntry]) -> list[SubtitleEntry]:
    filtered: list[SubtitleEntry] = []
    for entry in entries:
        entry_norm = normalize(entry.text)
        previous_norm = normalize(filtered[-1].text) if filtered else ""
        if (
            filtered
            and entry_norm
            and entry_norm in previous_norm
            and len(entry_norm) <= len(previous_norm) * 0.8
        ):
            filtered[-1].end = max(filtered[-1].end, entry.end)
            continue
        filtered.append(entry)
    return filtered


def srt_timestamp(seconds: float) -> str:
    millis = round(max(seconds, 0) * 1000)
    hours = millis // 3_600_000
    minutes = (millis % 3_600_000) // 60_000
    secs = (millis % 60_000) // 1000
    ms = millis % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_outputs(entries: list[SubtitleEntry], srt_path, txt_path) -> None:
    srt_lines: list[str] = []
    txt_lines: list[str] = []
    for index, entry in enumerate(entries, start=1):
        srt_lines.append(str(index))
        srt_lines.append(f"{srt_timestamp(entry.start)} --> {srt_timestamp(entry.end)}")
        srt_lines.append(entry.text)
        srt_lines.append("")
        txt_lines.append(entry.text)

    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
    txt_path.write_text("\n".join(txt_lines) + ("\n" if txt_lines else ""), encoding="utf-8")
