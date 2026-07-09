from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.ffmpeg_tools import ffmpeg_exe


TEXT_SUBTITLE_CODECS = {
    "ass",
    "ssa",
    "subrip",
    "srt",
    "text",
    "webvtt",
    "mov_text",
}


@dataclass(frozen=True)
class SubtitleStream:
    subtitle_index: int
    stream_id: str
    codec: str
    language: str | None = None
    title: str | None = None


def extract_first_text_subtitle(input_path: Path, srt_path: Path, txt_path: Path) -> SubtitleStream | None:
    stream = select_text_subtitle_stream(probe_subtitle_streams(input_path))
    if stream is None:
        return None

    srt_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-map",
        f"0:s:{stream.subtitle_index}",
        str(srt_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "soft subtitle extraction failed"
        raise RuntimeError(message)

    txt_path.write_text(srt_to_plain_text(srt_path.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
    return stream


def probe_subtitle_streams(input_path: Path) -> list[SubtitleStream]:
    command = [
        ffmpeg_exe(),
        "-hide_banner",
        "-i",
        str(input_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    output = "\n".join(part for part in [result.stderr, result.stdout] if part)
    streams: list[SubtitleStream] = []
    subtitle_index = 0

    for line in output.splitlines():
        parsed = _parse_subtitle_stream_line(line, subtitle_index)
        if parsed is None:
            continue
        streams.append(parsed)
        subtitle_index += 1

    return streams


def select_text_subtitle_stream(streams: list[SubtitleStream]) -> SubtitleStream | None:
    text_streams = [stream for stream in streams if stream.codec in TEXT_SUBTITLE_CODECS]
    if not text_streams:
        return None

    for language in ("chi", "zho", "chs", "cht", "zh", "cn"):
        for stream in text_streams:
            if (stream.language or "").lower() == language:
                return stream

    return text_streams[0]


def srt_to_plain_text(content: str) -> str:
    lines: list[str] = []
    seen_blank = False
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            seen_blank = True
            continue
        if line.isdigit():
            continue
        if "-->" in line:
            continue
        line = re.sub(r"<[^>]+>", "", line)
        if not line:
            continue
        if seen_blank and lines and lines[-1] != "":
            lines.append("")
        seen_blank = False
        lines.append(line)

    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + ("\n" if lines else "")


def _parse_subtitle_stream_line(line: str, subtitle_index: int) -> SubtitleStream | None:
    if "Subtitle:" not in line or "Stream #" not in line:
        return None

    stream_match = re.search(r"Stream #(?P<id>\d+:\d+)(?:\((?P<lang>[^)]+)\))?", line)
    codec_match = re.search(r"Subtitle:\s*(?P<codec>[^,\s]+)", line)
    if not stream_match or not codec_match:
        return None

    title_match = re.search(r"title\s*:\s*(?P<title>.+)$", line, flags=re.IGNORECASE)
    return SubtitleStream(
        subtitle_index=subtitle_index,
        stream_id=stream_match.group("id"),
        codec=codec_match.group("codec").lower(),
        language=stream_match.group("lang"),
        title=title_match.group("title").strip() if title_match else None,
    )
