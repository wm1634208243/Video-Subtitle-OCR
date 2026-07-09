from __future__ import annotations

from pathlib import Path

import cv2

from app.config import DEFAULT_PADDLE_BATCH_SIZE
from app.models import JobOptions


def build_auto_options(input_path: Path, mode: str, engine: str, language: str) -> JobOptions:
    profile = mode if mode in {"fast", "balanced", "accurate"} else "balanced"
    metadata = probe_video(input_path)
    duration = metadata.get("duration") or 0
    width = metadata.get("width") or 0
    height = metadata.get("height") or 0
    is_portrait = bool(width and height and height > width * 1.2)
    detected_crop = estimate_subtitle_crop(input_path)

    if profile == "fast":
        fps = 0.75 if duration >= 20 * 60 else 1.0
        crop_bottom = 0.45 if is_portrait else 0.55
        similarity = 0.76
        batch_size = min(8, max(DEFAULT_PADDLE_BATCH_SIZE, 4))
        frame_diff_threshold = 0.4
    elif profile == "accurate":
        fps = 2.0 if duration <= 20 * 60 else 1.5
        crop_bottom = 0.55 if is_portrait else 0.65 if height and height <= 720 else 0.6
        similarity = 0.68
        batch_size = min(6, max(DEFAULT_PADDLE_BATCH_SIZE, 4))
        frame_diff_threshold = 0.16
    else:
        fps = 1.0 if is_portrait or duration >= 30 * 60 else 1.5
        crop_bottom = 0.5 if is_portrait else 0.6 if height and height <= 720 else 0.55
        similarity = 0.72
        batch_size = min(6, max(DEFAULT_PADDLE_BATCH_SIZE, 4))
        frame_diff_threshold = 0.3

    if width and width <= 960 and not is_portrait:
        crop_bottom = min(crop_bottom + 0.05, 0.75)
    if detected_crop:
        crop_bottom = max(crop_bottom, min(detected_crop, 0.58 if is_portrait else 0.65))

    return JobOptions(
        engine=engine,
        fps=fps,
        crop_bottom=crop_bottom,
        similarity=similarity,
        min_length=2,
        ocr_batch_size=batch_size,
        language=language,
        skip_unchanged_frames=True,
        frame_diff_threshold=frame_diff_threshold,
        auto_subtitle_strip=True,
        mode=profile,
        auto_config=True,
    )


def probe_video(input_path: Path) -> dict[str, float]:
    capture = cv2.VideoCapture(str(input_path))
    try:
        if not capture.isOpened():
            return {}

        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        native_fps = capture.get(cv2.CAP_PROP_FPS) or 0
        duration = frame_count / native_fps if frame_count > 0 and native_fps > 0 else 0
        return {
            "width": float(width),
            "height": float(height),
            "frame_count": float(frame_count),
            "native_fps": float(native_fps),
            "duration": float(duration),
        }
    finally:
        capture.release()


def estimate_subtitle_crop(input_path: Path) -> float | None:
    capture = cv2.VideoCapture(str(input_path))
    try:
        if not capture.isOpened():
            return None

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            return None

        positions = [0.25, 0.5, 0.75]
        crop_candidates: list[float] = []
        for position in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_count * position)))
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            candidate = _estimate_crop_from_frame(frame)
            if candidate:
                crop_candidates.append(candidate)

        if not crop_candidates:
            return None
        crop_candidates.sort()
        return crop_candidates[len(crop_candidates) // 2]
    finally:
        capture.release()


def _estimate_crop_from_frame(frame) -> float | None:
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        return None

    lower_start = int(height * 0.45)
    lower = frame[lower_start:, :]
    gray = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 80, 180)

    band_height = max(6, height // 90)
    scores = []
    for y in range(0, edges.shape[0], band_height):
        band = edges[y : y + band_height, :]
        scores.append(float(band.mean()))

    if not scores:
        return None

    max_score = max(scores)
    if max_score < 2.0:
        return None

    threshold = max(2.0, max_score * 0.42)
    active_rows = [index for index, score in enumerate(scores) if score >= threshold]
    if not active_rows:
        return None

    top_band = max(0, min(active_rows) - 2)
    top_y = lower_start + top_band * band_height
    crop_ratio = (height - top_y) / height
    return round(min(max(crop_ratio, 0.35), 0.65), 2)
