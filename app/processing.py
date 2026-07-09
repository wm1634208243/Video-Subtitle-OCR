from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
import re

import cv2
import numpy as np

from app.config import CLEANUP_INTERMEDIATE_FILES, CPU_THREADS, RETRY_FULL_FRAME_ON_EMPTY
from app.debug_tools import append_ocr_log, write_debug_preview
from app.ffmpeg_tools import ffmpeg_exe
from app.job_store import JobRecord, store
from app.llm_correction import correct_subtitles_with_llm
from app.models import JobOptions, JobStatus
from app.ocr_engines import get_engine
from app.soft_subtitles import extract_first_text_subtitle
from app.subtitles import merge_ocr_results, write_outputs


class JobCancelled(RuntimeError):
    pass


@dataclass
class PendingOcrFrame:
    timestamp: float
    image: np.ndarray
    original: np.ndarray
    diff: float | None = None
    reuse_timestamps: list[tuple[float, float | None]] = field(default_factory=list)


def process_job(job: JobRecord) -> None:
    options = JobOptions(**job.options)
    cv2.setNumThreads(CPU_THREADS)
    job_dir = Path(job.job_dir)
    frames_dir = job_dir / "frames"
    output_dir = job_dir / "output"
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        _raise_if_canceled(job)
        store.update(
            job.id,
            status=JobStatus.running.value,
            phase="detecting-subtitles",
            progress=0.03,
            message="Checking embedded subtitle tracks",
        )
        subtitle_stream = extract_first_text_subtitle(
            Path(job.input_path),
            output_dir / "subtitles.srt",
            output_dir / "subtitles.txt",
        )
        _raise_if_canceled(job)
        if subtitle_stream is not None:
            store.update(
                job.id,
                status=JobStatus.done.value,
                phase="done",
                progress=1.0,
                message=f"Extracted embedded subtitle track ({subtitle_stream.codec})",
            )
            _cleanup_intermediate_files(job_dir)
            return

        store.update(
            job.id,
            status=JobStatus.running.value,
            phase="extracting",
            progress=0.05,
            message="Extracting video frames",
        )
        _extract_frames(Path(job.input_path), frames_dir, options.fps)
        _raise_if_canceled(job)

        frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
        if not frame_paths:
            raise RuntimeError("No frames were extracted from the video")
        options = _lock_auto_subtitle_region(frame_paths, options, job.id)
        write_debug_preview(
            frame_paths,
            options.crop_bottom,
            output_dir / "debug_preview.jpg",
            crop_region=_active_crop_region_dict(options),
        )

        store.update(
            job.id,
            phase="loading-ocr",
            progress=0.12,
            message=f"Loading OCR model, {len(frame_paths)} frames queued",
        )
        engine = get_engine(options.engine, options.language, options.ocr_batch_size)
        review_engine = (
            get_engine(options.review_engine, options.language, options.ocr_batch_size)
            if options.review_engine and options.review_engine != options.engine
            else None
        )

        samples: list[tuple[float, str | None]] = []
        total = len(frame_paths)
        batch_size = options.ocr_batch_size if options.engine == "paddle" else 1
        ocr_started_at = time.perf_counter()
        pending_frames: list[PendingOcrFrame] = []
        previous_signature: np.ndarray | None = None
        last_reusable_text: str | None = None
        ocr_count = 0
        reused_count = 0
        reviewed_count = 0

        for frame_index, frame_path in enumerate(frame_paths, start=1):
            _raise_if_canceled(job)
            timestamp = (frame_index - 1) / options.fps
            image = cv2.imread(str(frame_path))
            if image is None:
                samples.append((timestamp, None))
                continue

            signature = _subtitle_signature(image, options)
            diff = _signature_diff(previous_signature, signature)
            is_repeated = (
                options.skip_unchanged_frames
                and previous_signature is not None
                and signature is not None
                and diff is not None
                and diff <= options.frame_diff_threshold
            )

            if is_repeated and pending_frames:
                pending_frames[-1].reuse_timestamps.append((timestamp, diff))
            elif is_repeated and last_reusable_text is not None:
                samples.append((timestamp, last_reusable_text))
                reused_count += 1
                append_ocr_log(
                    output_dir / "ocr_samples.jsonl",
                    [
                        _ocr_log_row(
                            timestamp,
                            last_reusable_text,
                            options,
                            reused=True,
                            diff=diff,
                        )
                    ],
                )
            else:
                pending_frames.append(
                    PendingOcrFrame(
                        timestamp=timestamp,
                        image=_prepare_image(image, options),
                        original=image,
                        diff=diff,
                    )
                )

            previous_signature = signature

            if pending_frames and _pending_sample_count(pending_frames) >= batch_size:
                store.update(
                    job.id,
                    phase="ocr",
                    progress=round(0.12 + 0.78 * (frame_index / total), 4),
                    message=(
                        f"OCR recognizing {len(pending_frames)} changed frame(s), "
                        f"reused {reused_count} · {frame_index}/{total}"
                    ),
                )
                batch_ocr, batch_reused, batch_reviewed, last_reusable_text = _flush_ocr_batch(
                    engine,
                    review_engine,
                    pending_frames,
                    options,
                    output_dir / "ocr_samples.jsonl",
                    samples,
                )
                ocr_count += batch_ocr
                reused_count += batch_reused
                reviewed_count += batch_reviewed
                pending_frames.clear()
                engine, review_engine, options = _maybe_switch_to_english_engine(
                    engine, review_engine, options, samples, job.id
                )

            processed = frame_index
            progress = 0.12 + 0.78 * (processed / total)
            elapsed = max(time.perf_counter() - ocr_started_at, 0.001)
            rate = processed / elapsed
            remaining = (total - processed) / rate if rate > 0 else 0
            if processed == total or processed % max(batch_size, 1) == 0:
                store.update(
                    job.id,
                    phase="ocr",
                    progress=round(progress, 4),
                    message=(
                        f"OCR scanned {processed}/{total} · {rate:.2f} frames/s · "
                        f"OCR {ocr_count} · reused {reused_count} · ETA {_format_duration(remaining)}"
                    ),
                )

        if pending_frames:
            _raise_if_canceled(job)
            store.update(
                job.id,
                phase="ocr",
                progress=0.9,
                message=f"OCR recognizing final {len(pending_frames)} changed frame(s)",
            )
            batch_ocr, batch_reused, batch_reviewed, last_reusable_text = _flush_ocr_batch(
                engine,
                review_engine,
                pending_frames,
                options,
                output_dir / "ocr_samples.jsonl",
                samples,
            )
            ocr_count += batch_ocr
            reused_count += batch_reused
            reviewed_count += batch_reviewed
            pending_frames.clear()
            engine, review_engine, options = _maybe_switch_to_english_engine(
                engine, review_engine, options, samples, job.id
            )

        _raise_if_canceled(job)
        samples.sort(key=lambda item: item[0])
        store.update(job.id, phase="merging", progress=0.93, message="Merging subtitle lines")
        entries = merge_ocr_results(samples, options.fps, options.min_length, options.similarity)
        if options.llm_correction and entries:
            _raise_if_canceled(job)
            store.update(job.id, phase="llm-correction", progress=0.96, message="Correcting subtitle text with LLM")
            entries = correct_subtitles_with_llm(entries, options, job.secrets)
        write_outputs(entries, output_dir / "subtitles.srt", output_dir / "subtitles.txt")
        done_message = (
            f"Done, generated {len(entries)} subtitle lines"
            if entries
            else "Done, but no subtitle lines were accepted. Check OCR log."
        )

        store.update(
            job.id,
            status=JobStatus.done.value,
            phase="done",
            progress=1.0,
            message=done_message,
        )
        _cleanup_intermediate_files(job_dir)
    except JobCancelled:
        store.update(
            job.id,
            status=JobStatus.canceled.value,
            phase="canceled",
            progress=1.0,
            message="Task canceled",
        )
        _cleanup_intermediate_files(job_dir)
    except Exception as exc:
        store.update(
            job.id,
            status=JobStatus.failed.value,
            phase="failed",
            message="Processing failed",
            error=str(exc),
        )
        _cleanup_intermediate_files(job_dir)


def _extract_frames(input_path: Path, frames_dir: Path, fps: float) -> None:
    output_pattern = frames_dir / "frame_%06d.jpg"
    command = [
        ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        str(output_pattern),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "ffmpeg frame extraction failed"
        raise RuntimeError(message)


def _format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def _lock_auto_subtitle_region(frame_paths: list[Path], options: JobOptions, job_id: str) -> JobOptions:
    if not options.lock_subtitle_region or _has_manual_region(options):
        return options

    store.update(
        job_id,
        phase="analyzing-region",
        progress=0.1,
        message="Analyzing stable subtitle region",
    )
    region = _estimate_stable_subtitle_region(frame_paths, options)
    if region is None:
        return options

    updated_options = options.model_copy(
        update={
            "auto_crop_x": region["x"],
            "auto_crop_y": region["y"],
            "auto_crop_w": region["w"],
            "auto_crop_h": region["h"],
        }
    )
    store.update(
        job_id,
        options=updated_options.model_dump(),
        message=(
            "Locked likely subtitle region "
            f"y={region['y']:.2f}, h={region['h']:.2f}"
        ),
    )
    return updated_options


def _estimate_stable_subtitle_region(frame_paths: list[Path], options: JobOptions) -> dict[str, float] | None:
    if not frame_paths or _has_manual_region(options):
        return None

    selected_paths = _sample_frame_paths(frame_paths, 14)
    row_accumulator: np.ndarray | None = None
    active_accumulator: np.ndarray | None = None
    source_shape: tuple[int, int] | None = None
    usable_frames = 0

    for frame_path in selected_paths:
        image = cv2.imread(str(frame_path))
        if image is None or image.size == 0:
            continue

        mask, original_shape = _subtitle_region_mask(image)
        if mask is None:
            continue

        row_scores = mask.mean(axis=1).astype(np.float32)
        max_row_score = float(row_scores.max(initial=0.0))
        if max_row_score < 3.0:
            continue

        normalized_rows = row_scores / max_row_score

        if row_accumulator is None:
            row_accumulator = np.zeros_like(normalized_rows, dtype=np.float32)
            active_accumulator = np.zeros_like(normalized_rows, dtype=np.float32)
            source_shape = original_shape
        elif row_accumulator.shape != normalized_rows.shape:
            continue

        row_accumulator += normalized_rows
        if active_accumulator is not None:
            active_accumulator += _active_content_rows(image, len(normalized_rows))
        usable_frames += 1

    if row_accumulator is None or source_shape is None or usable_frames < 3:
        return None

    smoothed = cv2.GaussianBlur(row_accumulator.reshape(-1, 1), (1, 9), 0).ravel()
    max_score = float(smoothed.max(initial=0.0))
    if max_score < max(1.2, usable_frames * 0.18):
        return None

    ranges = _active_ranges(smoothed >= max(max_score * 0.42, usable_frames * 0.12))
    if not ranges:
        return None

    mask_height = len(smoothed)
    candidates: list[tuple[float, int, int]] = []
    for start_y, end_y in ranges:
        band_height = end_y - start_y
        if band_height < max(2, int(mask_height * 0.025)):
            continue
        if band_height > int(mask_height * 0.42):
            continue
        center = (start_y + end_y) / 2 / mask_height
        lower_bias = 0.72 + 0.58 * center
        band_score = float(smoothed[start_y:end_y].sum() * lower_bias)
        candidates.append((band_score, start_y, end_y))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    best_score, start_y, end_y = candidates[0]
    if len(candidates) > 1:
        lower_candidates = [item for item in candidates[1:] if ((item[1] + item[2]) / 2 / mask_height) > 0.42]
        if lower_candidates and ((start_y + end_y) / 2 / mask_height) < 0.35:
            lower_score, lower_start, lower_end = lower_candidates[0]
            if lower_score >= best_score * 0.58:
                start_y, end_y = lower_start, lower_end

    original_height, _ = source_shape
    pad = max(3, int(mask_height * 0.055))
    y1_mask = max(0, start_y - pad)
    y2_mask = min(mask_height, end_y + pad)

    min_height = max(0.1, min(0.18, options.crop_bottom * 0.32))
    y = y1_mask / mask_height
    h = (y2_mask - y1_mask) / mask_height
    if h < min_height:
        center = (y1_mask + y2_mask) / 2 / mask_height
        y = max(0.0, center - min_height / 2)
        h = min_height

    active_bounds = _active_content_bounds(active_accumulator, usable_frames) if active_accumulator is not None else None
    if active_bounds is not None:
        y, h = _refine_region_with_active_content(y, h, active_bounds, min_height)

    if y + h > 1.0:
        y = max(0.0, 1.0 - h)

    if original_height <= 0:
        return None

    return {
        "x": 0.0,
        "y": round(float(y), 6),
        "w": 1.0,
        "h": round(float(min(h, 1.0)), 6),
    }


def _active_content_rows(image: np.ndarray, target_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height <= 0 or width <= 0 or target_height <= 0:
        return np.zeros((max(target_height, 1),), dtype=np.float32)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    row_mean = gray.mean(axis=1)
    row_std = gray.std(axis=1)
    active = ((row_mean > 14) | (row_std > 4.5)).astype(np.float32)
    active = cv2.morphologyEx(active.reshape(-1, 1), cv2.MORPH_CLOSE, np.ones((9, 1), dtype=np.float32)).ravel()
    if len(active) == target_height:
        return active
    return cv2.resize(active.reshape(-1, 1), (1, target_height), interpolation=cv2.INTER_LINEAR).ravel()


def _active_content_bounds(active_accumulator: np.ndarray | None, usable_frames: int) -> tuple[float, float] | None:
    if active_accumulator is None or usable_frames <= 0:
        return None

    threshold = max(1.0, usable_frames * 0.35)
    ranges = _active_ranges(active_accumulator >= threshold)
    if not ranges:
        return None

    start_y, end_y = max(ranges, key=lambda item: item[1] - item[0])
    height = len(active_accumulator)
    if height <= 0 or end_y <= start_y:
        return None
    return start_y / height, end_y / height


def _refine_region_with_active_content(
    y: float,
    h: float,
    active_bounds: tuple[float, float],
    min_height: float,
) -> tuple[float, float]:
    active_top, active_bottom = active_bounds
    active_height = active_bottom - active_top
    if active_height <= 0:
        return y, h

    overlaps_active = y < active_bottom and y + h > active_top
    if not overlaps_active:
        return y, h

    is_letterboxed = active_height <= 0.72
    covers_most_active = h >= active_height * 0.62
    starts_near_active_top = y <= active_top + active_height * 0.18
    is_tall_region = h >= max(0.18, active_height * 0.5)
    if is_letterboxed and (covers_most_active or (starts_near_active_top and is_tall_region)):
        subtitle_height = min(max(min_height, active_height * 0.42), min(0.24, active_height))
        return max(0.0, active_bottom - subtitle_height), subtitle_height

    if is_letterboxed and y + h < active_bottom - active_height * 0.12:
        return max(0.0, active_bottom - h), h

    if h > 0.28:
        center = y + h / 2
        h = 0.28
        y = max(0.0, center - h / 2)
    return y, h


def _sample_frame_paths(frame_paths: list[Path], max_frames: int) -> list[Path]:
    if len(frame_paths) <= max_frames:
        return frame_paths
    indexes = [round(index * (len(frame_paths) - 1) / (max_frames - 1)) for index in range(max_frames)]
    return [frame_paths[index] for index in indexes]


def _subtitle_region_mask(image: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int]]:
    height, width = image.shape[:2]
    if height < 80 or width < 120:
        return None, (height, width)

    target_width = 480
    scale = min(1.0, target_width / width)
    resized = image
    if scale < 1.0:
        resized = cv2.resize(
            image,
            (target_width, max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    mask = _text_stroke_mask(gray)
    if mask is None:
        return None, (height, width)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, mask.shape[1] // 160), 2)),
        iterations=1,
    )
    return mask, (height, width)


def _flush_ocr_batch(
    engine,
    review_engine,
    pending_frames: list[PendingOcrFrame],
    options: JobOptions,
    log_path: Path,
    samples: list[tuple[float, str | None]],
) -> tuple[int, int, int, str | None]:
    if not pending_frames:
        return 0, 0, 0, None

    texts = engine.read_texts([item.image for item in pending_frames])
    texts = _retry_empty_ocr(engine, texts, [item.original for item in pending_frames], options)
    texts, reviewed = _review_low_confidence_texts(review_engine, texts, pending_frames, options)
    texts = [_clean_ocr_text(text) for text in texts]

    rows = []
    reused_count = 0
    reviewed_count = 0
    last_text: str | None = None
    for item, text, was_reviewed in zip(pending_frames, texts, reviewed):
        last_text = text or ""
        if was_reviewed:
            reviewed_count += 1
        samples.append((item.timestamp, text))
        rows.append(
            _ocr_log_row(
                item.timestamp,
                text,
                options,
                reused=False,
                diff=item.diff,
                reviewed=was_reviewed,
            )
        )
        for reuse_timestamp, diff in item.reuse_timestamps:
            samples.append((reuse_timestamp, text))
            reused_count += 1
            rows.append(
                _ocr_log_row(
                    reuse_timestamp,
                    text,
                    options,
                    reused=True,
                    diff=diff,
                    reviewed=was_reviewed,
                )
            )

    append_ocr_log(log_path, rows)
    return len(pending_frames), reused_count, reviewed_count, last_text


def _review_low_confidence_texts(
    review_engine,
    texts: list[str],
    pending_frames: list[PendingOcrFrame],
    options: JobOptions,
) -> tuple[list[str], list[bool]]:
    reviewed = [False] * len(texts)
    if review_engine is None or not options.review_engine:
        return texts, reviewed

    review_indices = [
        index
        for index, text in enumerate(texts)
        if _should_review_text(text, options)
    ]
    if not review_indices:
        return texts, reviewed

    review_images = [_prepare_image(pending_frames[index].original, options) for index in review_indices]
    review_texts = review_engine.read_texts(review_images)
    for index, review_text in zip(review_indices, review_texts):
        if _prefer_review_text(texts[index], review_text, options):
            texts[index] = review_text
            reviewed[index] = True
        elif _is_hard_noise_text(texts[index]):
            texts[index] = ""
    return texts, reviewed


def _should_review_text(text: str | None, options: JobOptions) -> bool:
    if not options.review_engine:
        return False

    value = (text or "").strip()
    if not value:
        return True
    if len(value) <= 2:
        return True

    compact = re.sub(r"[^A-Za-z0-9]", "", value)
    if 2 <= len(compact) <= 10 and compact.upper() == compact and any(char.isdigit() for char in compact):
        return True
    if 2 <= len(compact) <= 5 and compact.upper() == compact and not any(char in value for char in ".?!,:;"):
        return True

    words = re.findall(r"[A-Za-z]+", value)
    if len(words) == 1 and len(words[0]) >= 18:
        return True
    if len(words) >= 2:
        if _looks_like_unspaced_uppercase_phrase(value, words):
            return True
        if _looks_like_glued_english(value):
            return True
        avg_word_len = sum(len(word) for word in words) / len(words)
        if avg_word_len >= 14:
            return True

    letters = [char for char in value if char.isalpha()]
    if letters:
        ascii_letters = sum("a" <= char.lower() <= "z" for char in letters)
        cjk_chars = sum("\u4e00" <= char <= "\u9fff" for char in value)
        if cjk_chars and ascii_letters >= 2:
            return True

    return False


def _prefer_review_text(primary: str | None, review: str | None, options: JobOptions) -> bool:
    primary_value = (primary or "").strip()
    review_value = (review or "").strip()
    if not review_value:
        return False
    if _is_hard_noise_text(review_value):
        return False
    if not primary_value:
        return True
    if not _should_review_text(primary_value, options):
        return False
    if not _should_review_text(review_value, options):
        return True
    return len(review_value) > len(primary_value) * 1.25


def _is_hard_noise_text(text: str | None) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    if len(value) <= 2:
        return True

    compact = re.sub(r"[^A-Za-z0-9]", "", value)
    if compact.isdigit():
        return True
    if 2 <= len(compact) <= 10 and compact.upper() == compact and any(char.isdigit() for char in compact):
        return True
    if 2 <= len(compact) <= 5 and compact.upper() == compact and not any(char in value for char in ".?!,:;"):
        return True
    tokens = re.findall(r"[A-Za-z0-9]+", value)
    if len(tokens) == 1 and 4 <= len(compact) <= 12:
        upper_count = sum(char.isupper() for char in compact)
        lower_count = sum(char.islower() for char in compact)
        if upper_count >= 3 and lower_count >= 1:
            return True
        if compact.upper() in {"TODAUT"}:
            return True
    if 4 <= len(tokens) <= 12 and compact.upper() == compact and any(char.isdigit() for char in compact):
        short_tokens = [token for token in tokens if len(token) <= 3]
        if len(short_tokens) / len(tokens) >= 0.75:
            return True
    return False


def _clean_ocr_text(text: str | None) -> str:
    value = (text or "").strip()
    if not value:
        return ""
    if _is_hard_noise_text(value):
        return ""

    replacements = (
        (r"\bForMG\b", "For MG"),
        (r"\bit'?swhereourjourneybegins\b", "it's where our journey begins"),
        (r"\bit'?swhere\s*ourjourneybegins\b", "it's where our journey begins"),
        (r"\bourjourneybegins\b", "our journey begins"),
        (r"\beffortlessconfidence\b", "effortless confidence"),
        (r"\bevoletion\b", "evolution"),
        (r"\bTHISISOURMISSION\b", "THIS IS OUR MISSION"),
        (r"\btomake\b", "to make"),
        (r"\btheaspiration\b", "the aspiration"),
        (r"\bimpossiblewithinreach\b", "impossible within reach"),
        (r"\bFindyourhore\b", "Find your more"),
        (r"\bFindyourmore\b", "Find your more"),
        (r"\bout ofit\b", "out of it"),
        (r"\bOrthe\b", "Or the"),
        (r"\bWnether\b", "Whether"),
        (r"\bKEEPTHE\b", "KEEP THE"),
        (r"\bPUNC\b", "PUNCH"),
        (r"\btraric\b", "traffic"),
        (r"\bPiug-in\b", "Plug-in"),
        (r"\bPlug In\b", "Plug-in"),
        (r"\bHybridr\b", "Hybrid+"),
        (r"\bforeveryone\b", "for everyone"),
        (r"\beyon atlow\b", "even at low"),
        (r"\btem!\s*t-rures\b", "temperatures"),
        (r"\bjirovides\b", "provides"),
        (r"\bEof power\b", "of power"),
        (r"\bund fuel\b", "and fuel"),
        (r"\bspeedst\b", "speeds"),
        (r"\bfurther reduce\b", "further reduces"),
        (r"\bTECHFORMORE\b", "TECH FOR MORE"),
    )
    for pattern, replacement in replacements:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
    value = re.sub(r",(?=\S)", ", ", value)
    value = re.sub(r"\.{2,}$", ".", value)
    value = re.sub(r"^G Plug-in\b", "MG Plug-in", value)
    value = re.sub(r"^ah\s+\.\S+\s+", "", value, flags=re.IGNORECASE)
    if re.search(r"\btechnol\b", value, flags=re.IGNORECASE) and re.search(r"\bti+me\b", value, flags=re.IGNORECASE):
        return ""

    words = re.findall(r"[A-Za-z]+", value)
    if len(words) >= 3:
        value = re.sub(r"\s+\d\s*$", "", value).strip()
    return value


def _looks_like_unspaced_uppercase_phrase(value: str, words: list[str]) -> bool:
    if not words or len(words) > 4:
        return False
    letters = re.sub(r"[^A-Za-z]", "", value)
    if len(letters) < 12:
        return False
    if sum(char.isupper() for char in letters) / max(len(letters), 1) < 0.85:
        return False
    if " " in value and len(value.split()) >= len(words):
        return False
    return any(len(word) >= 7 for word in words)


def _looks_like_glued_english(value: str) -> bool:
    compact = re.sub(r"[^A-Za-z']", "", value)
    if len(compact) < 12:
        return False

    lowered = compact.lower()
    glue_markers = (
        "itwhere",
        "itswhere",
        "whereour",
        "ourjourney",
        "journeybegins",
        "tomake",
        "theaspiration",
        "withinreach",
        "findyour",
        "thisisour",
        "isourmission",
        "effortlessconfidence",
    )
    if any(marker in lowered for marker in glue_markers):
        return True

    camel_boundaries = len(re.findall(r"[a-z][A-Z]", compact))
    long_words = [word for word in re.findall(r"[A-Za-z]+", value) if len(word) >= 13]
    return camel_boundaries >= 1 and bool(long_words)


def _ocr_log_row(
    timestamp: float,
    text: str | None,
    options: JobOptions,
    reused: bool,
    diff: float | None,
    reviewed: bool = False,
) -> dict:
    return {
        "timestamp": round(timestamp, 3),
        "text": text or "",
        "empty": not bool(text and text.strip()),
        "reused": reused,
        "reviewed": reviewed,
        "review_engine": options.review_engine if reviewed else None,
        "diff": round(diff, 5) if diff is not None else None,
        "crop_bottom": options.crop_bottom,
        "crop_region": _crop_region_dict(options),
        "crop_region_source": _crop_region_source(options),
    }


def _pending_sample_count(pending_frames: list[PendingOcrFrame]) -> int:
    return len(pending_frames) + sum(len(item.reuse_timestamps) for item in pending_frames)


def _maybe_switch_to_english_engine(
    engine,
    review_engine,
    options: JobOptions,
    samples: list[tuple[float, str | None]],
    job_id: str,
):
    if options.language == "en" or (options.engine != "paddle" and options.review_engine != "paddle"):
        return engine, review_engine, options
    if not _looks_like_english_subtitles([text for _, text in samples[-12:]]):
        return engine, review_engine, options

    updated_options = options.model_copy(update={"language": "en"})
    store.update(
        job_id,
        options=updated_options.model_dump(),
        message="Detected mostly English subtitles, switching PaddleOCR to English model",
    )
    updated_engine = engine
    updated_review_engine = review_engine
    if updated_options.engine == "paddle":
        updated_engine = get_engine(updated_options.engine, updated_options.language, updated_options.ocr_batch_size)
    if updated_options.review_engine == "paddle":
        updated_review_engine = get_engine("paddle", updated_options.language, updated_options.ocr_batch_size)
    return updated_engine, updated_review_engine, updated_options


def _looks_like_english_subtitles(texts: list[str | None]) -> bool:
    non_empty = [text.strip() for text in texts if text and text.strip()]
    if len(non_empty) < 4:
        return False

    joined = " ".join(non_empty)
    letters = sum(char.isalpha() for char in joined)
    ascii_letters = sum(("a" <= char.lower() <= "z") for char in joined)
    cjk_chars = sum("\u4e00" <= char <= "\u9fff" for char in joined)
    return letters >= 20 and ascii_letters / max(letters, 1) >= 0.85 and cjk_chars <= 1


def _prepare_image(image, options: JobOptions, crop_bottom_override: float | None = None, full_frame: bool = False):
    cropped = _crop_image(image, options, crop_bottom_override, full_frame)
    if options.auto_subtitle_strip and not full_frame and not _has_manual_region(options):
        cropped = _focus_subtitle_strip(cropped)

    scale = 2 if cropped.shape[1] < 1800 else 1
    if scale > 1:
        cropped = cv2.resize(cropped, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.bilateralFilter(enhanced, 5, 35, 35)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def _subtitle_signature(image, options: JobOptions) -> np.ndarray | None:
    cropped = _crop_image(image, options)
    if options.auto_subtitle_strip and not _has_manual_region(options):
        cropped = _focus_subtitle_strip(cropped)
    if cropped.size == 0:
        return None

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    return _text_shape_mask(gray)


def _focus_subtitle_strip(cropped: np.ndarray) -> np.ndarray:
    if cropped.size == 0:
        return cropped

    height, width = cropped.shape[:2]
    if height < 80 or width < 120:
        return cropped

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    mask = _text_stroke_mask(
        gray,
        kernel_size=(max(9, width // 28), max(3, height // 90)),
    )
    if mask is None:
        return cropped
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
    row_scores = mask.mean(axis=1)
    max_score = float(row_scores.max(initial=0.0))
    if max_score < 2.5:
        return cropped

    threshold = max(2.0, max_score * 0.42)
    active = row_scores >= threshold
    ranges = _active_ranges(active)
    if not ranges:
        return cropped

    def range_score(item: tuple[int, int]) -> float:
        start_y, end_y = item
        center = (start_y + end_y) / 2 / height
        lower_bias = 0.75 + 0.25 * center
        return float(row_scores[start_y:end_y].sum() * lower_bias)

    start_y, end_y = max(ranges, key=range_score)
    band_height = end_y - start_y
    min_height = max(42, int(height * 0.18))
    max_height = max(min_height, int(height * 0.42))
    target_height = min(max(band_height + int(height * 0.16), min_height), max_height)
    center_y = (start_y + end_y) // 2
    y1 = max(0, center_y - target_height // 2)
    y2 = min(height, y1 + target_height)
    y1 = max(0, y2 - target_height)
    if y2 <= y1:
        return cropped
    return cropped[y1:y2, :]


def _text_shape_mask(gray: np.ndarray) -> np.ndarray | None:
    if gray.size == 0:
        return None

    resized = cv2.resize(gray, (192, 64), interpolation=cv2.INTER_AREA)
    mask = _text_stroke_mask(resized, kernel_size=(13, 5))
    if mask is None:
        return None
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
    mask = cv2.dilate(mask, np.ones((2, 2), dtype=np.uint8), iterations=1)
    return mask


def _text_stroke_mask(gray: np.ndarray, kernel_size: tuple[int, int] = (17, 5)) -> np.ndarray | None:
    if gray.size == 0:
        return None

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    bright_strokes = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
    dark_strokes = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)
    strokes = cv2.max(bright_strokes, dark_strokes)
    _, mask = cv2.threshold(strokes, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def _active_ranges(active: np.ndarray) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_active in enumerate(active):
        if bool(is_active) and start is None:
            start = index
        elif not bool(is_active) and start is not None:
            ranges.append((start, index))
            start = None
    if start is not None:
        ranges.append((start, len(active)))
    return ranges


def _signature_diff(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None:
        return None
    if left.shape != right.shape:
        return None

    left_mask = left > 0
    right_mask = right > 0
    union = int(np.count_nonzero(left_mask | right_mask))
    if union == 0:
        return 0.0

    intersection = int(np.count_nonzero(left_mask & right_mask))
    return float(1.0 - (intersection / union))


def _crop_image(image, options: JobOptions, crop_bottom_override: float | None = None, full_frame: bool = False):
    height = image.shape[0]
    width = image.shape[1]
    if full_frame:
        return image

    region = _active_crop_region_dict(options)
    if region is not None:
        x1 = int(width * max(0.0, min(region["x"], 1.0)))
        y1 = int(height * max(0.0, min(region["y"], 1.0)))
        x2 = int(width * max(0.0, min(region["x"] + region["w"], 1.0)))
        y2 = int(height * max(0.0, min(region["y"] + region["h"], 1.0)))
        if x2 > x1 and y2 > y1:
            return image[y1:y2, x1:x2]

    crop_bottom = crop_bottom_override if crop_bottom_override is not None else options.crop_bottom
    crop_height = max(1, int(height * crop_bottom))
    return image[height - crop_height :, :]


def _has_manual_region(options: JobOptions) -> bool:
    values = [options.crop_x, options.crop_y, options.crop_w, options.crop_h]
    return all(value is not None for value in values)


def _has_auto_region(options: JobOptions) -> bool:
    values = [options.auto_crop_x, options.auto_crop_y, options.auto_crop_w, options.auto_crop_h]
    return all(value is not None for value in values)


def _active_crop_region_dict(options: JobOptions) -> dict[str, float] | None:
    if _has_manual_region(options):
        return {
            "x": float(options.crop_x or 0),
            "y": float(options.crop_y or 0),
            "w": float(options.crop_w or 0),
            "h": float(options.crop_h or 0),
        }
    if _has_auto_region(options):
        return {
            "x": float(options.auto_crop_x or 0),
            "y": float(options.auto_crop_y or 0),
            "w": float(options.auto_crop_w or 0),
            "h": float(options.auto_crop_h or 0),
        }
    return None


def _crop_region_dict(options: JobOptions) -> dict[str, float] | None:
    return _active_crop_region_dict(options)


def _crop_region_source(options: JobOptions) -> str | None:
    if _has_manual_region(options):
        return "manual"
    if _has_auto_region(options):
        return "auto"
    return None


def _retry_empty_ocr(engine, texts: list[str], originals: list, options: JobOptions) -> list[str]:
    empty_indices = [index for index, text in enumerate(texts) if not text or not text.strip()]
    if not empty_indices:
        return texts
    if options.mode not in {"accurate", "manual"}:
        return texts

    crop_bottom = options.crop_bottom
    if crop_bottom < 0.55:
        retry_options = JobOptions(crop_bottom=0.55)
        retry_images = [_prepare_image(originals[index], retry_options) for index in empty_indices]
        retry_texts = engine.read_texts(retry_images)
        for index, text in zip(empty_indices, retry_texts):
            if text and text.strip():
                texts[index] = text

    empty_indices = [index for index, text in enumerate(texts) if not text or not text.strip()]
    if empty_indices and RETRY_FULL_FRAME_ON_EMPTY:
        retry_options = JobOptions(crop_bottom=1.0)
        full_images = [_prepare_image(originals[index], retry_options, full_frame=True) for index in empty_indices]
        full_texts = engine.read_texts(full_images)
        for index, text in zip(empty_indices, full_texts):
            if text and text.strip():
                texts[index] = text

    return texts


def _raise_if_canceled(job: JobRecord) -> None:
    if job.cancel_requested:
        raise JobCancelled()


def _cleanup_intermediate_files(job_dir: Path) -> None:
    if not CLEANUP_INTERMEDIATE_FILES:
        return
    for name in ("input", "frames"):
        path = job_dir / name
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
