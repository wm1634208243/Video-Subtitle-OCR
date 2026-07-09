from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"
    canceled = "canceled"


class JobOptions(BaseModel):
    engine: Literal["paddle", "openvino", "onnxruntime", "easyocr", "tesseract"] = "paddle"
    review_engine: Literal["", "paddle"] = ""
    mode: Literal["fast", "balanced", "accurate", "manual"] = "balanced"
    auto_config: bool = True
    fps: float = Field(default=1.5, ge=0.1, le=10)
    crop_bottom: float = Field(default=0.55, ge=0.05, le=1.0)
    similarity: float = Field(default=0.72, ge=0.1, le=0.98)
    min_length: int = Field(default=2, ge=1, le=20)
    ocr_batch_size: int = Field(default=4, ge=1, le=32)
    language: str = "ch"
    skip_unchanged_frames: bool = True
    frame_diff_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    auto_subtitle_strip: bool = True
    lock_subtitle_region: bool = True
    crop_x: float | None = Field(default=None, ge=0.0, le=1.0)
    crop_y: float | None = Field(default=None, ge=0.0, le=1.0)
    crop_w: float | None = Field(default=None, ge=0.01, le=1.0)
    crop_h: float | None = Field(default=None, ge=0.01, le=1.0)
    auto_crop_x: float | None = Field(default=None, ge=0.0, le=1.0)
    auto_crop_y: float | None = Field(default=None, ge=0.0, le=1.0)
    auto_crop_w: float | None = Field(default=None, ge=0.01, le=1.0)
    auto_crop_h: float | None = Field(default=None, ge=0.01, le=1.0)
    llm_correction: bool = False
    llm_provider: Literal["openai", "anthropic", "ollama", "openai-compatible"] = "openai"
    llm_model: str = ""
    llm_base_url: str = ""


class JobInfo(BaseModel):
    id: str
    status: JobStatus
    filename: str
    engine: str
    progress: float = 0.0
    phase: str = "queued"
    message: str = ""
    error: str | None = None
    srt_url: str | None = None
    txt_url: str | None = None
    preview_url: str | None = None
    ocr_log_url: str | None = None
    options: JobOptions | None = None
    cancel_requested: bool = False


class OcrLogRow(BaseModel):
    timestamp: float
    text: str = ""
    empty: bool = False
    reused: bool = False
    reviewed: bool = False
    review_engine: str | None = None
    diff: float | None = None
    crop_bottom: float | None = None
    crop_region: dict[str, float] | None = None
    crop_region_source: str | None = None


class OcrLogInfo(BaseModel):
    rows: list[OcrLogRow] = Field(default_factory=list)
    total: int = 0


class EngineInfo(BaseModel):
    id: str
    name: str
    available: bool
    default: bool = False
    description: str = ""
    reason: str | None = None
    install_hint: str | None = None


class HealthInfo(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    queue_size: int
    worker_count: int
    worker_source: str = "auto"
    max_workers: int = 1
    cpu_count: int | None = None
    cpu_threads: int | None = None
    total_memory_gb: float | None = None
    available_memory_gb: float | None = None
    estimated_worker_memory_gb: float | None = None
    max_upload_mb: int
    ffmpeg_available: bool
    ffmpeg_path: str | None = None
    ffmpeg_error: str | None = None


class InstallProfileInfo(BaseModel):
    id: str
    name: str
    engine: str = ""
    description: str = ""


class InstallJobInfo(BaseModel):
    id: str
    profile: str
    profile_name: str
    status: Literal["running", "done", "failed"]
    progress: float = 0.0
    message: str = ""
    log: list[str] = Field(default_factory=list)
    started_at: float
    finished_at: float | None = None
    returncode: int | None = None


class LlmConfigInfo(BaseModel):
    enabled: bool = False
    provider: Literal["openai", "anthropic", "ollama", "openai-compatible"] = "openai"
    model: str = ""
    base_url: str = ""
    has_api_key: bool = False


class LlmConfigUpdate(BaseModel):
    enabled: bool = False
    provider: Literal["openai", "anthropic", "ollama", "openai-compatible"] = "openai"
    model: str = ""
    base_url: str = ""
    api_key: str = ""


class LlmTestResult(BaseModel):
    ok: bool
    message: str
    provider: str = ""
    model: str = ""
    corrected_text: str = ""
    elapsed_ms: int | None = None
