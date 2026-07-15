from __future__ import annotations

import os
from pathlib import Path

from app.resource_planner import build_resource_plan


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
JOBS_DIR = DATA_DIR / "jobs"
MODELS_DIR = DATA_DIR / "models"
STATIC_DIR = ROOT_DIR / "app" / "static"

RESOURCE_PLAN = build_resource_plan()
MAX_UPLOAD_MB = int(os.getenv("VSO_MAX_UPLOAD_MB", "4096"))
WORKER_COUNT = RESOURCE_PLAN.worker_count
CPU_THREADS = RESOURCE_PLAN.cpu_threads
DEFAULT_PADDLE_BATCH_SIZE = max(1, int(os.getenv("VSO_PADDLE_BATCH_SIZE", "4")))
CLEANUP_INTERMEDIATE_FILES = os.getenv("VSO_CLEANUP_INTERMEDIATE_FILES", "1").lower() not in {"0", "false", "no"}
RETRY_FULL_FRAME_ON_EMPTY = os.getenv("VSO_RETRY_FULL_FRAME_ON_EMPTY", "0").lower() in {"1", "true", "yes"}
ALLOW_WEB_INSTALL = os.getenv("VSO_ALLOW_WEB_INSTALL", "0").lower() in {"1", "true", "yes"}
DEFAULT_FPS = 1.5
DEFAULT_CROP_BOTTOM = 0.55
DEFAULT_SIMILARITY = 0.72
DEFAULT_MIN_LENGTH = 2

ALLOWED_VIDEO_SUFFIXES = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".m4v",
    ".ts",
    ".flv",
    ".wmv",
    ".3gp",
    ".rmvb",
}


def ensure_data_dirs() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
