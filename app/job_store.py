from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.models import JobInfo, JobOptions, JobStatus


@dataclass
class JobRecord:
    id: str
    filename: str
    input_path: str
    job_dir: str
    options: dict[str, Any]
    status: str = JobStatus.queued.value
    progress: float = 0.0
    phase: str = "queued"
    message: str = ""
    error: str | None = None
    cancel_requested: bool = False
    secrets: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def srt_path(self) -> Path:
        return Path(self.job_dir) / "output" / "subtitles.srt"

    @property
    def txt_path(self) -> Path:
        return Path(self.job_dir) / "output" / "subtitles.txt"

    @property
    def preview_path(self) -> Path:
        return Path(self.job_dir) / "output" / "debug_preview.jpg"

    @property
    def ocr_log_path(self) -> Path:
        return Path(self.job_dir) / "output" / "ocr_samples.jsonl"


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}

    def add(self, job: JobRecord) -> None:
        with self._lock:
            self._jobs[job.id] = job
            self._write(job)

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in changes.items():
                setattr(job, key, value)
            self._write(job)

    def cancel(self, job_id: str) -> JobRecord | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status in {JobStatus.done.value, JobStatus.failed.value, JobStatus.canceled.value}:
                return job
            job.cancel_requested = True
            if job.status == JobStatus.queued.value:
                job.status = JobStatus.canceled.value
                job.phase = "canceled"
                job.progress = 1.0
                job.message = "Task canceled"
            else:
                job.message = "Cancel requested"
            self._write(job)
            return job

    def to_info(self, job: JobRecord) -> JobInfo:
        done = job.status == JobStatus.done.value
        return JobInfo(
            id=job.id,
            status=JobStatus(job.status),
            filename=job.filename,
            engine=str(job.options.get("engine", "")),
            progress=job.progress,
            phase=job.phase,
            message=job.message,
            error=job.error,
            srt_url=f"/api/jobs/{job.id}/download/srt" if done and job.srt_path.exists() else None,
            txt_url=f"/api/jobs/{job.id}/download/txt" if done and job.txt_path.exists() else None,
            preview_url=f"/api/jobs/{job.id}/download/preview" if job.preview_path.exists() else None,
            ocr_log_url=f"/api/jobs/{job.id}/download/ocr-log" if job.ocr_log_path.exists() else None,
            options=JobOptions(**job.options),
            cancel_requested=job.cancel_requested,
        )

    def _write(self, job: JobRecord) -> None:
        path = Path(job.job_dir) / "job.json"
        data = asdict(job)
        data.pop("secrets", None)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


store = JobStore()
