from __future__ import annotations

import json
import threading
import time
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
    created_at: float = field(default_factory=time.time)
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


_RECORD_FIELDS = frozenset(JobRecord.__dataclass_fields__)


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}

    def load_from_disk(self, jobs_dir: Path) -> int:
        """扫描 jobs 目录，从已有的 job.json 恢复任务记录。
        服务重启前处于 running/queued 的任务会被标为 failed。
        返回成功加载的任务数。"""
        loaded = 0
        if not jobs_dir.is_dir():
            return loaded
        paths = sorted(jobs_dir.glob("*/job.json"), key=lambda p: p.stat().st_mtime)
        for job_json in paths:
            try:
                data = json.loads(job_json.read_text(encoding="utf-8", errors="replace"))
                data.pop("secrets", None)
                # 将上次运行中断的任务标为失败
                if data.get("status") in (JobStatus.running.value, JobStatus.queued.value):
                    data["status"] = JobStatus.failed.value
                    data["phase"] = "failed"
                    data["message"] = "服务重启，任务中断"
                    data["error"] = "Server restarted while task was running."
                    data["progress"] = 1.0
                    job_json.write_text(
                        json.dumps({k: v for k, v in data.items()}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                record = JobRecord(**{k: v for k, v in data.items() if k in _RECORD_FIELDS})
                with self._lock:
                    self._jobs[record.id] = record
                loaded += 1
            except Exception:
                continue
        return loaded

    def add(self, job: JobRecord) -> None:
        with self._lock:
            self._jobs[job.id] = job
            self._write(job)

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_recent(self, limit: int = 30) -> list[JobRecord]:
        """返回最近 limit 条任务，按创建时间倒序。"""
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return jobs[:limit]

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
            created_at=job.created_at,
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
