from __future__ import annotations

import queue
import json
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.autotune import build_auto_options
from app.config import (
    ALLOW_WEB_INSTALL,
    ALLOWED_VIDEO_SUFFIXES,
    CPU_THREADS,
    JOBS_DIR,
    MAX_UPLOAD_MB,
    RESOURCE_PLAN,
    STATIC_DIR,
    WORKER_COUNT,
    ensure_data_dirs,
)
from app.ffmpeg_tools import ffmpeg_exe
from app.installer import (
    INSTALL_HEADER,
    INSTALL_HEADER_VALUE,
    InstallerBusyError,
    InstallerError,
    get_current_install,
    list_install_profiles,
    start_install,
)
from app.job_store import JobRecord, store
from app.llm_correction import LlmCorrectionError, test_llm_connection
from app.models import (
    EngineInfo,
    HealthInfo,
    InstallJobInfo,
    InstallProfileInfo,
    JobInfo,
    JobOptions,
    LlmConfigInfo,
    LlmConfigUpdate,
    LlmTestResult,
    OcrLogInfo,
    OcrLogRow,
)
from app.ocr_engines import OcrEngineError, get_engine_status, list_engine_statuses
from app.processing import process_job
from app.user_config import get_llm_config, reset_llm_config, save_llm_config


APP_VERSION = "1.7.1"
job_queue: queue.Queue[JobRecord] = queue.Queue()


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_data_dirs()
    for index in range(WORKER_COUNT):
        worker = threading.Thread(target=_worker_loop, name=f"ocr-worker-{index + 1}", daemon=True)
        worker.start()
    yield


app = FastAPI(title="Video Subtitle OCR", version=APP_VERSION, lifespan=lifespan)


@app.get("/api/health", response_model=HealthInfo)
def health() -> HealthInfo:
    ffmpeg_path: str | None = None
    ffmpeg_error: str | None = None
    try:
        ffmpeg_path = ffmpeg_exe()
    except Exception as exc:
        ffmpeg_error = str(exc)

    return HealthInfo(
        status="ok" if ffmpeg_path else "degraded",
        version=APP_VERSION,
        queue_size=job_queue.qsize(),
        worker_count=WORKER_COUNT,
        worker_source=RESOURCE_PLAN.worker_source,
        max_workers=RESOURCE_PLAN.max_workers,
        cpu_count=RESOURCE_PLAN.cpu_count,
        cpu_threads=CPU_THREADS,
        total_memory_gb=RESOURCE_PLAN.total_memory_gb,
        available_memory_gb=RESOURCE_PLAN.available_memory_gb,
        estimated_worker_memory_gb=RESOURCE_PLAN.estimated_worker_memory_gb,
        max_upload_mb=MAX_UPLOAD_MB,
        ffmpeg_available=bool(ffmpeg_path),
        ffmpeg_path=ffmpeg_path,
        ffmpeg_error=ffmpeg_error,
    )


@app.get("/api/engines", response_model=list[EngineInfo])
def engines() -> list[EngineInfo]:
    return [EngineInfo(**engine) for engine in list_engine_statuses()]


@app.get("/api/install/profiles", response_model=list[InstallProfileInfo])
def install_profiles() -> list[InstallProfileInfo]:
    return [InstallProfileInfo(**profile) for profile in list_install_profiles()]


@app.get("/api/install/current", response_model=InstallJobInfo | None)
def current_install() -> InstallJobInfo | None:
    job = get_current_install()
    return InstallJobInfo(**job) if job else None


@app.get("/api/config/llm", response_model=LlmConfigInfo)
def read_llm_config() -> LlmConfigInfo:
    return LlmConfigInfo(**get_llm_config(include_secret=False))


@app.put("/api/config/llm", response_model=LlmConfigInfo)
def update_llm_config(config: LlmConfigUpdate) -> LlmConfigInfo:
    return LlmConfigInfo(**save_llm_config(config.model_dump()))


@app.post("/api/config/llm/reset", response_model=LlmConfigInfo)
def reset_saved_llm_config() -> LlmConfigInfo:
    return LlmConfigInfo(**reset_llm_config())


@app.post("/api/config/llm/test", response_model=LlmTestResult)
def test_saved_or_submitted_llm_config(config: LlmConfigUpdate) -> LlmTestResult:
    saved = get_llm_config(include_secret=True)
    provider = _normalize_llm_provider(config.provider or str(saved.get("provider") or "openai"))
    model = (config.model or str(saved.get("model") or "")).strip()
    base_url = (config.base_url or str(saved.get("base_url") or "")).strip()
    api_key = "" if provider == "ollama" else (config.api_key or str(saved.get("api_key") or "")).strip()
    try:
        result = test_llm_connection(provider, model, base_url, api_key)
    except LlmCorrectionError as exc:
        return LlmTestResult(ok=False, message=str(exc), provider=provider, model=model)
    except Exception as exc:
        return LlmTestResult(ok=False, message=f"Connection test failed: {exc}", provider=provider, model=model)
    return LlmTestResult(ok=True, message="LLM connection test passed.", **result)


@app.post("/api/install/{profile}", response_model=InstallJobInfo)
def install_engine(profile: str, request: Request) -> InstallJobInfo:
    _guard_install_request(request)
    try:
        job = start_install(profile)
    except InstallerBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InstallerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return InstallJobInfo(**job)


@app.post("/api/jobs", response_model=JobInfo)
async def create_job(
    file: UploadFile = File(...),
    mode: str = Form("balanced"),
    engine: str = Form("auto"),
    fps: float = Form(1.5),
    crop_bottom: float = Form(0.55),
    similarity: float = Form(0.72),
    min_length: int = Form(2),
    ocr_batch_size: int = Form(4),
    language: str = Form("ch"),
    skip_unchanged_frames: bool = Form(True),
    frame_diff_threshold: float = Form(0.2),
    lock_subtitle_region: bool = Form(True),
    crop_x: float | None = Form(None),
    crop_y: float | None = Form(None),
    crop_w: float | None = Form(None),
    crop_h: float | None = Form(None),
    llm_correction: bool = Form(False),
    llm_provider: str = Form("openai"),
    llm_model: str = Form(""),
    llm_base_url: str = Form(""),
    llm_api_key: str = Form(""),
) -> JobInfo:
    filename = Path(file.filename or "video").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported video format")
    language = _normalize_language(language)
    try:
        engine, review_engine = _resolve_engine_plan(engine, mode, language)
        engine_status = get_engine_status(engine)
    except OcrEngineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not engine_status.available:
        detail = f"{engine_status.name} is not available. {engine_status.reason or ''}"
        if engine_status.install_hint:
            detail = f"{detail} {engine_status.install_hint}".strip()
        raise HTTPException(status_code=400, detail=detail)

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / filename

    size = 0
    with input_path.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB}MB limit")
            out.write(chunk)
    try:
        saved_llm = get_llm_config(include_secret=True)
        effective_llm_provider = _normalize_llm_provider(llm_provider or str(saved_llm.get("provider") or "openai"))
        effective_llm_model = (llm_model or str(saved_llm.get("model") or "")).strip()
        effective_llm_base_url = (llm_base_url or str(saved_llm.get("base_url") or "")).strip()
        effective_llm_api_key = (llm_api_key or str(saved_llm.get("api_key") or "")).strip()
        if mode == "manual":
            options = JobOptions(
                engine=engine,
                review_engine=review_engine,
                mode="manual",
                auto_config=False,
                fps=fps,
                crop_bottom=crop_bottom,
                similarity=similarity,
                min_length=min_length,
                ocr_batch_size=ocr_batch_size,
                language=language,
                skip_unchanged_frames=skip_unchanged_frames,
                frame_diff_threshold=frame_diff_threshold,
                lock_subtitle_region=lock_subtitle_region,
                crop_x=crop_x,
                crop_y=crop_y,
                crop_w=crop_w,
                crop_h=crop_h,
                llm_correction=llm_correction,
                llm_provider=effective_llm_provider,
                llm_model=effective_llm_model,
                llm_base_url=effective_llm_base_url,
            )
        else:
            options = build_auto_options(input_path, mode, engine, language)
            options.review_engine = review_engine
            options.crop_x = crop_x
            options.crop_y = crop_y
            options.crop_w = crop_w
            options.crop_h = crop_h
            options.lock_subtitle_region = lock_subtitle_region
            options.llm_correction = llm_correction
            options.llm_provider = effective_llm_provider
            options.llm_model = effective_llm_model
            options.llm_base_url = effective_llm_base_url
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    record = JobRecord(
        id=job_id,
        filename=filename,
        input_path=str(input_path),
        job_dir=str(job_dir),
        options=options.model_dump(),
        secrets={"llm_api_key": effective_llm_api_key} if effective_llm_api_key else {},
    )
    store.add(record)
    job_queue.put(record)
    return store.to_info(record)


@app.get("/api/jobs/{job_id}", response_model=JobInfo)
def get_job(job_id: str) -> JobInfo:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return store.to_info(job)


@app.post("/api/jobs/{job_id}/cancel", response_model=JobInfo)
def cancel_job(job_id: str) -> JobInfo:
    job = store.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return store.to_info(job)


@app.get("/api/jobs/{job_id}/ocr-log", response_model=OcrLogInfo)
def read_ocr_log(job_id: str, limit: int = 80) -> OcrLogInfo:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.ocr_log_path.exists():
        return OcrLogInfo(rows=[], total=0)

    limit = min(max(limit, 1), 300)
    raw_lines = job.ocr_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows: list[OcrLogRow] = []
    for line in raw_lines[-limit:]:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(OcrLogRow(**data))
    return OcrLogInfo(rows=rows, total=len(raw_lines))


@app.get("/api/jobs/{job_id}/download/{kind}")
def download(job_id: str, kind: str) -> FileResponse:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    path_map = {
        "srt": job.srt_path,
        "txt": job.txt_path,
        "preview": job.preview_path,
        "ocr-log": job.ocr_log_path,
    }
    path = path_map.get(kind)
    if path is None:
        raise HTTPException(status_code=404, detail="Download type not found")
    if not path.exists():
        raise HTTPException(status_code=404, detail="File has not been generated")

    suffix = {"preview": "debug-preview.jpg", "ocr-log": "ocr-samples.jsonl"}.get(kind, kind)
    return FileResponse(path, filename=f"{Path(job.filename).stem}.{suffix}")


def _worker_loop() -> None:
    while True:
        job = job_queue.get()
        try:
            process_job(job)
        finally:
            job_queue.task_done()


def _guard_install_request(request: Request) -> None:
    client_host = request.client.host if request.client else ""
    if not ALLOW_WEB_INSTALL and client_host not in {"127.0.0.1", "::1", "localhost"}:
        raise HTTPException(status_code=403, detail="Install actions are local-only")
    if request.headers.get(INSTALL_HEADER) != INSTALL_HEADER_VALUE:
        raise HTTPException(status_code=403, detail="Install action header is required")


def _normalize_language(language: str) -> str:
    value = (language or "ch").strip().lower().replace("_", "-")
    if value in {"en", "eng", "english"}:
        return "en"
    if value in {"ch", "zh", "zh-cn", "ch-sim", "chi-sim", "chinese", "cn"}:
        return "ch"
    return "ch"


def _normalize_llm_provider(provider: str) -> str:
    value = (provider or "openai").strip().lower()
    if value in {"openai", "anthropic", "ollama", "openai-compatible"}:
        return value
    return "openai"


def _resolve_engine(engine: str, mode: str, language: str) -> str:
    primary, _ = _resolve_engine_plan(engine, mode, language)
    return primary


def _resolve_engine_plan(engine: str, mode: str, language: str) -> tuple[str, str]:
    requested = (engine or "auto").strip().lower()
    if requested != "auto":
        review_engine = "paddle" if _should_enable_paddle_review(requested, mode) else ""
        return requested, review_engine

    profile = mode if mode in {"fast", "balanced", "accurate"} else "balanced"
    if profile == "fast":
        candidates = ("openvino", "onnxruntime", "paddle", "easyocr", "tesseract")
    elif profile == "accurate":
        candidates = ("openvino", "onnxruntime", "paddle", "easyocr", "tesseract")
    else:
        candidates = ("openvino", "onnxruntime", "paddle", "easyocr", "tesseract")

    for candidate in candidates:
        try:
            if get_engine_status(candidate).available:
                review_engine = "paddle" if _should_enable_paddle_review(candidate, profile) else ""
                return candidate, review_engine
        except OcrEngineError:
            continue

    raise OcrEngineError("No available OCR engine. Install PaddleOCR, OpenVINO OCR, or ONNXRuntime OCR first.")


def _should_enable_paddle_review(engine: str, mode: str) -> bool:
    profile = mode if mode in {"fast", "balanced", "accurate", "manual"} else "balanced"
    if engine not in {"openvino", "onnxruntime"} or profile == "fast":
        return False
    try:
        return get_engine_status("paddle").available
    except OcrEngineError:
        return False


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
