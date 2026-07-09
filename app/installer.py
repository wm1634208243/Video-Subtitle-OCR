from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.config import DATA_DIR, ROOT_DIR


INSTALL_HEADER = "x-vso-action"
INSTALL_HEADER_VALUE = "install"


class InstallerError(RuntimeError):
    pass


class InstallerBusyError(InstallerError):
    pass


INSTALL_PROFILES: dict[str, dict[str, str]] = {
    "recommended": {
        "id": "recommended",
        "name": "Recommended",
        "engine": "paddle",
        "description": "Install PaddleOCR CPU, the recommended default OCR engine.",
    },
    "easyocr": {
        "id": "easyocr",
        "name": "EasyOCR",
        "engine": "easyocr",
        "description": "Install EasyOCR. This may download PyTorch and can take a while.",
    },
    "openvino": {
        "id": "openvino",
        "name": "OpenVINO",
        "engine": "openvino",
        "description": "Install RapidOCR with OpenVINO backend for Intel acceleration experiments.",
    },
    "onnxruntime": {
        "id": "onnxruntime",
        "name": "ONNXRuntime",
        "engine": "onnxruntime",
        "description": "Install RapidOCR with ONNXRuntime backend, a fast cross-platform CPU option.",
    },
    "tesseract": {
        "id": "tesseract",
        "name": "Tesseract",
        "engine": "tesseract",
        "description": "Install pytesseract and try to install the system Tesseract executable.",
    },
    "full": {
        "id": "full",
        "name": "Full",
        "engine": "",
        "description": "Install PaddleOCR, EasyOCR, and Tesseract support.",
    },
    "dev": {
        "id": "dev",
        "name": "Developer",
        "engine": "",
        "description": "Install lightweight dependencies for tests and CI.",
    },
    "core": {
        "id": "core",
        "name": "Core",
        "engine": "",
        "description": "Install the web app and processing basics without OCR engines.",
    },
}


@dataclass
class InstallJob:
    id: str
    profile: str
    profile_name: str
    status: str = "running"
    progress: float = 0.0
    message: str = "Starting installation"
    log: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    returncode: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["log"] = data["log"][-160:]
        return data


_lock = threading.Lock()
_current_job: InstallJob | None = None


def list_install_profiles() -> list[dict[str, str]]:
    return list(INSTALL_PROFILES.values())


def get_current_install() -> dict[str, Any] | None:
    with _lock:
        return _current_job.to_dict() if _current_job else None


def start_install(profile: str) -> dict[str, Any]:
    if profile not in INSTALL_PROFILES:
        raise InstallerError(f"Unknown install profile: {profile}")

    global _current_job
    with _lock:
        if _current_job and _current_job.status == "running":
            raise InstallerBusyError("Another installation is already running")
        config = INSTALL_PROFILES[profile]
        _current_job = InstallJob(id=uuid.uuid4().hex, profile=profile, profile_name=config["name"])
        job = _current_job

    worker = threading.Thread(target=_run_install, args=(job,), name=f"install-{profile}", daemon=True)
    worker.start()
    return job.to_dict()


def _run_install(job: InstallJob) -> None:
    try:
        _append_log(job, f"Installing profile: {job.profile_name}")
        _set_job(job, progress=0.05, message="Preparing installer")

        command = _install_command(job.profile)
        _append_log(job, " ".join(command))
        _set_job(job, progress=0.1, message="Installing dependencies")

        process = subprocess.Popen(
            command,
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=_creation_flags(),
        )

        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if line:
                _append_log(job, line)

        returncode = process.wait()
        if returncode != 0:
            _set_job(
                job,
                status="failed",
                progress=1.0,
                message=f"Installation failed with exit code {returncode}",
                finished_at=time.time(),
                returncode=returncode,
            )
            return

        _set_job(
            job,
            status="done",
            progress=1.0,
            message="Installation finished",
            finished_at=time.time(),
            returncode=returncode,
        )
    except Exception as exc:
        _append_log(job, str(exc))
        _set_job(job, status="failed", progress=1.0, message=str(exc), finished_at=time.time(), returncode=-1)


def _install_command(profile: str) -> list[str]:
    if os.name == "nt":
        powershell = _find_powershell()
        script_path = ROOT_DIR / "scripts" / "install.ps1"
        if not script_path.exists():
            raise InstallerError("scripts/install.ps1 was not found")
        return [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script_path),
            "-Profile",
            profile,
        ]

    bash = _find_bash()
    script_path = ROOT_DIR / "scripts" / "install.sh"
    if not script_path.exists():
        raise InstallerError("scripts/install.sh was not found")
    return [bash, str(script_path), "--profile", profile]


def _find_powershell() -> str:
    candidates = ["powershell", "pwsh"]
    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "-NoProfile", "-Command", "$PSVersionTable.PSVersion.Major"],
                capture_output=True,
                text=True,
                check=False,
                timeout=8,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            return candidate
    raise InstallerError("PowerShell was not found. Run install.bat from Windows instead.")


def _find_bash() -> str:
    for candidate in ("bash", "/bin/bash", "/usr/bin/bash"):
        path = shutil.which(candidate) if "/" not in candidate else candidate
        if path and Path(path).exists():
            return path
    raise InstallerError("bash was not found. Run scripts/install.sh manually with a POSIX shell.")


def _creation_flags() -> int:
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        return int(subprocess.CREATE_NO_WINDOW)
    return 0


def _append_log(job: InstallJob, line: str) -> None:
    with _lock:
        job.log.append(line)
        job.log = job.log[-300:]
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        log_path = DATA_DIR / "install.log"
        log_path.write_text("\n".join(job.log) + "\n", encoding="utf-8")


def _set_job(job: InstallJob, **changes: Any) -> None:
    with _lock:
        for key, value in changes.items():
            setattr(job, key, value)
