from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ResourcePlan:
    cpu_count: int
    total_memory_gb: float | None
    available_memory_gb: float | None
    cpu_threads: int
    worker_count: int
    worker_source: str
    max_workers: int
    estimated_worker_memory_gb: float


def build_resource_plan() -> ResourcePlan:
    cpu_count = os.cpu_count() or 4
    total_memory_gb, available_memory_gb = _memory_gb()
    cpu_threads = _configured_int("VSO_CPU_THREADS") or _auto_cpu_threads(cpu_count)
    max_workers = max(1, _configured_int("VSO_MAX_WORKERS") or _auto_max_workers(cpu_count))
    estimated_worker_memory_gb = max(1.0, _configured_float("VSO_ESTIMATED_WORKER_GB") or 4.5)

    forced_workers = _configured_int("VSO_WORKERS")
    if forced_workers:
        worker_count = max(1, min(forced_workers, max_workers))
        worker_source = "manual"
    else:
        worker_count = _auto_worker_count(
            cpu_count=cpu_count,
            cpu_threads=cpu_threads,
            max_workers=max_workers,
            available_memory_gb=available_memory_gb,
            total_memory_gb=total_memory_gb,
            estimated_worker_memory_gb=estimated_worker_memory_gb,
        )
        worker_source = "auto"

    return ResourcePlan(
        cpu_count=cpu_count,
        total_memory_gb=total_memory_gb,
        available_memory_gb=available_memory_gb,
        cpu_threads=cpu_threads,
        worker_count=worker_count,
        worker_source=worker_source,
        max_workers=max_workers,
        estimated_worker_memory_gb=estimated_worker_memory_gb,
    )


def _auto_cpu_threads(cpu_count: int) -> int:
    return min(max(4, cpu_count // 2), 12)


def _auto_max_workers(cpu_count: int) -> int:
    if cpu_count >= 20:
        return 3
    if cpu_count >= 12:
        return 2
    return 1


def _auto_worker_count(
    cpu_count: int,
    cpu_threads: int,
    max_workers: int,
    available_memory_gb: float | None,
    total_memory_gb: float | None,
    estimated_worker_memory_gb: float,
) -> int:
    by_cpu = max(1, cpu_count // max(4, min(cpu_threads, 8)))
    by_memory = max_workers
    memory_basis = available_memory_gb if available_memory_gb is not None else total_memory_gb
    if memory_basis is not None:
        reserve_gb = max(3.5, (total_memory_gb or memory_basis) * 0.15)
        usable_gb = max(0.0, memory_basis - reserve_gb)
        by_memory = max(1, int(usable_gb // estimated_worker_memory_gb))
    return max(1, min(max_workers, by_cpu, by_memory))


def _configured_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    try:
        return max(1, int(value))
    except ValueError:
        return None


def _configured_float(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    try:
        return max(0.1, float(value))
    except ValueError:
        return None


def _memory_gb() -> tuple[float | None, float | None]:
    if os.name == "nt":
        return _windows_memory_gb()
    return _posix_memory_gb()


def _windows_memory_gb() -> tuple[float | None, float | None]:
    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None, None
    return _bytes_to_gb(status.ullTotalPhys), _bytes_to_gb(status.ullAvailPhys)


def _posix_memory_gb() -> tuple[float | None, float | None]:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_pages = os.sysconf("SC_PHYS_PAGES")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None, None
    return _bytes_to_gb(page_size * total_pages), _bytes_to_gb(page_size * available_pages)


def _bytes_to_gb(value: int | float) -> float:
    return round(float(value) / 1024**3, 2)
