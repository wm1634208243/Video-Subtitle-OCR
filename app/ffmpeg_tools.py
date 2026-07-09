from __future__ import annotations

import shutil


def ffmpeg_exe() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("ffmpeg was not found. Install ffmpeg or run: pip install imageio-ffmpeg") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()
