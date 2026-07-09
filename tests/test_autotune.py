from pathlib import Path

from app.autotune import build_auto_options


def test_balanced_portrait_video_uses_tighter_crop(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.autotune.probe_video",
        lambda _: {"duration": 60.0, "width": 368.0, "height": 640.0},
    )
    monkeypatch.setattr("app.autotune.estimate_subtitle_crop", lambda _: None)

    options = build_auto_options(Path("portrait.mp4"), "balanced", "paddle", "en")

    assert options.crop_bottom == 0.5
    assert options.fps == 1.0
    assert options.frame_diff_threshold == 0.3


def test_portrait_detected_crop_is_capped(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.autotune.probe_video",
        lambda _: {"duration": 60.0, "width": 368.0, "height": 640.0},
    )
    monkeypatch.setattr("app.autotune.estimate_subtitle_crop", lambda _: 0.65)

    options = build_auto_options(Path("portrait.mp4"), "balanced", "paddle", "en")

    assert options.crop_bottom == 0.58


def test_landscape_small_video_keeps_wider_crop(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.autotune.probe_video",
        lambda _: {"duration": 60.0, "width": 854.0, "height": 480.0},
    )
    monkeypatch.setattr("app.autotune.estimate_subtitle_crop", lambda _: None)

    options = build_auto_options(Path("landscape.mp4"), "balanced", "paddle", "en")

    assert options.crop_bottom == 0.65
