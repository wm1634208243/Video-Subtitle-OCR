from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def write_debug_preview(
    frame_paths: list[Path],
    crop_bottom: float,
    output_path: Path,
    max_frames: int = 6,
    crop_region: dict[str, float] | None = None,
) -> None:
    selected = _sample_paths(frame_paths, max_frames)
    panels = []
    for frame_path in selected:
        image = cv2.imread(str(frame_path))
        if image is not None:
            panels.append(_make_panel(image, frame_path.name, crop_bottom, crop_region))
    if not panels:
        return

    width = max(panel.shape[1] for panel in panels)
    sheet = np.vstack([_pad_to_width(panel, width) for panel in panels])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def append_ocr_log(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sample_paths(frame_paths: list[Path], max_frames: int) -> list[Path]:
    if len(frame_paths) <= max_frames:
        return frame_paths
    indexes = [round(index * (len(frame_paths) - 1) / (max_frames - 1)) for index in range(max_frames)]
    return [frame_paths[index] for index in indexes]


def _make_panel(image, name: str, crop_bottom: float, crop_region: dict[str, float] | None = None):
    target_width = 360
    scale = target_width / image.shape[1]
    target_height = max(1, round(image.shape[0] * scale))
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    if crop_region:
        x1 = int(target_width * max(0.0, min(crop_region.get("x", 0.0), 1.0)))
        y1 = int(target_height * max(0.0, min(crop_region.get("y", 0.0), 1.0)))
        x2 = int(target_width * max(0.0, min(crop_region.get("x", 0.0) + crop_region.get("w", 1.0), 1.0)))
        y2 = int(target_height * max(0.0, min(crop_region.get("y", 0.0) + crop_region.get("h", 1.0), 1.0)))
        if x2 > x1 and y2 > y1:
            cv2.rectangle(resized, (x1, y1), (x2 - 1, y2 - 1), (80, 170, 255), 2)
    else:
        line_y = int(target_height * (1.0 - crop_bottom))
        line_y = min(max(line_y, 0), target_height - 1)
        cv2.line(resized, (0, line_y), (target_width - 1, line_y), (0, 180, 255), 2)
        cv2.rectangle(resized, (0, line_y), (target_width - 1, target_height - 1), (0, 180, 255), 1)

    label = np.full((26, target_width, 3), 246, dtype=np.uint8)
    cv2.putText(label, name, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 45, 55), 1, cv2.LINE_AA)
    return np.vstack([label, resized])


def _pad_to_width(image, width: int):
    if image.shape[1] == width:
        return image
    return cv2.copyMakeBorder(image, 0, 0, 0, width - image.shape[1], cv2.BORDER_CONSTANT, value=(246, 246, 246))
