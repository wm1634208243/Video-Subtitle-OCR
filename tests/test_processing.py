import numpy as np
import cv2

from app.models import JobOptions
from app.ocr_engines import _accept_rapidocr_text, _extract_rapidocr_text, _install_openvino_runtime_compat
from app.processing import (
    _clean_ocr_text,
    _crop_image,
    _estimate_stable_subtitle_region,
    _flush_ocr_batch,
    _focus_subtitle_strip,
    _is_hard_noise_text,
    _looks_like_english_subtitles,
    _prefer_review_text,
    _prepare_image,
    _retry_empty_ocr,
    _should_review_text,
    _signature_diff,
    _subtitle_signature,
    PendingOcrFrame,
)


def test_crop_image_uses_manual_region() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    options = JobOptions(crop_x=0.25, crop_y=0.2, crop_w=0.5, crop_h=0.4)

    cropped = _crop_image(image, options)

    assert cropped.shape[:2] == (40, 100)


def test_crop_image_uses_auto_region_when_manual_region_is_absent() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    options = JobOptions(auto_crop_x=0.0, auto_crop_y=0.6, auto_crop_w=1.0, auto_crop_h=0.25)

    cropped = _crop_image(image, options)

    assert cropped.shape[:2] == (25, 200)


def test_manual_region_takes_priority_over_auto_region() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    options = JobOptions(
        crop_x=0.25,
        crop_y=0.2,
        crop_w=0.5,
        crop_h=0.4,
        auto_crop_x=0.0,
        auto_crop_y=0.6,
        auto_crop_w=1.0,
        auto_crop_h=0.25,
    )

    cropped = _crop_image(image, options)

    assert cropped.shape[:2] == (40, 100)


def test_crop_image_falls_back_to_bottom_area() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    options = JobOptions(crop_bottom=0.55)

    cropped = _crop_image(image, options)

    assert cropped.shape[:2] == (55, 200)


def test_estimate_stable_subtitle_region_prefers_repeated_subtitle_band(tmp_path) -> None:
    frame_paths = []
    for index in range(8):
        image = np.full((240, 420, 3), 46, dtype=np.uint8)
        cv2.putText(image, "LOGO", (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA)
        cv2.rectangle(image, (0, 62), (420, 126), (62 + index * 8, 70, 84), -1)
        cv2.putText(
            image,
            f"stable subtitle {index % 2}",
            (72, 184),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        frame_path = tmp_path / f"frame_{index:06d}.jpg"
        cv2.imwrite(str(frame_path), image)
        frame_paths.append(frame_path)

    region = _estimate_stable_subtitle_region(frame_paths, JobOptions())

    assert region is not None
    assert 0.58 <= region["y"] <= 0.76
    assert 0.12 <= region["h"] <= 0.3
    assert region["x"] == 0.0
    assert region["w"] == 1.0


def test_estimate_stable_subtitle_region_shifts_letterboxed_video_to_lower_band(tmp_path) -> None:
    frame_paths = []
    for index in range(8):
        image = np.zeros((640, 360, 3), dtype=np.uint8)
        cv2.rectangle(image, (0, 218), (360, 422), (40 + index * 4, 48, 58), -1)
        cv2.line(image, (0, 220), (360, 220), (120, 120, 120), 2)
        cv2.line(image, (0, 420), (360, 420), (120, 120, 120), 2)
        cv2.circle(image, (120 + index * 8, 300), 48, (110, 116, 124), -1)
        cv2.putText(
            image,
            "SAVE FUEL",
            (62, 398),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.05,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        frame_path = tmp_path / f"frame_{index:06d}.jpg"
        cv2.imwrite(str(frame_path), image)
        frame_paths.append(frame_path)

    region = _estimate_stable_subtitle_region(frame_paths, JobOptions(crop_bottom=0.5))

    assert region is not None
    assert region["y"] >= 0.45
    assert region["h"] <= 0.18


def test_estimate_stable_subtitle_region_skips_manual_region(tmp_path) -> None:
    image = np.full((160, 300, 3), 30, dtype=np.uint8)
    cv2.putText(image, "subtitle", (48, 116), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    frame_path = tmp_path / "frame_000001.jpg"
    cv2.imwrite(str(frame_path), image)

    region = _estimate_stable_subtitle_region(
        [frame_path],
        JobOptions(crop_x=0.1, crop_y=0.2, crop_w=0.8, crop_h=0.3),
    )

    assert region is None


def test_focus_subtitle_strip_keeps_subtitle_band() -> None:
    image = np.zeros((240, 360, 3), dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (360, 240), (28, 34, 42), -1)
    cv2.circle(image, (80, 48), 32, (92, 92, 92), -1)
    cv2.putText(image, "same subtitle", (48, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    focused = _focus_subtitle_strip(image)

    assert focused.shape[0] < image.shape[0]
    assert focused.shape[1] == image.shape[1]
    assert focused.shape[0] >= 42


def test_prepare_image_keeps_manual_region_height() -> None:
    image = np.zeros((240, 360, 3), dtype=np.uint8)
    options = JobOptions(crop_x=0.1, crop_y=0.25, crop_w=0.8, crop_h=0.5, auto_subtitle_strip=True)

    prepared = _prepare_image(image, options)

    assert prepared.shape[0] == 240


def test_subtitle_signature_detects_changed_region() -> None:
    options = JobOptions(crop_bottom=1.0)
    first = np.zeros((120, 300, 3), dtype=np.uint8)
    second = first.copy()
    third = first.copy()
    cv2.putText(first, "hello", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(second, "hello", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(third, "world", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)

    first_signature = _subtitle_signature(first, options)
    second_signature = _subtitle_signature(second, options)
    third_signature = _subtitle_signature(third, options)

    assert _signature_diff(first_signature, second_signature) == 0
    assert (_signature_diff(first_signature, third_signature) or 0) > options.frame_diff_threshold


def test_job_options_enable_unchanged_frame_skip_by_default() -> None:
    options = JobOptions()

    assert options.skip_unchanged_frames is True
    assert options.frame_diff_threshold == 0.2


def test_subtitle_signature_ignores_soft_background_motion() -> None:
    options = JobOptions(crop_bottom=1.0)
    first = np.full((160, 360, 3), 80, dtype=np.uint8)
    second = np.full((160, 360, 3), 110, dtype=np.uint8)
    for x in range(0, 360, 24):
        cv2.line(first, (x, 0), (x + 80, 160), (95, 95, 95), 2)
        cv2.line(second, (x + 8, 0), (x + 88, 160), (125, 125, 125), 2)
    cv2.putText(first, "same subtitle", (42, 102), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(second, "same subtitle", (42, 102), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 255, 255), 2, cv2.LINE_AA)

    diff = _signature_diff(_subtitle_signature(first, options), _subtitle_signature(second, options))

    assert diff is not None
    assert diff <= options.frame_diff_threshold


class NeverCalledEngine:
    def read_texts(self, images):
        raise AssertionError("balanced mode should not retry empty OCR frames")


def test_balanced_mode_does_not_retry_empty_frames() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    texts = _retry_empty_ocr(
        NeverCalledEngine(),
        [""],
        [image],
        JobOptions(mode="balanced", crop_bottom=0.5),
    )

    assert texts == [""]


def test_detects_mostly_english_subtitles() -> None:
    assert _looks_like_english_subtitles(
        [
            "Impossible:",
            "the word that defines the limits.",
            "For MG, it is where our journey begins.",
            "The thrill of a sports car",
        ]
    )


def test_english_detection_rejects_chinese_subtitles() -> None:
    assert not _looks_like_english_subtitles(
        [
            "这是第一句字幕",
            "这是第二句字幕",
            "MG 让驾驶更有乐趣",
            "继续向前",
        ]
    )


def test_extract_rapidocr_text_from_common_result_shapes() -> None:
    assert _extract_rapidocr_text(([[[0, 0], [1, 0]], "hello", 0.99],)) == ["hello"]
    assert _extract_rapidocr_text(([[None, "hello", 0.99], [None, "world", 0.98]], 0.12)) == ["hello", "world"]
    assert _extract_rapidocr_text({"txts": ["hello", "world"]}) == ["hello", "world"]


def test_rapidocr_text_filter_removes_common_video_noise() -> None:
    assert not _accept_rapidocr_text("LA22WNP", 0.98)
    assert not _accept_rapidocr_text("MIOICOUAIIIOOOLOUI", 0.59)
    assert not _accept_rapidocr_text("DHI", 0.95)
    assert not _accept_rapidocr_text("Cynua", 0.93)
    assert _accept_rapidocr_text("Impossible:", 0.99)
    assert _accept_rapidocr_text("Fun.", 0.95)
    assert _accept_rapidocr_text("through the tightest corners", 0.99)


def test_review_heuristic_catches_unspaced_english() -> None:
    options = JobOptions(engine="openvino", review_engine="paddle")

    assert _should_review_text("thewordthatdefinesthelimits.", options)
    assert _should_review_text("SAVEFUEL,ORKEEPTHEPUN", options)
    assert _should_review_text("ForMG, it'swhereourjourneybegins.", options)
    assert _should_review_text("tomake theaspiration attainable,", options)
    assert _should_review_text("and bring the impossiblewithinreach.", options)
    assert _prefer_review_text(
        "thewordthatdefinesthelimits.",
        "the word that defines the limits.",
        options,
    )


def test_review_heuristic_rejects_plate_like_noise() -> None:
    options = JobOptions(engine="openvino", review_engine="paddle")

    assert _is_hard_noise_text("LA22 WNP")
    assert _is_hard_noise_text("IV D MG MD MD MD MG A 0 A")
    assert _is_hard_noise_text("0060960")
    assert _is_hard_noise_text("MGRanDAI")
    assert _is_hard_noise_text("TODAUT")
    assert not _prefer_review_text("", "LA22 WNP", options)
    assert _clean_ocr_text("No less a dream 4") == "No less a dream"
    assert _clean_ocr_text("IV D MG MD MD MD MG A 0 A") == ""
    assert _clean_ocr_text("MG Plug-in Hybrid+ takes the choosing out ofit.") == (
        "MG Plug-in Hybrid+ takes the choosing out of it."
    )
    assert _clean_ocr_text("Wnether in the traric,") == "Whether in the traffic,"
    assert _clean_ocr_text("Orthe highways.") == "Or the highways."
    assert _clean_ocr_text("SAVE FUEL, OR KEEPTHE PUNC") == "SAVE FUEL, OR KEEP THE PUNCH"
    assert _clean_ocr_text("even at low SoC and low tem! t-rures.") == "even at low SoC and low temperatures."
    assert _clean_ocr_text("MG Plug-in Hybridr provides a balanced combination") == (
        "MG Plug-in Hybrid+ provides a balanced combination"
    )
    assert _clean_ocr_text("it significantly elevates overall power output..") == (
        "it significantly elevates overall power output."
    )
    assert _clean_ocr_text("G Plug-in Hybrid+ provides a balanced combination") == (
        "MG Plug-in Hybrid+ provides a balanced combination"
    )
    assert _clean_ocr_text("MG Plug In Hybrid+ jirovides a balanced combination") == (
        "MG Plug-in Hybrid+ provides a balanced combination"
    )
    assert _clean_ocr_text("Eof power performance und fuel consumption.") == (
        "of power performance and fuel consumption."
    )
    assert _clean_ocr_text("Across all speedst") == "Across all speeds"
    assert _clean_ocr_text("Moreover, the semi-solid electrolyte further reduce") == (
        "Moreover, the semi-solid electrolyte further reduces"
    )
    assert _clean_ocr_text("TECHFORMORE") == "TECH FOR MORE"
    assert _clean_ocr_text("battery technol Wst tiime") == ""
    assert _clean_ocr_text("ah .dmwmy scenarios and all climate conditions.") == (
        "scenarios and all climate conditions."
    )
    assert _clean_ocr_text("ForMG, it'swhereourjourneybegins.") == (
        "For MG, it's where our journey begins."
    )
    assert _clean_ocr_text("into effortlessconfidence") == "into effortless confidence"
    assert _clean_ocr_text("Behind every evoletion,") == "Behind every evolution,"
    assert _clean_ocr_text("THISISOURMISSION") == "THIS IS OUR MISSION"
    assert _clean_ocr_text("tomake theaspiration attainable,") == "to make the aspiration attainable,"
    assert _clean_ocr_text("and bring the impossiblewithinreach.") == "and bring the impossible within reach."
    assert _clean_ocr_text("Findyourhore") == "Find your more"


class StaticEngine:
    def __init__(self, texts) -> None:
        self.texts = texts

    def read_texts(self, images):
        return self.texts[: len(images)]


def test_flush_ocr_batch_uses_paddle_review_for_low_confidence_text(tmp_path) -> None:
    image = np.zeros((80, 200, 3), dtype=np.uint8)
    pending = [PendingOcrFrame(timestamp=1.0, image=image, original=image)]
    samples = []

    result = _flush_ocr_batch(
        StaticEngine(["thewordthatdefinesthelimits."]),
        StaticEngine(["the word that defines the limits."]),
        pending,
        JobOptions(engine="openvino", review_engine="paddle"),
        tmp_path / "ocr.jsonl",
        samples,
    )

    assert result == (1, 0, 1, "the word that defines the limits.")
    assert samples == [(1.0, "the word that defines the limits.")]
    assert '"reviewed": true' in (tmp_path / "ocr.jsonl").read_text(encoding="utf-8")


def test_flush_ocr_batch_drops_plate_like_review_noise(tmp_path) -> None:
    image = np.zeros((80, 200, 3), dtype=np.uint8)
    pending = [PendingOcrFrame(timestamp=1.0, image=image, original=image)]
    samples = []

    result = _flush_ocr_batch(
        StaticEngine(["LA22WNP"]),
        StaticEngine(["LA22 WNP"]),
        pending,
        JobOptions(engine="openvino", review_engine="paddle"),
        tmp_path / "ocr.jsonl",
        samples,
    )

    assert result == (1, 0, 0, "")
    assert samples == [(1.0, "")]


def test_openvino_runtime_compatibility_shim() -> None:
    _install_openvino_runtime_compat()

    try:
        from openvino.runtime import Core
    except Exception:
        return

    assert Core is not None
