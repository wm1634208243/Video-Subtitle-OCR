from __future__ import annotations

import importlib.util
import importlib
import os
import re
import shutil
import sys
import threading
import types
from dataclasses import asdict, dataclass
from typing import Any

from app.config import CPU_THREADS, DEFAULT_PADDLE_BATCH_SIZE, MODELS_DIR


RAPIDOCR_MIN_SCORE = float(os.getenv("VSO_RAPIDOCR_MIN_SCORE", "0.82"))
_ENGINE_LOCAL = threading.local()


def _set_default_cpu_env() -> None:
    value = str(CPU_THREADS)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, value)
    os.environ.setdefault("PADDLE_NUM_THREADS", value)


_set_default_cpu_env()

import numpy as np


class OcrEngineError(RuntimeError):
    pass


@dataclass(frozen=True)
class EngineStatus:
    id: str
    name: str
    available: bool
    default: bool = False
    description: str = ""
    reason: str | None = None
    install_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseOcrEngine:
    def read_text(self, image: np.ndarray) -> str:
        raise NotImplementedError

    def read_texts(self, images: list[np.ndarray]) -> list[str]:
        return [self.read_text(image) for image in images]


class PaddleEngine(BaseOcrEngine):
    def __init__(self, language: str, batch_size: int = DEFAULT_PADDLE_BATCH_SIZE) -> None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(MODELS_DIR / "paddlex"))
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        try:
            import paddle
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise OcrEngineError("PaddleOCR is not installed. Run: pip install -r requirements.txt") from exc
        try:
            paddle.set_num_threads(CPU_THREADS)
        except Exception:
            pass

        batch_size = max(1, int(batch_size))
        self.ocr = PaddleOCR(
            lang=language,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            textline_orientation_batch_size=batch_size,
            text_recognition_batch_size=batch_size,
        )

    def read_text(self, image: np.ndarray) -> str:
        return self.read_texts([image])[0]

    def read_texts(self, images: list[np.ndarray]) -> list[str]:
        if hasattr(self.ocr, "predict"):
            result = self.ocr.predict(images if len(images) > 1 else images[0])
        else:
            result = self.ocr.ocr(images if len(images) > 1 else images[0], cls=True)

        if len(images) == 1:
            return [" ".join(_extract_paddle_text(result))]

        pages = result if isinstance(result, list) else [result]
        texts = [" ".join(_extract_paddle_text(page)) for page in pages]
        if len(texts) < len(images):
            texts.extend([""] * (len(images) - len(texts)))
        return texts[: len(images)]


class EasyOcrEngine(BaseOcrEngine):
    def __init__(self, language: str) -> None:
        try:
            import easyocr
        except ImportError as exc:
            raise OcrEngineError("EasyOCR is not installed. Run: pip install -r requirements-optional.txt") from exc

        langs = ["ch_sim", "en"] if language in {"ch", "ch_sim", "chi_sim"} else [language]
        self.reader = easyocr.Reader(langs, gpu=False)

    def read_text(self, image: np.ndarray) -> str:
        rows = self.reader.readtext(image, detail=1, paragraph=False)
        return " ".join(str(row[1]) for row in rows if len(row) >= 2)


class RapidOcrEngine(BaseOcrEngine):
    def __init__(self, module_name: str, device_env: str | None = None) -> None:
        if module_name == "rapidocr_openvino":
            _install_openvino_runtime_compat()
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise OcrEngineError(f"{module_name} import failed: {exc}") from exc

        rapid_ocr = getattr(module, "RapidOCR", None)
        if rapid_ocr is None:
            raise OcrEngineError(f"{module_name}.RapidOCR was not found.")

        device_name = (os.getenv(device_env or "") or "").strip() if device_env else ""
        if not device_name and module_name == "rapidocr_openvino":
            device_name = "AUTO"
        if device_name:
            try:
                self.ocr = rapid_ocr(device_name=device_name)
                return
            except TypeError:
                pass
            except Exception:
                pass
            try:
                self.ocr = rapid_ocr(device=device_name)
                return
            except TypeError:
                pass
            except Exception:
                pass
        self.ocr = rapid_ocr()

    def read_text(self, image: np.ndarray) -> str:
        result = self.ocr(image)
        rows = _extract_rapidocr_rows(result)
        texts = [
            text
            for text, score in rows
            if _accept_rapidocr_text(text, score, RAPIDOCR_MIN_SCORE)
        ]
        return " ".join(texts)


class TesseractEngine(BaseOcrEngine):
    def __init__(self, language: str) -> None:
        try:
            import pytesseract
        except ImportError as exc:
            raise OcrEngineError("pytesseract is not installed. Run: pip install -r requirements-optional.txt") from exc
        if shutil.which("tesseract") is None:
            raise OcrEngineError("Tesseract executable was not found. Install Tesseract OCR and add it to PATH.")
        self.pytesseract = pytesseract
        self.language = "chi_sim+eng" if language in {"ch", "ch_sim", "chi_sim"} else language

    def read_text(self, image: np.ndarray) -> str:
        return str(self.pytesseract.image_to_string(image, lang=self.language))


def list_engine_statuses() -> list[dict[str, Any]]:
    engine_ids = ("paddle", "openvino", "onnxruntime", "easyocr", "tesseract")
    return [get_engine_status(engine_id).to_dict() for engine_id in engine_ids]


def get_engine_status(engine_id: str) -> EngineStatus:
    if engine_id == "paddle":
        missing = _missing_modules("paddle", "paddleocr")
        return EngineStatus(
            id="paddle",
            name="PaddleOCR",
            available=not missing,
            default=True,
            description="Recommended CPU engine with the best balance for Chinese/English subtitles.",
            reason=_missing_reason(missing),
            install_hint="Use the web installer or run install.bat and choose Recommended.",
        )
    if engine_id == "easyocr":
        missing = _missing_modules("easyocr")
        return EngineStatus(
            id="easyocr",
            name="EasyOCR",
            available=not missing,
            description="Optional OCR engine. It is easy to try but pulls PyTorch and is usually heavier on CPU.",
            reason=_missing_reason(missing),
            install_hint="Use the web installer or run install.bat and choose EasyOCR.",
        )
    if engine_id == "openvino":
        missing = _missing_modules("rapidocr_openvino", "openvino")
        import_error = _module_import_error("rapidocr_openvino") if not missing else None
        device_hint = _openvino_device_hint() if not missing and import_error is None else ""
        return EngineStatus(
            id="openvino",
            name="OpenVINO OCR",
            available=not missing and import_error is None,
            description=(
                "Fast RapidOCR/OpenVINO backend for Intel acceleration. "
                "AUTO device mode can use GPU/NPU/CPU when available. "
                f"{device_hint}Use PaddleOCR when accuracy matters most."
            ),
            reason=_missing_reason(missing) or import_error,
            install_hint="Use the web installer or run install.bat and choose OpenVINO.",
        )
    if engine_id == "onnxruntime":
        missing = _missing_modules("rapidocr_onnxruntime", "onnxruntime")
        import_error = _module_import_error("rapidocr_onnxruntime") if not missing else None
        return EngineStatus(
            id="onnxruntime",
            name="ONNXRuntime OCR",
            available=not missing and import_error is None,
            description="Fast RapidOCR/ONNXRuntime CPU backend. Use PaddleOCR when accuracy matters most.",
            reason=_missing_reason(missing) or import_error,
            install_hint="Use the web installer or run install.bat and choose ONNXRuntime.",
        )
    if engine_id == "tesseract":
        missing = _missing_modules("pytesseract")
        binary_missing = shutil.which("tesseract") is None
        reason = _missing_reason(missing)
        if binary_missing:
            binary_reason = "Tesseract executable was not found in PATH."
            reason = f"{reason} {binary_reason}".strip() if reason else binary_reason
        return EngineStatus(
            id="tesseract",
            name="Tesseract",
            available=not missing and not binary_missing,
            description="Optional lightweight wrapper around the system Tesseract OCR executable.",
            reason=reason,
            install_hint="Use the web installer or run install.bat and choose Tesseract.",
        )
    raise OcrEngineError(f"Unknown OCR engine: {engine_id}")


def get_engine(name: str, language: str, batch_size: int = DEFAULT_PADDLE_BATCH_SIZE) -> BaseOcrEngine:
    cache = getattr(_ENGINE_LOCAL, "cache", None)
    if cache is None:
        cache = {}
        _ENGINE_LOCAL.cache = cache
    key = (name, language, int(batch_size))
    if key in cache:
        return cache[key]

    status = get_engine_status(name)
    if not status.available:
        hint = f" {status.install_hint}" if status.install_hint else ""
        raise OcrEngineError(f"{status.name} is not available. {status.reason or ''}{hint}".strip())

    if name == "paddle":
        engine = PaddleEngine(language, batch_size)
    elif name == "openvino":
        engine = RapidOcrEngine("rapidocr_openvino", "VSO_OPENVINO_DEVICE")
    elif name == "onnxruntime":
        engine = RapidOcrEngine("rapidocr_onnxruntime")
    elif name == "easyocr":
        engine = EasyOcrEngine(language)
    elif name == "tesseract":
        engine = TesseractEngine(language)
    else:
        raise OcrEngineError(f"Unknown OCR engine: {name}")
    cache[key] = engine
    return engine


def _missing_modules(*module_names: str) -> list[str]:
    return [module_name for module_name in module_names if importlib.util.find_spec(module_name) is None]


def _missing_reason(missing_modules: list[str]) -> str | None:
    if not missing_modules:
        return None
    return "Missing Python module(s): " + ", ".join(missing_modules) + "."


def _module_import_error(module_name: str) -> str | None:
    try:
        if module_name == "rapidocr_openvino":
            _install_openvino_runtime_compat()
        importlib.import_module(module_name)
    except Exception as exc:
        return f"{module_name} import failed: {exc}"
    return None


def _openvino_device_hint() -> str:
    try:
        _install_openvino_runtime_compat()
        import openvino

        devices = list(getattr(openvino.Core(), "available_devices", []) or [])
    except Exception:
        return ""
    if not devices:
        return ""
    return "Detected devices: " + ", ".join(devices) + ". "


def _install_openvino_runtime_compat() -> None:
    if "openvino.runtime" in sys.modules:
        return
    try:
        import openvino
    except Exception:
        return
    core = getattr(openvino, "Core", None)
    if core is None:
        return
    runtime = types.ModuleType("openvino.runtime")
    runtime.Core = core
    sys.modules["openvino.runtime"] = runtime


def _extract_paddle_text(result: Any) -> list[str]:
    texts: list[str] = []
    if not result:
        return texts

    if isinstance(result, dict):
        rec_texts = result.get("rec_texts")
        if isinstance(rec_texts, list):
            return [str(text) for text in rec_texts]

    pages = result if isinstance(result, list) else [result]
    for page in pages:
        if page is None:
            continue
        if hasattr(page, "json"):
            try:
                data = page.json
                if callable(data):
                    data = data()
                texts.extend(_extract_paddle_text(data))
                continue
            except Exception:
                pass
        if hasattr(page, "res"):
            texts.extend(_extract_paddle_text(getattr(page, "res")))
            continue
        if isinstance(page, dict):
            rec_texts = page.get("rec_texts")
            if isinstance(rec_texts, list):
                texts.extend(str(text) for text in rec_texts)
            data = page.get("res")
            if data is not None:
                texts.extend(_extract_paddle_text(data))
            continue
        for item in page:
            try:
                text = item[1][0]
            except (TypeError, IndexError):
                continue
            texts.append(str(text))
    return texts


def _extract_rapidocr_text(result: Any) -> list[str]:
    return [text for text, _ in _extract_rapidocr_rows(result)]


def _extract_rapidocr_rows(result: Any) -> list[tuple[str, float | None]]:
    if result is None:
        return []

    payload = result[0] if isinstance(result, tuple) and result else result
    if payload is None:
        return []

    if isinstance(payload, dict):
        for key in ("txts", "texts", "rec_texts"):
            values = payload.get(key)
            if isinstance(values, list):
                return [(str(text), None) for text in values if str(text).strip()]
        payload = payload.get("result") or payload.get("res") or []

    if isinstance(payload, (list, tuple)) and any(isinstance(value, str) for value in payload):
        payload = [payload]

    texts: list[tuple[str, float | None]] = []
    for row in payload:
        if isinstance(row, dict):
            text = row.get("text") or row.get("rec_text") or row.get("txt")
            if text:
                score = row.get("score") or row.get("confidence") or row.get("rec_score")
                texts.append((str(text), _safe_float(score)))
            continue
        if isinstance(row, (list, tuple)):
            text_score: float | None = None
            for value in row:
                if isinstance(value, (float, int, np.floating)):
                    text_score = float(value)
            for value in row:
                if isinstance(value, str) and value.strip():
                    texts.append((value, text_score))
                    break
    return texts


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _accept_rapidocr_text(text: str, score: float | None, min_score: float = RAPIDOCR_MIN_SCORE) -> bool:
    value = text.strip()
    if not value:
        return False
    if score is not None and score < min_score:
        return False

    letters = [char for char in value if char.isalpha()]
    if len(value) <= 1:
        return False
    if len(value) <= 3 and not any(char in value for char in ".?!,:;"):
        return False

    compact = re.sub(r"[^A-Za-z0-9]", "", value)
    if 2 <= len(compact) <= 10 and compact.upper() == compact and any(char.isdigit() for char in compact):
        return False
    if 2 <= len(compact) <= 5 and compact.upper() == compact and not any(char in value for char in ".?!,:;"):
        return False

    tokens = re.findall(r"[A-Za-z]+", value)
    if len(tokens) == 1 and len(tokens[0]) <= 6 and not any(char in value for char in ".?!,:;"):
        return False
    if letters:
        ascii_letters = sum("a" <= char.lower() <= "z" for char in letters)
        cjk_chars = sum("\u4e00" <= char <= "\u9fff" for char in value)
        if cjk_chars and ascii_letters >= 2:
            return False

    return True
