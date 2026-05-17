"""
OCR backend adapters.

Supports:
  - PaddleOCR 3.x pipeline API: PaddleOCR(...).predict(...)
  - PaddleOCR 2.x legacy API: PaddleOCR(...).ocr(...)
  - Null OCR backend for deterministic no-OCR runs

Important: PaddleOCR 3.x returns result objects/dicts whose useful data is usually
stored in fields such as rec_texts, rec_scores, rec_polys/dt_polys. The exact
container changed between 3.0/3.1/3.2, therefore this adapter parses recursively
and stores diagnostics for each OCR call.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .image_ops import normalize_for_ocr
from .io_utils import imwrite_unicode
from .pipeline_types import OCRItem


class OCRBackend:
    def recognize(self, image: np.ndarray, zone: Optional[str] = None) -> List[OCRItem]:
        raise NotImplementedError

    def recognize_batch(
        self,
        images: Sequence[np.ndarray],
        zones: Optional[Sequence[Optional[str]]] = None,
        batch_size: int = 16,
    ) -> List[List[OCRItem]]:
        """Recognize a list of crops.

        Backends that do not support real batching inherit this deterministic
        sequential fallback.  The method is intentionally part of the base
        adapter so the high-level pipeline can batch OCR jobs without knowing
        whether PaddleOCR/another backend can execute them in one call.
        """
        out: List[List[OCRItem]] = []
        zone_list = list(zones or [])
        for i, img in enumerate(images):
            zone = zone_list[i] if i < len(zone_list) else None
            out.append(self.recognize(img, zone=zone))
        return out

    def get_debug_info(self) -> Dict[str, Any]:
        return {}


class NullOCR(OCRBackend):
    def __init__(self) -> None:
        self.last_debug: Dict[str, Any] = {"backend": "none"}

    def recognize(self, image: np.ndarray, zone: Optional[str] = None) -> List[OCRItem]:
        self.last_debug = {"backend": "none", "zone": zone, "items": 0}
        return []

    def recognize_batch(
        self,
        images: Sequence[np.ndarray],
        zones: Optional[Sequence[Optional[str]]] = None,
        batch_size: int = 16,
    ) -> List[List[OCRItem]]:
        self.last_debug = {"backend": "none", "batch": True, "batch_size": len(images), "items": 0}
        return [[] for _ in images]

    def get_debug_info(self) -> Dict[str, Any]:
        return dict(self.last_debug)


class PaddleOCRBackend(OCRBackend):
    """PaddleOCR adapter with v2/v3 compatibility and diagnostics."""

    def __init__(
        self,
        lang: str = "ru",
        use_angle_cls: bool = True,
        use_gpu: bool = False,
        rec_model_dir: str = "",
        det_model_dir: str = "",
        cls_model_dir: str = "",
        ocr_version: str = "PP-OCRv5",
        engine: str = "",
        text_detection_model_name: str = "",
        text_recognition_model_name: str = "",
        debug: bool = False,
        input_mode: str = "ndarray",  # ndarray | path
        mkldnn: bool = False,
        pir_api: bool = False,
    ) -> None:
        # Work around PaddlePaddle 3.3.x CPU oneDNN/PIR regression:
        # NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
        # [pir::ArrayAttribute<pir::DoubleAttribute>] in onednn_instruction.cc.
        # These variables must be set before importing paddle/paddleocr.
        # Runtime policy is deliberately expressed with positive booleans:
        #   mkldnn=false disables oneDNN/MKLDNN
        #   pir_api=false disables the PIR API
        self.mkldnn = bool(mkldnn)
        self.pir_api = bool(pir_api)
        if not self.pir_api:
            os.environ.setdefault("FLAGS_enable_pir_api", "0")
        if not self.mkldnn:
            os.environ.setdefault("FLAGS_use_onednn", "0")
            os.environ.setdefault("FLAGS_use_mkldnn", "0")
            os.environ.setdefault("MKLDNN_DISABLE_WORKSPACE", "1")

        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("paddleocr is not installed. Run: pip install paddleocr") from e

        self.use_angle_cls = bool(use_angle_cls)
        self.debug = bool(debug)
        self.input_mode = str(input_mode or "ndarray").lower()
        if self.input_mode not in {"ndarray", "path"}:
            self.input_mode = "ndarray"

        self.init_api = "unknown"
        self.init_kwargs: Dict[str, Any] = {}
        self.last_debug: Dict[str, Any] = {"backend": "paddle", "event": "not_called"}

        # PaddleOCR 3.x pipeline API. Do not pass show_log here: recent versions reject it.
        v3_kwargs: Dict[str, Any] = {
            "lang": lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": bool(use_angle_cls),
            "device": "gpu" if use_gpu else "cpu",
        }
        if ocr_version:
            v3_kwargs["ocr_version"] = ocr_version
        if engine:
            v3_kwargs["engine"] = engine
        if text_detection_model_name:
            v3_kwargs["text_detection_model_name"] = text_detection_model_name
        if text_recognition_model_name:
            v3_kwargs["text_recognition_model_name"] = text_recognition_model_name
        # PaddleOCR/PaddleX 3.x accepts this in many versions; if not,
        # _init_with_pruned_kwargs will remove it safely.
        v3_kwargs["enable_mkldnn"] = bool(self.mkldnn)
        if det_model_dir:
            v3_kwargs["text_detection_model_dir"] = det_model_dir
        if rec_model_dir:
            v3_kwargs["text_recognition_model_dir"] = rec_model_dir
        if cls_model_dir:
            v3_kwargs["textline_orientation_model_dir"] = cls_model_dir

        # PaddleOCR 2.x legacy API.
        v2_kwargs: Dict[str, Any] = {
            "lang": lang,
            "use_angle_cls": bool(use_angle_cls),
            "use_gpu": bool(use_gpu),
            "show_log": False,
            "enable_mkldnn": bool(self.mkldnn),
        }
        if rec_model_dir:
            v2_kwargs["rec_model_dir"] = rec_model_dir
        if det_model_dir:
            v2_kwargs["det_model_dir"] = det_model_dir
        if cls_model_dir:
            v2_kwargs["cls_model_dir"] = cls_model_dir

        errors: List[str] = []
        for api_name, kwargs in (("paddleocr_v3_pipeline", v3_kwargs), ("paddleocr_v2_legacy", v2_kwargs)):
            try:
                self.ocr, final_kwargs = self._init_with_pruned_kwargs(PaddleOCR, kwargs)
                self.init_api = api_name
                self.init_kwargs = final_kwargs
                self.last_debug = {
                    "backend": "paddle",
                    "event": "init_ok",
                    "init_api": self.init_api,
                    "init_kwargs": self.init_kwargs,
                    "input_mode": self.input_mode,
                    "mkldnn": self.mkldnn,
                    "pir_api": self.pir_api,
                }
                return
            except Exception as e:
                errors.append(f"{api_name}: {type(e).__name__}: {e}")

        raise RuntimeError("Failed to initialize PaddleOCR. Attempts:\n" + "\n".join(errors))

    @staticmethod
    def _extract_bad_kwarg(exc: BaseException) -> Optional[str]:
        msg = str(exc)
        patterns = [
            r"Unknown argument:\s*([A-Za-z_][A-Za-z0-9_]*)",
            r"got an unexpected keyword argument ['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]",
            r"__init__\(\) got an unexpected keyword argument ['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]",
        ]
        for pat in patterns:
            m = re.search(pat, msg)
            if m:
                return m.group(1)
        return None

    def _init_with_pruned_kwargs(self, paddle_ocr_cls: Any, kwargs: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        cur = dict(kwargs)
        while True:
            try:
                return paddle_ocr_cls(**cur), cur
            except (TypeError, ValueError) as e:
                bad = self._extract_bad_kwarg(e)
                if bad and bad in cur:
                    cur.pop(bad, None)
                    continue
                msg = str(e)
                if "device" in cur and "device" in msg:
                    cur.pop("device", None)
                    continue
                if "use_gpu" in cur and "use_gpu" in msg:
                    cur.pop("use_gpu", None)
                    continue
                if "show_log" in cur and "show_log" in msg:
                    cur.pop("show_log", None)
                    continue
                raise

    def recognize(self, image: np.ndarray, zone: Optional[str] = None) -> List[OCRItem]:
        self.last_debug = {
            "backend": "paddle",
            "zone": zone,
            "init_api": self.init_api,
            "input_shape": list(image.shape) if hasattr(image, "shape") else None,
            "input_mode": self.input_mode,
            "calls": [],
        }
        if image.size == 0:
            self.last_debug["error"] = "empty_image"
            return []

        norm = normalize_for_ocr(image, clahe=True, sharpen=True)
        if norm.ndim == 2:
            norm = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
        if norm.dtype != np.uint8:
            norm = np.clip(norm, 0, 255).astype(np.uint8)

        # PaddleOCR 3.x preferred API.
        if hasattr(self.ocr, "predict"):
            input_obj: Any = norm
            tmp_path: Optional[Path] = None
            if self.input_mode == "path":
                tmp = tempfile.NamedTemporaryFile(prefix="ptag_ocr_", suffix=".png", delete=False)
                tmp_path = Path(tmp.name)
                tmp.close()
                imwrite_unicode(tmp_path, norm)
                input_obj = str(tmp_path)
            try:
                result = self.ocr.predict(input_obj)
                items = parse_paddle_result(result, zone=zone)
                self.last_debug["calls"].append({
                    "api": "predict",
                    "ok": True,
                    "items": len(items),
                    "raw_summary": summarize_raw_result(result),
                    "tmp_path_used": str(tmp_path) if tmp_path else "",
                })
                if tmp_path is not None:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                if items:
                    return items
            except Exception as e:
                self.last_debug["calls"].append({
                    "api": "predict",
                    "ok": False,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(limit=3) if self.debug else "",
                })
                if tmp_path is not None:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass

        # PaddleOCR 2.x legacy API.
        try:
            result = self.ocr.ocr(norm, cls=self.use_angle_cls)
            items = parse_paddle_result(result, zone=zone)
            self.last_debug["calls"].append({
                "api": "ocr(cls=...)",
                "ok": True,
                "items": len(items),
                "raw_summary": summarize_raw_result(result),
            })
            return items
        except TypeError as e1:
            self.last_debug["calls"].append({"api": "ocr(cls=...)", "ok": False, "error_type": type(e1).__name__, "error": str(e1)})
            try:
                result = self.ocr.ocr(norm)
                items = parse_paddle_result(result, zone=zone)
                self.last_debug["calls"].append({
                    "api": "ocr()",
                    "ok": True,
                    "items": len(items),
                    "raw_summary": summarize_raw_result(result),
                })
                return items
            except Exception as e2:
                self.last_debug["calls"].append({"api": "ocr()", "ok": False, "error_type": type(e2).__name__, "error": str(e2)})
                return []
        except Exception as e:
            self.last_debug["calls"].append({"api": "ocr(cls=...)", "ok": False, "error_type": type(e).__name__, "error": str(e)})
            return []

    def recognize_batch(
        self,
        images: Sequence[np.ndarray],
        zones: Optional[Sequence[Optional[str]]] = None,
        batch_size: int = 16,
    ) -> List[List[OCRItem]]:
        """Batch OCR for many crops.

        PaddleOCR 3.x pipeline builds support list inputs in many releases.  We
        try that fast path first and fall back to the sequential ``recognize``
        path per chunk if the installed version rejects list inputs.
        """
        imgs = list(images or [])
        zone_list = list(zones or [])
        if not imgs:
            self.last_debug = {"backend": "paddle", "batch": True, "event": "empty"}
            return []

        bs = max(1, int(batch_size or 1))
        outputs: List[List[OCRItem]] = [[] for _ in imgs]
        calls: List[Dict[str, Any]] = []
        fast_path_ok = False

        for start in range(0, len(imgs), bs):
            end = min(len(imgs), start + bs)
            chunk = imgs[start:end]
            chunk_zones = [zone_list[i] if i < len(zone_list) else None for i in range(start, end)]

            prepared: List[np.ndarray] = []
            tmp_paths: List[Path] = []
            try:
                for img in chunk:
                    if img is None or getattr(img, "size", 0) == 0:
                        prepared.append(np.zeros((1, 1, 3), dtype=np.uint8))
                        continue
                    norm = normalize_for_ocr(img, clahe=True, sharpen=True)
                    if norm.ndim == 2:
                        norm = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
                    if norm.dtype != np.uint8:
                        norm = np.clip(norm, 0, 255).astype(np.uint8)
                    prepared.append(norm)

                if hasattr(self.ocr, "predict"):
                    input_obj: Any
                    if self.input_mode == "path":
                        input_list: List[str] = []
                        for arr in prepared:
                            tmp = tempfile.NamedTemporaryFile(prefix="ptag_ocr_batch_", suffix=".png", delete=False)
                            tmp_path = Path(tmp.name)
                            tmp.close()
                            imwrite_unicode(tmp_path, arr)
                            tmp_paths.append(tmp_path)
                            input_list.append(str(tmp_path))
                        input_obj = input_list
                    else:
                        input_obj = prepared

                    raw = self.ocr.predict(input_obj)
                    raw_list = list(raw) if isinstance(raw, (list, tuple)) else [raw]
                    if len(raw_list) == len(prepared):
                        for j, sub in enumerate(raw_list):
                            outputs[start + j] = parse_paddle_result(sub, zone=chunk_zones[j])
                        calls.append({
                            "api": "predict(batch)",
                            "ok": True,
                            "start": start,
                            "end": end,
                            "items": [len(outputs[start + j]) for j in range(len(prepared))],
                        })
                        fast_path_ok = True
                        continue
                    if len(prepared) == 1:
                        outputs[start] = parse_paddle_result(raw, zone=chunk_zones[0])
                        calls.append({"api": "predict(single_as_batch)", "ok": True, "start": start, "end": end, "items": [len(outputs[start])]})
                        fast_path_ok = True
                        continue
                    raise RuntimeError(f"unexpected PaddleOCR batch output length: got={len(raw_list)} expected={len(prepared)}")

                raise RuntimeError("backend_has_no_predict_batch_fast_path")
            except Exception as e:
                calls.append({
                    "api": "predict(batch)",
                    "ok": False,
                    "start": start,
                    "end": end,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "fallback": "sequential_recognize",
                })
                for j, img in enumerate(chunk):
                    outputs[start + j] = self.recognize(img, zone=chunk_zones[j])
            finally:
                for tmp_path in tmp_paths:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass

        self.last_debug = {
            "backend": "paddle",
            "batch": True,
            "init_api": self.init_api,
            "input_mode": self.input_mode,
            "requested": len(imgs),
            "batch_size": bs,
            "fast_path_ok": bool(fast_path_ok),
            "calls": calls,
        }
        return outputs

    def get_debug_info(self) -> Dict[str, Any]:
        return dict(self.last_debug)


def parse_paddle_result(result: Any, zone: Optional[str]) -> List[OCRItem]:
    """Parse PaddleOCR 2.x and 3.x outputs into OCRItem list.

    Handles:
      - v2: [[box, (text, score)], ...] or [[[box, (text, score)], ...]]
      - v3: OCRResult objects/dicts with rec_texts, rec_scores, rec_polys/dt_polys/rec_boxes
      - nested containers with res/pruned_result/_res/data fields
    """
    items: List[OCRItem] = []
    _collect_paddle_items(result, zone=zone, out=items, depth=0, max_depth=8)

    # Deduplicate exact duplicates caused by recursive parsing of v3 result containers.
    seen = set()
    deduped: List[OCRItem] = []
    for it in items:
        key = (it.text, round(float(it.conf), 4), json.dumps(it.box, ensure_ascii=False, sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)
    return deduped


def _collect_paddle_items(obj: Any, zone: Optional[str], out: List[OCRItem], depth: int, max_depth: int) -> None:
    if obj is None or depth > max_depth:
        return

    mapping = _to_mapping(obj)
    if mapping is not None:
        _collect_from_mapping(mapping, zone=zone, out=out)
        # Recursively scan nested values. PaddleOCR versions differ: res, _res,
        # pruned_result, json, data, output, etc. Generic scan is safer.
        for k, v in mapping.items():
            if k in {"rec_texts", "texts", "text", "rec_scores", "scores", "confidence", "confidences", "rec_polys", "dt_polys", "boxes", "polys", "textline_boxes", "rec_boxes"}:
                continue
            if _is_recursable(v):
                _collect_paddle_items(v, zone=zone, out=out, depth=depth + 1, max_depth=max_depth)
        return

    # PaddleOCR 2.x list forms and v3 result lists.
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return
        if len(obj) == 1 and isinstance(obj[0], (list, tuple)) and not _looks_like_v2_line(obj[0]):
            _collect_paddle_items(obj[0], zone=zone, out=out, depth=depth + 1, max_depth=max_depth)
            return

        for item in obj:
            if _looks_like_v2_line(item):
                try:
                    box = item[0]
                    text_conf = item[1]
                    text = str(text_conf[0]).strip()
                    conf = float(text_conf[1])
                    if text:
                        out.append(OCRItem(text=text, conf=conf, box=_box_to_list(box), zone=zone))
                except Exception:
                    pass
            else:
                _collect_paddle_items(item, zone=zone, out=out, depth=depth + 1, max_depth=max_depth)
        return


def _collect_from_mapping(mapping: Dict[str, Any], zone: Optional[str], out: List[OCRItem]) -> None:
    # Common v3 structure: {"rec_texts": [...], "rec_scores": [...], "rec_polys": [...]}.
    rec_texts = _first_present(mapping, ["rec_texts", "texts"])
    if isinstance(rec_texts, str):
        rec_texts = [rec_texts]
    if isinstance(rec_texts, np.ndarray):
        rec_texts = rec_texts.tolist()
    if isinstance(rec_texts, (list, tuple)):
        scores = _first_present(mapping, ["rec_scores", "scores", "confidence", "confidences"])
        boxes = _first_present(mapping, ["rec_polys", "dt_polys", "boxes", "polys", "textline_boxes", "rec_boxes"])
        for i, text in enumerate(rec_texts):
            text_s = str(text or "").strip()
            if not text_s:
                continue
            conf = _index_or_scalar_float(scores, i, default=0.0)
            box = _index_or_none(boxes, i)
            out.append(OCRItem(text=text_s, conf=conf, box=_box_to_list(box), zone=zone))

    # Some dict records: {"text": ..., "confidence": ..., "box": ...}.
    text = _first_present(mapping, ["text", "rec_text"])
    if isinstance(text, str) and text.strip():
        conf = _to_float(_first_present(mapping, ["confidence", "rec_score", "score"]), default=0.0)
        box = _first_present(mapping, ["box", "dt_poly", "poly", "bbox"])
        out.append(OCRItem(text=text.strip(), conf=conf, box=_box_to_list(box), zone=zone))


def _looks_like_v2_line(x: Any) -> bool:
    if not isinstance(x, (list, tuple)) or len(x) < 2:
        return False
    text_conf = x[1]
    return isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2 and isinstance(text_conf[0], (str, bytes))


def _to_mapping(obj: Any) -> Optional[Dict[str, Any]]:
    if isinstance(obj, dict):
        return obj

    # PaddleOCR 3.x result objects may have json/to_json as method OR property.
    for attr in ("json", "to_json"):
        val = getattr(obj, attr, None)
        try:
            data = val() if callable(val) else val
            if isinstance(data, str):
                data = json.loads(data)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # Some result objects expose .res or internal ._res dict.
    for attr in ("res", "_res", "pruned_result", "prunedResult", "data", "result"):
        data = getattr(obj, attr, None)
        if isinstance(data, dict):
            return data

    # Some objects implement get/keys without being a dict.
    try:
        keys = obj.keys()  # type: ignore[attr-defined]
        return {k: obj.get(k) for k in keys}  # type: ignore[attr-defined]
    except Exception:
        pass

    if hasattr(obj, "__dict__"):
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict) and d:
            return d
    return None


def _is_recursable(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, (str, bytes, int, float, bool)):
        return False
    if isinstance(v, np.ndarray):
        return v.dtype == object or v.ndim >= 2
    return isinstance(v, (dict, list, tuple)) or hasattr(v, "__dict__") or hasattr(v, "keys")


def _first_present(d: Dict[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if isinstance(x, np.ndarray):
            if x.size == 0:
                return default
            return float(x.reshape(-1)[0])
        return float(x)
    except Exception:
        return default


def _index_or_scalar_float(x: Any, idx: int, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        if isinstance(x, np.ndarray):
            if x.ndim == 0:
                return float(x)
            if len(x) > idx:
                return float(np.asarray(x[idx]).reshape(-1)[0])
        if isinstance(x, (list, tuple)):
            if len(x) > idx:
                return _to_float(x[idx], default=default)
        return _to_float(x, default=default)
    except Exception:
        return default


def _index_or_none(x: Any, idx: int) -> Any:
    if x is None:
        return None
    try:
        if isinstance(x, np.ndarray):
            if x.ndim >= 1 and len(x) > idx:
                return x[idx]
            return x
        if isinstance(x, (list, tuple)) and len(x) > idx:
            return x[idx]
        return x
    except Exception:
        return None


def _box_to_list(box: Any) -> Optional[List[List[float]]]:
    if box is None:
        return None
    try:
        arr = np.asarray(box, dtype=np.float32)
        if arr.size == 0:
            return None
        if arr.ndim == 1:
            if arr.size == 4:
                x1, y1, x2, y2 = arr.tolist()
                return [[float(x1), float(y1)], [float(x2), float(y1)], [float(x2), float(y2)], [float(x1), float(y2)]]
            return None
        arr = arr.reshape(-1, 2)
        return [[float(x), float(y)] for x, y in arr]
    except Exception:
        return None


def summarize_raw_result(result: Any, max_items: int = 4, max_repr: int = 700) -> Dict[str, Any]:
    """Small JSON-safe summary of raw OCR output for diagnostics."""
    summary: Dict[str, Any] = {
        "type": type(result).__name__,
        "repr": safe_repr(result, max_repr),
    }
    if isinstance(result, (list, tuple)):
        summary["len"] = len(result)
        summary["items"] = []
        for item in list(result)[:max_items]:
            m = _to_mapping(item)
            if m is not None:
                summary["items"].append({
                    "type": type(item).__name__,
                    "keys": list(m.keys())[:30],
                    "rec_texts_len": _safe_len(_first_present(m, ["rec_texts", "texts"])),
                    "repr": safe_repr(item, 250),
                })
            else:
                summary["items"].append({"type": type(item).__name__, "repr": safe_repr(item, 250)})
    else:
        m = _to_mapping(result)
        if m is not None:
            summary["keys"] = list(m.keys())[:30]
            summary["rec_texts_len"] = _safe_len(_first_present(m, ["rec_texts", "texts"]))
    return summary


def _safe_len(x: Any) -> Optional[int]:
    try:
        return len(x)  # type: ignore[arg-type]
    except Exception:
        return None


def safe_repr(x: Any, n: int = 700) -> str:
    try:
        s = repr(x)
    except Exception:
        s = f"<{type(x).__name__}>"
    if len(s) > n:
        s = s[:n] + "..."
    return s
