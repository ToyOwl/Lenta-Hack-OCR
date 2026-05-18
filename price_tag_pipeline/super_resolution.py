# -*- coding: utf-8 -*-
"""Super-resolution router for OCR zones.

Design notes for the price-tag OCR pipeline:

- OpenCV SR is the default fast path.  It is deterministic, cheap and safe for
  whole-tag crops and all text zones.
- PaddleOCR Text SR is a quality/recovery path.  PaddleOCR Telescope/TBSRN and
  Gestalt/TSRN are text-line models, so they must be applied only to OCR zones
  such as product_name, main_price, kopeeks, discount and old_price.  They are
  not suitable as a generic full-tag or QR/barcode upscaler.
- Paddle 3.x PIR inference models exported as inference.json are currently not
  reliable in some Windows builds: input/output shadow names may collide.  For a
  robust baseline this module supports a dynamic PaddleOCR checkpoint backend
  (backend: paddle_sr_dynamic) that builds the PaddleOCR model from its config,
  forces Transform.infer_mode=True and loads best_accuracy.pdparams directly.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from dataclasses import asdict, dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np

SCRIPT_VERSION = "2026-05-18.sr-router-opencv-paddle-dynamic-v1"


@dataclass
class SRVariant:
    name: str
    image: np.ndarray
    meta: Dict[str, Any]


@dataclass
class SRMeta:
    backend: str
    name: str
    input_shape: List[int]
    output_shape: List[int]
    status: str = "ok"
    scale: float = 1.0
    notes: List[str] | None = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["notes"] = list(self.notes or [])
        return d


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        return image
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _safe_shape(image: Any) -> List[int]:
    try:
        return [int(x) for x in image.shape]
    except Exception:
        return []


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _is_abs_or_drive_path(s: str) -> bool:
    if not s:
        return False
    p = Path(s)
    return p.is_absolute() or (len(s) > 2 and s[1:3] in {":\\", ":/"})


def _resolve_path(value: Any, *, base_dir: Path | None = None) -> Path:
    raw = str(value or "").strip().strip('"')
    if not raw:
        return Path("")
    p = Path(raw).expanduser()
    if p.is_absolute() or _is_abs_or_drive_path(raw):
        return p
    if base_dir is not None:
        return (base_dir / p).resolve()
    return p.resolve()


def _project_root_from_path(path: Path | None) -> Path:
    candidates: List[Path] = []
    if path:
        p = path.resolve()
        candidates.extend([p, *p.parents])
    # Prefer project root inferred from this package.  This makes relative YAML
    # paths stable even when the current working directory is not the project
    # root, for example when PyCharm runs a script with a custom cwd.
    try:
        this_file = Path(__file__).resolve()
        candidates.extend([this_file.parent.parent, *this_file.parents])
    except Exception:
        pass
    candidates.append(Path.cwd().resolve())
    seen = set()
    for c in candidates:
        try:
            c = c.resolve()
        except Exception:
            pass
        if c in seen:
            continue
        seen.add(c)
        if (c / "price_tag_pipeline").exists() or (c / "run_price_tag_pipeline.py").exists() or (c / "run_detected_tracks_dataset.py").exists():
            return c
        if c.name == "paddle_sr" and c.parent.name == "models":
            return c.parent.parent
        if c.name == "sr_telescope" and c.parent.name == "paddle_sr":
            return c.parent.parent.parent
        if c.name == "sr_gestalt" and c.parent.name == "paddle_sr":
            return c.parent.parent.parent
    return Path.cwd().resolve()


def opencv_super_resolve(
    image: np.ndarray,
    *,
    name: str = "opencv_lanczos",
    method: str = "lanczos",
    scale: float = 2.0,
    min_side: int = 0,
    max_side: int = 1400,
    sharpen: bool = True,
    sharpen_amount: float = 0.10,
    dnn_model_path: str = "",
    dnn_model_name: str = "fsrcnn",
    dnn_model_scale: int = 2,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """OpenCV-only SR/upscale.

    Supports:
    - cv2.resize: lanczos, bicubic, linear, area, nearest;
    - cv2.dnn_superres: FSRCNN/ESPCN/EDSR/LapSRN when opencv-contrib and a
      model file are available.
    """
    if image is None or image.size == 0:
        return image, {"backend": "opencv", "name": name, "status": "empty"}

    img = ensure_bgr(image)
    h, w = img.shape[:2]
    notes: List[str] = []
    requested_scale = max(1.0, float(scale or 1.0))
    target_scale = requested_scale

    if int(min_side or 0) > 0 and min(h, w) < int(min_side):
        target_scale = max(target_scale, min(4.0, float(int(min_side)) / max(1.0, float(min(h, w)))))
    if int(max_side or 0) > 0 and max(h, w) * target_scale > int(max_side):
        target_scale = max(1.0, float(int(max_side)) / max(1.0, float(max(h, w))))
        notes.append("scale_limited_by_max_side")
    if target_scale <= 1.001:
        return img, SRMeta("opencv", name, _safe_shape(img), _safe_shape(img), status="identity", scale=1.0, notes=notes).to_dict()

    method_l = str(method or "lanczos").lower().strip()
    out: np.ndarray | None = None
    used_backend = "opencv_resize"
    used_name = name

    if method_l in {"dnn", "dnn_superres", "opencv_dnn_superres"} and dnn_model_path:
        try:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()  # type: ignore[attr-defined]
            sr.readModel(str(Path(dnn_model_path)))
            model_scale = int(dnn_model_scale or round(target_scale))
            sr.setModel(str(dnn_model_name or "fsrcnn").lower(), max(2, model_scale))
            out = sr.upsample(img)
            used_backend = "opencv_dnn_superres"
            used_name = f"{dnn_model_name}_{model_scale}x"
            if out is not None and out.size:
                target_scale = float(out.shape[1]) / max(1.0, float(w))
        except Exception as e:
            notes.append(f"dnn_superres_failed:{type(e).__name__}:{str(e)[:180]}")
            out = None

    if out is None:
        interpolation = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "bilinear": cv2.INTER_LINEAR,
            "area": cv2.INTER_AREA,
            "cubic": cv2.INTER_CUBIC,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
            "lanczos4": cv2.INTER_LANCZOS4,
        }.get(method_l, cv2.INTER_LANCZOS4)
        new_w = max(1, int(round(w * target_scale)))
        new_h = max(1, int(round(h * target_scale)))
        out = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        used_name = f"opencv_{method_l if method_l else 'lanczos'}"

    if sharpen and float(sharpen_amount) > 0:
        try:
            blur = cv2.GaussianBlur(out, (0, 0), 0.85)
            a = float(sharpen_amount)
            out = cv2.addWeighted(out, 1.0 + a, blur, -a, 0)
        except Exception:
            notes.append("sharpen_failed")

    return out, SRMeta(used_backend, used_name, _safe_shape(img), _safe_shape(out), scale=float(target_scale), notes=notes).to_dict()


# -----------------------------------------------------------------------------
# Paddle inference backend for exported .pdmodel/.json models.
# Kept for completeness; dynamic backend is currently more reliable for Paddle 3.x.
# -----------------------------------------------------------------------------


class PaddleSRBackend:
    """Lazy Paddle inference wrapper for exported PaddleOCR SR models."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = dict(cfg or {})
        self.name = str(self.cfg.get("name") or self.cfg.get("backend") or "paddle_sr")
        self.model_dir = str(self.cfg.get("model_dir") or "")
        self.model_file = str(self.cfg.get("model_file") or "")
        self.params_file = str(self.cfg.get("params_file") or "")
        self.use_gpu = bool(self.cfg.get("use_gpu", False))
        self.gpu_mem_mb = int(self.cfg.get("gpu_mem_mb", 256))
        self.gpu_device_id = int(self.cfg.get("gpu_device_id", 0))
        self.mkldnn = bool(self.cfg.get("mkldnn", False))
        self.cpu_threads = int(self.cfg.get("cpu_threads", 2))
        self.ir_optim = bool(self.cfg.get("ir_optim", False))
        self.memory_optim = bool(self.cfg.get("memory_optim", False))
        self.fixed_input_shape = _parse_image_shape(self.cfg.get("image_shape", self.cfg.get("fixed_input_shape", "3,32,128")))
        self.network_downscale = int(self.cfg.get("network_downscale", 2))
        self.norm = str(self.cfg.get("norm", "minus_one_one") or "minus_one_one")
        self.output_channel_order = str(self.cfg.get("output_channel_order", "rgb") or "rgb").lower()
        self.input_channel_order = str(self.cfg.get("input_channel_order", "rgb") or "rgb").lower()
        self.resize_back = bool(self.cfg.get("resize_back", False))
        self.output_scale_hint = float(self.cfg.get("output_scale_hint", 2.0))
        self._predictor: Any = None
        self._input_name: str | None = None
        self._init_error: str = ""

    def super_resolve(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        if image is None or image.size == 0:
            return image, {"backend": "paddle_sr", "name": self.name, "status": "empty"}
        img = ensure_bgr(image)
        input_shape = _safe_shape(img)
        predictor = self._get_predictor()
        if predictor is None:
            return img, {
                "backend": "paddle_sr",
                "name": self.name,
                "status": "unavailable",
                "input_shape": input_shape,
                "output_shape": input_shape,
                "error": self._init_error,
            }

        try:
            x = self._preprocess(img)
            input_handle = predictor.get_input_handle(self._input_name)
            try:
                input_handle.reshape(x.shape)
            except Exception:
                pass
            input_handle.copy_from_cpu(x)
            t0 = _now_ms()
            predictor.run()
            latency = _now_ms() - t0
            output_names = list(predictor.get_output_names())
            outputs: Dict[str, np.ndarray] = {}
            for name in output_names:
                try:
                    outputs[name] = predictor.get_output_handle(name).copy_to_cpu()
                except Exception:
                    pass
            if not outputs:
                raise RuntimeError("no readable paddle inference outputs")
            chosen_name, output = _choose_sr_output(outputs, input_hw=(x.shape[-2], x.shape[-1]))
            out = self._postprocess(output, original_shape=img.shape)
            return out, {
                "backend": "paddle_sr",
                "name": self.name,
                "status": "ok",
                "input_shape": input_shape,
                "network_input_shape": list(x.shape),
                "chosen_output_name": chosen_name,
                "raw_output_shape": list(output.shape),
                "output_shape": _safe_shape(out),
                "model_dir": self.model_dir,
                "scale_hint": self.output_scale_hint,
                "latency_ms": latency,
            }
        except Exception as e:
            return img, {
                "backend": "paddle_sr",
                "name": self.name,
                "status": "predict_failed",
                "input_shape": input_shape,
                "output_shape": input_shape,
                "error": f"{type(e).__name__}: {str(e)[:320]}",
            }

    def _get_predictor(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        model_file, params_file = self._resolve_model_files()
        if not model_file or not params_file:
            self._init_error = "model_file_or_params_file_not_found"
            return None
        try:
            import paddle.inference as paddle_infer  # type: ignore

            cfg = paddle_infer.Config(str(model_file), str(params_file))
            try:
                cfg.disable_glog_info()
            except Exception:
                pass
            if self.use_gpu:
                cfg.enable_use_gpu(int(self.gpu_mem_mb), int(self.gpu_device_id))
            else:
                cfg.disable_gpu()
                if self.cpu_threads > 0:
                    cfg.set_cpu_math_library_num_threads(int(self.cpu_threads))
                if self.mkldnn:
                    cfg.enable_mkldnn()
            try:
                cfg.switch_ir_optim(bool(self.ir_optim))
            except Exception:
                pass
            for pass_name in ("memory_optimize_pass", "memory_optimization_pass"):
                try:
                    if hasattr(cfg, "delete_pass"):
                        cfg.delete_pass(pass_name)
                except Exception:
                    pass
            if self.memory_optim:
                try:
                    cfg.enable_memory_optim()
                except Exception:
                    pass
            predictor = paddle_infer.create_predictor(cfg)
            input_names = list(predictor.get_input_names())
            last_error: Exception | None = None
            for name in input_names:
                try:
                    _ = predictor.get_input_handle(name)
                    self._input_name = name
                    self._predictor = predictor
                    return self._predictor
                except Exception as exc:
                    last_error = exc
            raise RuntimeError(f"cannot obtain input handle; inputs={input_names}; outputs={list(predictor.get_output_names())}; last_error={last_error}")
        except Exception as e:
            self._init_error = f"{type(e).__name__}: {str(e)[:320]}"
            return None

    def _resolve_model_files(self) -> Tuple[str, str]:
        if self.model_file and self.params_file:
            return self.model_file, self.params_file
        if not self.model_dir:
            return "", ""
        p = Path(self.model_dir).expanduser()
        if not p.exists():
            return "", ""
        params = p / "inference.pdiparams"
        for model_name in ("inference.pdmodel", "inference.json"):
            model = p / model_name
            if model.exists() and params.exists():
                return str(model), str(params)
        models = sorted(list(p.glob("*.pdmodel")) + list(p.glob("*.json")))
        param_files = sorted(p.glob("*.pdiparams"))
        if models and param_files:
            return str(models[0]), str(param_files[0])
        return "", ""

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        c, hr_h, hr_w = self.fixed_input_shape
        in_h = max(1, int(round(hr_h / max(1, self.network_downscale))))
        in_w = max(1, int(round(hr_w / max(1, self.network_downscale))))
        return _bgr_to_network_input(
            image,
            image_shape=(c, in_h, in_w),
            norm=self.norm,
            input_channel_order=self.input_channel_order,
        )

    def _postprocess(self, output: np.ndarray, *, original_shape: Tuple[int, ...]) -> np.ndarray:
        y = _tensor_to_bgr_image(output, output_channel_order=self.output_channel_order)
        if self.resize_back:
            h, w = original_shape[:2]
            y = cv2.resize(y, (int(round(w * self.output_scale_hint)), int(round(h * self.output_scale_hint))), interpolation=cv2.INTER_CUBIC)
        return y


# -----------------------------------------------------------------------------
# PaddleOCR dynamic checkpoint backend.
# -----------------------------------------------------------------------------


class PaddleDynamicSRBackend:
    """Dynamic PaddleOCR SR backend based on PaddleOCR repo + checkpoint.

    This is the recommended backend for current PaddleOCR SR baseline when
    exported Paddle 3.x PIR inference models are problematic.
    """

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = dict(cfg or {})
        self.name = str(self.cfg.get("name") or "paddle_sr_dynamic")
        self.model_key = _infer_model_key(self.cfg)
        raw_model_dir = self.cfg.get("model_dir", "")
        self.model_dir = _resolve_path(raw_model_dir) if raw_model_dir else Path("")
        self.project_root = _project_root_from_path(self.model_dir if str(self.model_dir) else None)
        self.repo_dir = _resolve_path(self.cfg.get("repo_dir") or (self.project_root / "third_party" / "PaddleOCR"), base_dir=self.project_root)
        self.config_path = _resolve_path(self.cfg.get("config") or self.cfg.get("config_path") or _default_sr_config(self.repo_dir, self.model_key), base_dir=self.project_root)
        self.checkpoint_prefix = _resolve_path(self.cfg.get("checkpoint_prefix") or _default_checkpoint_prefix(self.project_root, self.model_key), base_dir=self.project_root)
        self.use_gpu = bool(self.cfg.get("use_gpu", False))
        self.fixed_input_shape = _parse_image_shape(self.cfg.get("image_shape", "3,32,128"))
        self.network_downscale = int(self.cfg.get("network_downscale", 2))
        self.norm = str(self.cfg.get("norm", "minus_one_one") or "minus_one_one")
        self.input_channel_order = str(self.cfg.get("input_channel_order", "rgb") or "rgb").lower()
        self.output_channel_order = str(self.cfg.get("output_channel_order", "rgb") or "rgb").lower()
        self.resize_back = bool(self.cfg.get("resize_back", False))
        self.output_scale_hint = float(self.cfg.get("output_scale_hint", 2.0))
        self.strict_state = bool(self.cfg.get("strict_state", False))
        self.force_infer_mode = bool(self.cfg.get("force_infer_mode", True))
        self._model: Any = None
        self._paddle: Any = None
        self._init_error: str = ""
        self._load_stats: Dict[str, Any] = {}

    def super_resolve(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        if image is None or image.size == 0:
            return image, {"backend": "paddle_sr_dynamic", "name": self.name, "status": "empty"}
        img = ensure_bgr(image)
        input_shape = _safe_shape(img)
        if not self._ensure_model():
            return img, {
                "backend": "paddle_sr_dynamic",
                "name": self.name,
                "status": "unavailable",
                "input_shape": input_shape,
                "output_shape": input_shape,
                "error": self._init_error,
                "repo_dir": str(self.repo_dir),
                "config": str(self.config_path),
                "checkpoint_prefix": str(self.checkpoint_prefix),
            }
        try:
            c, hr_h, hr_w = self.fixed_input_shape
            in_h = max(1, int(round(hr_h / max(1, self.network_downscale))))
            in_w = max(1, int(round(hr_w / max(1, self.network_downscale))))
            x_np = _bgr_to_network_input(
                img,
                image_shape=(c, in_h, in_w),
                norm=self.norm,
                input_channel_order=self.input_channel_order,
            )
            paddle = self._paddle
            x = paddle.to_tensor(x_np)
            t0 = _now_ms()
            with paddle.no_grad():
                raw_out = self._model(x)
            latency = _now_ms() - t0
            outputs = _flatten_tensor_outputs(raw_out)
            if not outputs:
                raise RuntimeError(f"model returned no tensor-like output: {type(raw_out)}")
            arrays = {f"out_{i}": (item.numpy() if hasattr(item, "numpy") else np.asarray(item)) for i, item in enumerate(outputs)}
            chosen_name, chosen_arr = _choose_sr_output(arrays, input_hw=(x_np.shape[-2], x_np.shape[-1]))
            out = _tensor_to_bgr_image(chosen_arr, output_channel_order=self.output_channel_order)
            if self.resize_back:
                h, w = img.shape[:2]
                out = cv2.resize(out, (int(round(w * self.output_scale_hint)), int(round(h * self.output_scale_hint))), interpolation=cv2.INTER_CUBIC)
            meta = dict(self._load_stats)
            meta.update(
                {
                    "backend": "paddle_sr_dynamic",
                    "name": self.name,
                    "status": "ok",
                    "input_shape": input_shape,
                    "network_input_shape": list(x_np.shape),
                    "sr_image_shape_hr": list(self.fixed_input_shape),
                    "chosen_output_name": chosen_name,
                    "raw_output_shape": list(chosen_arr.shape),
                    "output_shape": _safe_shape(out),
                    "latency_ms": latency,
                    "model_key": self.model_key,
                    "repo_dir": str(self.repo_dir),
                    "config": str(self.config_path),
                    "checkpoint_prefix": str(self.checkpoint_prefix),
                }
            )
            return out, meta
        except Exception as e:
            return img, {
                "backend": "paddle_sr_dynamic",
                "name": self.name,
                "status": "predict_failed",
                "input_shape": input_shape,
                "output_shape": input_shape,
                "error": f"{type(e).__name__}: {str(e)[:420]}",
            }

    def _ensure_model(self) -> bool:
        if self._model is not None and self._paddle is not None:
            return True
        try:
            import yaml  # type: ignore
            import paddle  # type: ignore

            if not self.repo_dir.exists():
                raise FileNotFoundError(f"PaddleOCR repo not found: {self.repo_dir}")
            if not self.config_path.exists():
                raise FileNotFoundError(f"PaddleOCR SR config not found: {self.config_path}")
            params_path = self.checkpoint_prefix.with_suffix(".pdparams")
            if not params_path.exists():
                raise FileNotFoundError(f"PaddleOCR SR checkpoint not found: {params_path}")

            repo_str = str(self.repo_dir)
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)

            try:
                paddle.set_device("gpu" if self.use_gpu else "cpu")
            except Exception:
                paddle.set_device("cpu")

            with self.config_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            arch = cfg.get("Architecture")
            if not isinstance(arch, Mapping):
                raise RuntimeError(f"No Architecture section in config: {self.config_path}")

            transform_cfg = arch.get("Transform") if isinstance(arch, dict) else None
            forced_infer_mode = False
            if self.force_infer_mode and isinstance(transform_cfg, dict):
                t_name = str(transform_cfg.get("name", "")).upper()
                if t_name in {"TBSRN", "TSRN"}:
                    transform_cfg["infer_mode"] = True
                    forced_infer_mode = True

            build_model = importlib.import_module("ppocr.modeling.architectures").build_model
            model = build_model(arch)
            raw_state = paddle.load(str(params_path))
            if isinstance(raw_state, dict) and "state_dict" in raw_state and isinstance(raw_state["state_dict"], dict):
                raw_state = raw_state["state_dict"]
            if isinstance(raw_state, dict) and "model" in raw_state and isinstance(raw_state["model"], dict):
                raw_state = raw_state["model"]
            if not isinstance(raw_state, dict):
                raise RuntimeError(f"Unsupported checkpoint object type: {type(raw_state)}")

            model_state = model.state_dict()
            filtered: Dict[str, Any] = {}
            skipped_shape: List[Any] = []
            skipped_name: List[str] = []
            for k, v in raw_state.items():
                kk = str(k)
                if kk not in model_state:
                    if kk.startswith("module.") and kk[7:] in model_state:
                        kk = kk[7:]
                    elif kk.startswith("model.") and kk[6:] in model_state:
                        kk = kk[6:]
                    else:
                        skipped_name.append(str(k))
                        continue
                try:
                    if list(v.shape) != list(model_state[kk].shape):
                        skipped_shape.append((str(k), list(v.shape), list(model_state[kk].shape)))
                        continue
                except Exception:
                    pass
                filtered[kk] = v

            if self.strict_state and len(filtered) != len(model_state):
                raise RuntimeError(f"strict_state=True but loaded {len(filtered)}/{len(model_state)} keys")

            merged = dict(model_state)
            merged.update(filtered)
            model.set_state_dict(merged)
            model.eval()
            self._model = model
            self._paddle = paddle
            self._load_stats = {
                "loaded_keys": len(filtered),
                "model_keys": len(model_state),
                "checkpoint_keys": len(raw_state),
                "skipped_name_count": len(skipped_name),
                "skipped_shape_count": len(skipped_shape),
                "skipped_name_preview": skipped_name[:20],
                "skipped_shape_preview": skipped_shape[:10],
                "forced_infer_mode": forced_infer_mode,
            }
            return True
        except Exception as e:
            self._init_error = f"{type(e).__name__}: {str(e)[:640]}"
            return False


# -----------------------------------------------------------------------------
# SR router
# -----------------------------------------------------------------------------


class SuperResolutionPipeline:
    """Configurable SR router for OCR crops."""

    def __init__(self, cfg: Mapping[str, Any] | None = None) -> None:
        self.cfg = dict(cfg or {})
        self.enabled = bool(self.cfg.get("enabled", False))
        self.profile = str(self.cfg.get("profile", "light") or "light").lower().strip()
        self.strategy = str(self.cfg.get("strategy", "replace_if_small") or "replace_if_small").lower().strip()
        self.apply_to = list(self.cfg.get("apply_to", ["main_price*", "*price*", "*kopeeks*", "*discount*", "product_name", "scale_number", "promo_header", "footer_note", "card_price_small", "no_card_price_small"]) or [])
        self.skip_for = list(self.cfg.get("skip_for", ["full_tag", "qr_code", "linear_barcode", "barcode", "datamatrix"]) or [])
        self.save_crops = bool(self.cfg.get("save_crops", False))
        self.include_raw_when_appending = bool(self.cfg.get("include_raw_when_appending", True))
        self.max_variants_per_crop = max(1, int(self.cfg.get("max_variants_per_crop", 3)))
        self.keep_failed_records = bool(self.cfg.get("keep_failed_records", False))
        self._paddle_backends: Dict[str, PaddleSRBackend] = {}
        self._paddle_dynamic_backends: Dict[str, PaddleDynamicSRBackend] = {}

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> "SuperResolutionPipeline":
        return cls(cfg or {})

    def is_enabled(self) -> bool:
        return bool(self.enabled)

    def describe(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "profile": self.profile,
            "strategy": self.strategy,
            "apply_to": list(self.apply_to),
            "skip_for": list(self.skip_for),
            "script_version": SCRIPT_VERSION,
        }

    def build_variants(self, image: np.ndarray, *, zone_class: str = "") -> List[SRVariant]:
        raw_meta = {"backend": "none", "name": "raw", "status": "raw", "input_shape": _safe_shape(image), "output_shape": _safe_shape(image)}
        if image is None or image.size == 0:
            return [SRVariant("raw", image, raw_meta)]
        if not self.enabled or not self._zone_allowed(zone_class) or not self._trigger(image, zone_class):
            return [SRVariant("raw", image, raw_meta)]

        backend_cfgs = self._profile_backends()
        produced: List[SRVariant] = []
        failed: List[SRVariant] = []
        for bcfg in backend_cfgs:
            if len(produced) >= self.max_variants_per_crop:
                break
            if not self._backend_zone_allowed(bcfg, zone_class):
                continue
            variant = self._run_backend(image, bcfg)
            if variant is None:
                continue
            status = str(variant.meta.get("status") or "")
            if status not in {"ok", "identity"}:
                failed.append(variant)
                continue
            if variant.image is None or variant.image.size == 0:
                continue
            produced.append(variant)

        if not produced:
            if self.keep_failed_records and self.strategy in {"append", "append_variants", "raw_plus_sr", "multi"}:
                return [SRVariant("raw", image, raw_meta)] + failed[: self.max_variants_per_crop]
            return [SRVariant("raw", image, raw_meta)]

        if self.strategy in {"append", "append_variants", "raw_plus_sr", "multi"}:
            out = [SRVariant("raw", image, raw_meta)] if self.include_raw_when_appending else []
            out.extend(produced[: self.max_variants_per_crop])
            return out

        return [produced[0]]

    def _profile_backends(self) -> List[Mapping[str, Any]]:
        profiles = self.cfg.get("profiles", {}) if isinstance(self.cfg.get("profiles"), Mapping) else {}
        pcfg = profiles.get(self.profile, {}) if isinstance(profiles.get(self.profile, {}), Mapping) else {}
        backends = pcfg.get("backends") if isinstance(pcfg, Mapping) else None
        if not backends:
            backends = self.cfg.get("backends")
        if isinstance(backends, Sequence) and not isinstance(backends, (str, bytes)):
            return [b for b in backends if isinstance(b, Mapping) and bool(b.get("enabled", True))]
        return [{"name": "opencv_lanczos_x2", "backend": "opencv_resize", "method": "lanczos", "scale": 2.0, "max_side": 1200}]

    def _run_backend(self, image: np.ndarray, bcfg: Mapping[str, Any]) -> SRVariant | None:
        backend = str(bcfg.get("backend", bcfg.get("type", "opencv_resize")) or "opencv_resize").lower().strip()
        name = str(bcfg.get("name") or backend)
        if backend in {"opencv", "opencv_resize", "resize", "lanczos", "bicubic", "opencv_dnn_superres", "dnn_superres"}:
            method = str(bcfg.get("method") or ("dnn_superres" if backend in {"opencv_dnn_superres", "dnn_superres"} else backend))
            out, meta = opencv_super_resolve(
                image,
                name=name,
                method=method,
                scale=float(bcfg.get("scale", 2.0)),
                min_side=int(bcfg.get("min_side", self.cfg.get("min_side", 0) or 0)),
                max_side=int(bcfg.get("max_side", self.cfg.get("max_side", 1400) or 1400)),
                sharpen=bool(bcfg.get("sharpen", True)),
                sharpen_amount=float(bcfg.get("sharpen_amount", self.cfg.get("sharpen_amount", 0.10) or 0.10)),
                dnn_model_path=str(bcfg.get("model_path", "") or ""),
                dnn_model_name=str(bcfg.get("model_name", "fsrcnn") or "fsrcnn"),
                dnn_model_scale=int(bcfg.get("model_scale", bcfg.get("scale", 2))),
            )
            meta["zone_sr_backend_name"] = name
            return SRVariant(name, out, meta)

        if backend in {"paddle", "paddle_sr", "paddleocr_sr", "tsrn", "tbsrn", "gestalt", "telescope"}:
            key = name + ":" + str(bcfg.get("model_dir") or bcfg.get("model_file") or "")
            if key not in self._paddle_backends:
                self._paddle_backends[key] = PaddleSRBackend({**dict(bcfg), "name": name})
            out, meta = self._paddle_backends[key].super_resolve(image)
            status = str(meta.get("status") or "")
            # Paddle 3.x PIR inference.json may fail on Windows because of
            # shadow input/output names.  If requested, transparently retry via
            # the dynamic checkpoint backend.  This keeps old YAML files usable
            # while the recommended config should still use backend=paddle_sr_dynamic.
            fallback_enabled = bool(bcfg.get("fallback_to_dynamic", self.cfg.get("fallback_paddle_to_dynamic", False)))
            if fallback_enabled and status not in {"ok", "identity"}:
                dyn_cfg = {**dict(bcfg), "backend": "paddle_sr_dynamic", "name": name + "_dynamic_fb"}
                dkey = ":".join([
                    str(dyn_cfg.get("name") or name),
                    str(dyn_cfg.get("checkpoint_prefix") or ""),
                    str(dyn_cfg.get("config") or dyn_cfg.get("config_path") or ""),
                    str(dyn_cfg.get("repo_dir") or ""),
                    str(dyn_cfg.get("model_dir") or ""),
                ])
                if dkey not in self._paddle_dynamic_backends:
                    self._paddle_dynamic_backends[dkey] = PaddleDynamicSRBackend(dyn_cfg)
                dout, dmeta = self._paddle_dynamic_backends[dkey].super_resolve(image)
                dmeta["fallback_from"] = meta
                return SRVariant(str(dyn_cfg.get("name")), dout, dmeta)
            return SRVariant(name, out, meta)

        if backend in {"paddle_sr_dynamic", "paddleocr_sr_dynamic", "paddle_dynamic", "paddle_checkpoint"}:
            key = ":".join([
                name,
                str(bcfg.get("checkpoint_prefix") or ""),
                str(bcfg.get("config") or bcfg.get("config_path") or ""),
                str(bcfg.get("repo_dir") or ""),
            ])
            if key not in self._paddle_dynamic_backends:
                self._paddle_dynamic_backends[key] = PaddleDynamicSRBackend({**dict(bcfg), "name": name})
            out, meta = self._paddle_dynamic_backends[key].super_resolve(image)
            return SRVariant(name, out, meta)
        return None

    def _zone_allowed(self, zone_class: str) -> bool:
        z = str(zone_class or "").strip()
        if _matches_any(z, self.skip_for):
            return False
        if not self.apply_to:
            return True
        return _matches_any(z, self.apply_to)

    def _backend_zone_allowed(self, bcfg: Mapping[str, Any], zone_class: str) -> bool:
        allow = bcfg.get("apply_to")
        skip = bcfg.get("skip_for")
        z = str(zone_class or "")
        if isinstance(skip, Sequence) and not isinstance(skip, (str, bytes)) and _matches_any(z, skip):
            return False
        if isinstance(allow, Sequence) and not isinstance(allow, (str, bytes)):
            return _matches_any(z, allow)
        return True

    def _trigger(self, image: np.ndarray, zone_class: str) -> bool:
        trigger = self.cfg.get("trigger", {}) if isinstance(self.cfg.get("trigger"), Mapping) else {}
        if bool(trigger.get("run_always", False)):
            return True
        force_classes = list(trigger.get("always_for_classes", []) or [])
        if force_classes and _matches_any(zone_class, force_classes):
            return True
        h, w = image.shape[:2]
        min_side_lt = int(trigger.get("run_if_min_side_lt", trigger.get("min_side_lt", 0)) or 0)
        height_lt = int(trigger.get("run_if_height_lt", trigger.get("height_lt", 0)) or 0)
        width_lt = int(trigger.get("run_if_width_lt", trigger.get("width_lt", 0)) or 0)
        area_lt = int(trigger.get("run_if_area_lt", trigger.get("area_lt", 0)) or 0)
        aspect_gt = float(trigger.get("run_if_aspect_gt", trigger.get("aspect_gt", 0.0)) or 0.0)
        aspect_lt = float(trigger.get("run_if_aspect_lt", trigger.get("aspect_lt", 0.0)) or 0.0)
        aspect = float(w) / max(1.0, float(h))
        if min_side_lt > 0 and min(h, w) < min_side_lt:
            return True
        if height_lt > 0 and h < height_lt:
            return True
        if width_lt > 0 and w < width_lt:
            return True
        if area_lt > 0 and h * w < area_lt:
            return True
        if aspect_gt > 0 and aspect > aspect_gt:
            return True
        if aspect_lt > 0 and aspect < aspect_lt:
            return True
        if not trigger:
            return True
        return False


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def _matches_any(value: str, patterns: Iterable[Any]) -> bool:
    v = str(value or "")
    for pat in patterns or []:
        p = str(pat or "")
        if not p:
            continue
        if fnmatch(v, p) or v == p:
            return True
    return False


def _parse_image_shape(value: Any) -> Tuple[int, int, int]:
    if isinstance(value, str):
        parts = [int(float(x.strip())) for x in value.replace("x", ",").split(",") if x.strip()]
    elif isinstance(value, Sequence):
        parts = [int(x) for x in value]
    else:
        parts = [3, 32, 128]
    if len(parts) == 3:
        return int(parts[0]), int(parts[1]), int(parts[2])
    if len(parts) == 2:
        return 3, int(parts[0]), int(parts[1])
    return 3, 32, 128


def _bgr_to_network_input(
    image: np.ndarray,
    *,
    image_shape: Tuple[int, int, int],
    norm: str = "minus_one_one",
    input_channel_order: str = "rgb",
) -> np.ndarray:
    c, h, w = image_shape
    img = ensure_bgr(image)
    if input_channel_order.lower() == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_CUBIC)
    arr = img.astype(np.float32)
    if c == 1:
        arr = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if input_channel_order.lower() == "rgb" else cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None]
    chw = np.transpose(arr, (2, 0, 1))
    if str(norm).lower() in {"minus_one_one", "-1_1", "m11"}:
        chw = chw / 127.5 - 1.0
    elif str(norm).lower() in {"zero_one", "0_1", "01"}:
        chw = chw / 255.0
    else:
        chw = chw / 255.0
    return chw[None, ...].astype(np.float32)


def _tensor_to_bgr_image(tensor: Any, *, output_channel_order: str = "rgb") -> np.ndarray:
    arr = np.asarray(tensor)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in {1, 3}:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise RuntimeError(f"unsupported output tensor shape: {list(np.asarray(tensor).shape)}")
    arr = arr.astype(np.float32)
    mn = float(np.nanmin(arr)) if arr.size else 0.0
    mx = float(np.nanmax(arr)) if arr.size else 1.0
    if mn >= -1.1 and mx <= 1.1 and mn < -0.05:
        arr = (arr + 1.0) * 127.5
    elif mn >= -0.05 and mx <= 1.1:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[2] == 1:
        return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
    if output_channel_order.lower() == "rgb":
        return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
    return arr[:, :, :3]


def _flatten_tensor_outputs(obj: Any) -> List[Any]:
    outs: List[Any] = []
    if obj is None:
        return outs
    if hasattr(obj, "numpy"):
        return [obj]
    if isinstance(obj, np.ndarray):
        return [obj]
    if isinstance(obj, Mapping):
        for v in obj.values():
            outs.extend(_flatten_tensor_outputs(v))
        return outs
    if isinstance(obj, (list, tuple)):
        for v in obj:
            outs.extend(_flatten_tensor_outputs(v))
        return outs
    return outs


def _choose_sr_output(outputs: Mapping[str, np.ndarray], *, input_hw: Tuple[int, int]) -> Tuple[str, np.ndarray]:
    candidates: List[Tuple[int, str, np.ndarray]] = []
    for name, arr in outputs.items():
        a = np.asarray(arr)
        score = int(a.size)
        if a.ndim == 4:
            score += 1_000_000
            hw = (int(a.shape[-2]), int(a.shape[-1]))
            if hw != tuple(input_hw):
                score += 100_000
            score += int(a.shape[-2]) * int(a.shape[-1])
        candidates.append((score, str(name), a))
    if not candidates:
        raise RuntimeError("no output candidates")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def _infer_model_key(cfg: Mapping[str, Any]) -> str:
    raw = " ".join(str(cfg.get(k, "")) for k in ("name", "model_key", "model_dir", "config", "config_path", "checkpoint_prefix"))
    raw_l = raw.lower()
    if "gestalt" in raw_l or "tsrn" in raw_l or "strock" in raw_l:
        return "gestalt"
    return "telescope"


def _default_sr_config(repo_dir: Path, model_key: str) -> Path:
    if model_key == "gestalt":
        return repo_dir / "configs" / "sr" / "sr_tsrn_transformer_strock.yml"
    return repo_dir / "configs" / "sr" / "sr_telescope.yml"


def _default_checkpoint_prefix(project_root: Path, model_key: str) -> Path:
    if model_key == "gestalt":
        return project_root / "models" / "paddle_sr" / "_work" / "extracted" / "gestalt" / "sr_tsrn_transformer_strock_train" / "best_accuracy"
    return project_root / "models" / "paddle_sr" / "_work" / "extracted" / "telescope" / "sr_telescope_train" / "best_accuracy"
