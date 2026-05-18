# -*- coding: utf-8 -*-
"""SR + correction utilities for blurred shelf-label crops.

The functions are intentionally OpenCV-only by default.  They are used as a
safe fallback before a model-based SR path is introduced for RKNN/ONNX.

Main goals:
- run NLM/fusion at a higher working resolution without hallucinating digits;
- correct red/orange promo tags and white header separately;
- generate QR-friendly variants for small, blurred or perspective-distorted QR.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np

from .super_resolution import opencv_super_resolve


@dataclass
class EnhancementMeta:
    enabled: bool
    method: str
    input_shape: List[int]
    output_shape: List[int]
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
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def super_resolve_crop(
    image: np.ndarray,
    *,
    enabled: bool = True,
    scale: float = 2.0,
    min_side: int = 320,
    max_side: int = 1400,
    method: str = "lanczos",
    model_path: str = "",
    model_name: str = "espcn",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Backward-compatible OpenCV SR wrapper.

    The project now has a dedicated ``super_resolution`` module.  QR/fusion
    paths still call this function, so keep the old signature and route it to
    the OpenCV backend only.  PaddleOCR SR is intentionally not used for QR.
    """
    if image is None or image.size == 0:
        return image, {"enabled": bool(enabled), "status": "empty"}
    if not enabled:
        return ensure_bgr(image), {"enabled": False, "status": "disabled"}
    out, meta = opencv_super_resolve(
        image,
        name=str(method or "opencv_sr"),
        method=str(method or "lanczos"),
        scale=float(scale),
        min_side=int(min_side),
        max_side=int(max_side),
        sharpen=True,
        dnn_model_path=str(model_path or ""),
        dnn_model_name=str(model_name or "espcn"),
        dnn_model_scale=int(round(float(scale or 2.0))),
    )
    meta["enabled"] = True
    return out, meta


def enhance_fused_price_tag(
    image: np.ndarray,
    *,
    enabled: bool = True,
    sr_enabled: bool = False,
    sr_scale: float = 1.0,
    sr_min_side: int = 320,
    sr_max_side: int = 1400,
    sr_method: str = "lanczos",
    glare_suppression: bool = True,
    clahe_clip: float = 2.4,
    text_gain: float = 0.42,
    red_zone_gain: float = 0.16,
    profile: str = "safe",
    gray_world_strength: float = 0.35,
    glare_max_area_ratio: float = 0.045,
    blackhat_kernel_ratio: float = 0.020,
    final_unsharp_amount: float = 0.10,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Correct a fused tag crop for OCR and QR fallback.

    The correction is conservative: gray-world balance, glare inpainting, LAB
    CLAHE and dark-text boost.  Red/orange promo areas receive a small local
    contrast gain, but the function avoids aggressive saturation shifts.
    """
    if image is None or image.size == 0:
        return image, {"enabled": bool(enabled), "status": "empty"}
    if not enabled:
        return image, {"enabled": False, "status": "disabled"}

    img = ensure_bgr(image).copy()
    input_shape = list(img.shape)
    notes: List[str] = []
    sr_meta: Dict[str, Any] = {}
    if sr_enabled:
        img, sr_meta = super_resolve_crop(
            img,
            enabled=True,
            scale=float(sr_scale),
            min_side=int(sr_min_side),
            max_side=int(sr_max_side),
            method=str(sr_method or "lanczos"),
        )

    profile_l = str(profile or "safe").lower().strip()
    if profile_l in {"safe", "mild", "ocr_safe"}:
        clahe_clip = min(float(clahe_clip), 1.85)
        text_gain = min(float(text_gain), 0.22)
        red_zone_gain = min(float(red_zone_gain), 0.07)
        gray_world_strength = min(float(gray_world_strength), 0.35)
        final_unsharp_amount = min(float(final_unsharp_amount), 0.12)
    elif profile_l in {"qr", "qr_only"}:
        clahe_clip = min(float(clahe_clip), 2.1)
        text_gain = min(float(text_gain), 0.16)
        red_zone_gain = 0.0
        gray_world_strength = min(float(gray_world_strength), 0.25)

    try:
        img = _gray_world_balance(img, strength=float(gray_world_strength))
    except Exception:
        notes.append("gray_world_failed")

    if glare_suppression:
        try:
            img, glare_info = _suppress_small_glare(img, max_area_ratio=float(glare_max_area_ratio))
            if glare_info.get("skipped"):
                notes.append(str(glare_info.get("reason") or "glare_skipped"))
        except Exception:
            notes.append("glare_suppression_failed")

    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=max(1.0, float(clahe_clip)), tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
    except Exception:
        notes.append("clahe_failed")

    try:
        img = _boost_dark_text(img, text_gain=float(text_gain), red_zone_gain=float(red_zone_gain), kernel_ratio=float(blackhat_kernel_ratio))
    except Exception:
        notes.append("text_boost_failed")

    try:
        amount = max(0.0, min(0.35, float(final_unsharp_amount)))
        if amount > 1e-6:
            blur = cv2.GaussianBlur(img, (0, 0), 0.9)
            img = cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)
    except Exception:
        pass

    meta = EnhancementMeta(True, f"price_tag_{profile_l}_grayworld_glare_clahe_masked_blackhat", input_shape, list(img.shape), 1.0, notes).to_dict()
    meta["profile"] = profile_l
    meta["gray_world_strength"] = float(gray_world_strength)
    meta["glare_max_area_ratio"] = float(glare_max_area_ratio)
    meta["blackhat_kernel_ratio"] = float(blackhat_kernel_ratio)
    meta["final_unsharp_amount"] = float(final_unsharp_amount)
    if sr_meta:
        meta["sr"] = sr_meta
    return img, meta


def make_qr_preprocess_variants(
    image: np.ndarray,
    *,
    sr_enabled: bool = True,
    sr_scale: float = 2.0,
    sr_min_side: int = 420,
    sr_max_side: int = 1400,
    sr_method: str = "lanczos",
    morphology: bool = True,
) -> List[Tuple[str, np.ndarray, float]]:
    """Return QR-focused variants and their coordinate scale.

    The scale is relative to the input image supplied to this function.
    """
    variants: List[Tuple[str, np.ndarray, float]] = []
    if image is None or image.size == 0:
        return variants
    base = ensure_bgr(image)
    h, w = base.shape[:2]
    variants.append(("qr_raw", base, 1.0))

    sr_scale_used = 1.0
    sr_img = base
    if sr_enabled:
        sr_img, sr_meta = super_resolve_crop(
            base,
            enabled=True,
            scale=float(sr_scale),
            min_side=int(sr_min_side),
            max_side=int(sr_max_side),
            method=str(sr_method or "lanczos"),
        )
        sr_scale_used = float(sr_meta.get("scale") or 1.0)
        if sr_img is not base or sr_scale_used > 1.001:
            variants.append(("qr_sr", sr_img, sr_scale_used))

    gray = cv2.cvtColor(sr_img, cv2.COLOR_BGR2GRAY) if sr_img.ndim == 3 else sr_img.copy()
    try:
        gray = cv2.fastNlMeansDenoising(gray, None, 7, 7, 21)
    except Exception:
        pass
    try:
        clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(6, 6)).apply(gray)
    except Exception:
        clahe = gray
    sharp = cv2.addWeighted(clahe, 1.65, cv2.GaussianBlur(clahe, (0, 0), 0.9), -0.65, 0)
    variants.append(("qr_sr_clahe_sharp", sharp, sr_scale_used))

    try:
        _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("qr_otsu", otsu, sr_scale_used))
        variants.append(("qr_otsu_inv", 255 - otsu, sr_scale_used))
    except Exception:
        pass

    try:
        block = max(21, min(71, (min(sharp.shape[:2]) // 3) | 1))
        adap = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 3)
        variants.append(("qr_adaptive", adap, sr_scale_used))
        variants.append(("qr_adaptive_inv", 255 - adap, sr_scale_used))
        if morphology:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            closed = cv2.morphologyEx(adap, cv2.MORPH_CLOSE, k, iterations=1)
            opened = cv2.morphologyEx(adap, cv2.MORPH_OPEN, k, iterations=1)
            variants.append(("qr_adaptive_close", closed, sr_scale_used))
            variants.append(("qr_adaptive_open", opened, sr_scale_used))
    except Exception:
        pass

    # Deduplicate exact shapes/identity-looking arrays lightly by name only; keep
    # several binarizations because QR decoders are sensitive to thresholding.
    return variants


def build_fused_tag_variants(image: np.ndarray, cfg: Mapping[str, Any] | None = None) -> List[Tuple[str, np.ndarray, Dict[str, Any]]]:
    """Build additional debug/QR variants for an already fused track image."""
    cfg = cfg or {}
    out: List[Tuple[str, np.ndarray, Dict[str, Any]]] = []
    if image is None or image.size == 0:
        return out
    corrected, meta = enhance_fused_price_tag(
        image,
        enabled=bool(cfg.get("tag_correction_enabled", True)),
        sr_enabled=bool(cfg.get("tag_correction_sr_enabled", False)),
        sr_scale=float(cfg.get("tag_correction_sr_scale", cfg.get("sr_scale", 1.0))),
        sr_min_side=int(cfg.get("tag_correction_sr_min_side", cfg.get("sr_min_side", 320))),
        sr_max_side=int(cfg.get("tag_correction_sr_max_side", cfg.get("sr_max_side", 1400))),
        sr_method=str(cfg.get("tag_correction_sr_method", cfg.get("sr_method", "lanczos"))),
        glare_suppression=bool(cfg.get("tag_correction_glare_suppression", True)),
        clahe_clip=float(cfg.get("tag_correction_clahe_clip", 2.4)),
        text_gain=float(cfg.get("tag_correction_text_gain", 0.42)),
        red_zone_gain=float(cfg.get("tag_correction_red_zone_gain", 0.16)),
        profile=str(cfg.get("tag_correction_profile", "safe")),
        gray_world_strength=float(cfg.get("tag_correction_gray_world_strength", 0.35)),
        glare_max_area_ratio=float(cfg.get("tag_correction_glare_max_area_ratio", 0.045)),
        blackhat_kernel_ratio=float(cfg.get("tag_correction_blackhat_kernel_ratio", 0.020)),
        final_unsharp_amount=float(cfg.get("tag_correction_final_unsharp_amount", 0.10)),
    )
    if corrected is not None and corrected.size > 0:
        out.append(("tag_corrected", corrected, meta))
    return out


def _gray_world_balance(image: np.ndarray, *, strength: float = 0.35) -> np.ndarray:
    strength = max(0.0, min(1.0, float(strength)))
    if strength <= 1e-6:
        return image
    img = image.astype(np.float32)
    # Use central percentiles instead of the mean: price-tag crops often contain
    # large orange fields and blue shelf plastic, so full gray-world overcorrects.
    flat = img.reshape(-1, 3)
    lo = np.percentile(flat, 8, axis=0)
    hi = np.percentile(flat, 92, axis=0)
    mask = np.all((flat >= lo) & (flat <= hi), axis=1)
    sample = flat[mask] if int(mask.sum()) >= 256 else flat
    means = sample.mean(axis=0)
    gray = float(means.mean())
    gains = gray / np.maximum(means, 1.0)
    gains = np.clip(gains, 0.82, 1.18)
    gains = 1.0 + (gains - 1.0) * strength
    out = img * gains.reshape(1, 1, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def _suppress_small_glare(image: np.ndarray, *, max_area_ratio: float = 0.045) -> Tuple[np.ndarray, Dict[str, Any]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    glare = ((v > 238) & (s < 58)).astype(np.uint8) * 255
    count = int(cv2.countNonZero(glare))
    area_ratio = count / max(1, int(glare.size))
    info: Dict[str, Any] = {"area_ratio": round(float(area_ratio), 6), "pixel_count": count}
    if count <= 0:
        info["skipped"] = True
        info["reason"] = "glare_empty"
        return image, info
    if area_ratio > float(max_area_ratio):
        # Large white blobs are usually shelf reflections/background, not tiny
        # specular highlights.  Inpainting them creates the watercolor artifacts
        # seen in the bad tag_corrected examples.
        info["skipped"] = True
        info["reason"] = "glare_area_too_large"
        return image, info
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    glare = cv2.dilate(glare, kernel, iterations=1)
    out = cv2.inpaint(image, glare, 2.0, cv2.INPAINT_TELEA)
    info["skipped"] = False
    return out, info


def _boost_dark_text(image: np.ndarray, *, text_gain: float, red_zone_gain: float, kernel_ratio: float = 0.020) -> np.ndarray:
    img = ensure_bgr(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Black-hat: closing - original.  Keep the kernel small and mask the effect
    # to actual dark strokes; otherwise blurred plastic reflections become
    # black watercolor blobs.
    kx = max(5, int(round(min(gray.shape[:2]) * max(0.008, float(kernel_ratio)))) | 1)
    ky = max(3, (kx // 3) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    red_mask = (((h <= 24) | (h >= 168)) & (s >= 45) & (v >= 45)).astype(np.float32)
    bh_thr = max(5.0, float(np.percentile(blackhat, 72)))
    dark_thr = max(48.0, float(np.percentile(gray, 38)))
    text_mask = ((blackhat.astype(np.float32) >= bh_thr) & (gray.astype(np.float32) <= dark_thr + 45.0)).astype(np.float32)
    text_mask = cv2.GaussianBlur(text_mask, (0, 0), 0.75)

    gain_map = (float(text_gain) + float(red_zone_gain) * red_mask) * text_mask
    luma = gray.astype(np.float32) - blackhat.astype(np.float32) * gain_map
    luma = np.clip(luma, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    _, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge([luma, a, b]), cv2.COLOR_LAB2BGR)


def order_quad_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 4:
        raise ValueError("need at least 4 points")
    if len(pts) > 4:
        # Use the convex hull and approximate to four points when possible.
        hull = cv2.convexHull(pts).reshape(-1, 2)
        if len(hull) >= 4:
            pts = hull[:4]
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = pts[int(np.argmin(s))]  # top-left
    rect[2] = pts[int(np.argmax(s))]  # bottom-right
    rect[1] = pts[int(np.argmin(d))]  # top-right
    rect[3] = pts[int(np.argmax(d))]  # bottom-left
    return rect


def warp_qr_from_points(image: np.ndarray, points: Any, *, border: int = 18, min_size: int = 192, max_size: int = 768) -> Tuple[np.ndarray, np.ndarray]:
    """Perspective-normalize QR points detected in the current image space."""
    rect = order_quad_points(np.asarray(points, dtype=np.float32))
    width_a = float(np.linalg.norm(rect[2] - rect[3]))
    width_b = float(np.linalg.norm(rect[1] - rect[0]))
    height_a = float(np.linalg.norm(rect[1] - rect[2]))
    height_b = float(np.linalg.norm(rect[0] - rect[3]))
    size = int(round(max(width_a, width_b, height_a, height_b)))
    size = max(int(min_size), min(int(max_size), size + 2 * int(border)))
    dst = np.array([
        [border, border],
        [size - border - 1, border],
        [size - border - 1, size - border - 1],
        [border, size - border - 1],
    ], dtype=np.float32)
    mat = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(ensure_bgr(image), mat, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return warped, rect
