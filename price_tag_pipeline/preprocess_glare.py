"""
Glare/haze suppression preprocessing for small price-tag crops.

This module is intentionally deterministic and OpenCV-only.  It is not a
photometric restoration model; it prepares difficult mobile-video crops for OCR.
The default mode is conservative and opt-in.

Available methods:
- ``clahe``: contrast equalization in LAB/L channel;
- ``inpaint_glare``: detect low-saturation over-bright specular blobs and inpaint;
- ``dark_channel``: lightweight dark-channel-prior dehazing;
- ``hybrid``/``auto``: inpaint_glare -> dark_channel -> CLAHE, with blending.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import cv2
import numpy as np

from .io_utils import imwrite_unicode


@dataclass
class GlareSuppressionMeta:
    enabled: bool
    applied: bool
    method: str = "none"
    input_shape: list[int] | None = None
    output_shape: list[int] | None = None
    glare_mask_ratio: float = 0.0
    dark_channel_mean: float = 0.0
    transmission_mean: float = 0.0
    blend_alpha: float = 1.0
    warnings: list[str] | None = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["warnings"] = d.get("warnings") or []
        return d


class GlareSuppressor:
    def __init__(
        self,
        *,
        enabled: bool = False,
        method: str = "hybrid",
        blend_alpha: float = 0.72,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: int = 8,
        glare_v_threshold: int = 210,
        glare_s_threshold: int = 72,
        glare_min_area_ratio: float = 0.002,
        glare_dilate_px: int = 2,
        inpaint_radius: float = 3.0,
        dcp_patch_size: int = 9,
        dcp_omega: float = 0.78,
        dcp_t0: float = 0.22,
        dcp_top_percent: float = 0.001,
        dcp_guided_blur: int = 9,
        max_work_side: int = 720,
        debug: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.method = str(method or "hybrid").lower().strip()
        self.blend_alpha = float(blend_alpha)
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_tile_grid_size = int(clahe_tile_grid_size)
        self.glare_v_threshold = int(glare_v_threshold)
        self.glare_s_threshold = int(glare_s_threshold)
        self.glare_min_area_ratio = float(glare_min_area_ratio)
        self.glare_dilate_px = int(glare_dilate_px)
        self.inpaint_radius = float(inpaint_radius)
        self.dcp_patch_size = int(dcp_patch_size)
        self.dcp_omega = float(dcp_omega)
        self.dcp_t0 = float(dcp_t0)
        self.dcp_top_percent = float(dcp_top_percent)
        self.dcp_guided_blur = int(dcp_guided_blur)
        self.max_work_side = int(max_work_side)
        self.debug = bool(debug)
        self._last_meta = GlareSuppressionMeta(enabled=self.enabled, applied=False, method=self.method)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "GlareSuppressor":
        gcfg = cfg.get("glare_suppression", {}) if isinstance(cfg, Mapping) else {}
        return cls(
            enabled=bool(gcfg.get("enabled", False)),
            method=str(gcfg.get("method", "hybrid")),
            blend_alpha=float(gcfg.get("blend_alpha", 0.72)),
            clahe_clip_limit=float(gcfg.get("clahe_clip_limit", 2.0)),
            clahe_tile_grid_size=int(gcfg.get("clahe_tile_grid_size", 8)),
            glare_v_threshold=int(gcfg.get("glare_v_threshold", 210)),
            glare_s_threshold=int(gcfg.get("glare_s_threshold", 72)),
            glare_min_area_ratio=float(gcfg.get("glare_min_area_ratio", 0.002)),
            glare_dilate_px=int(gcfg.get("glare_dilate_px", 2)),
            inpaint_radius=float(gcfg.get("inpaint_radius", 3.0)),
            dcp_patch_size=int(gcfg.get("dcp_patch_size", 9)),
            dcp_omega=float(gcfg.get("dcp_omega", 0.78)),
            dcp_t0=float(gcfg.get("dcp_t0", 0.22)),
            dcp_top_percent=float(gcfg.get("dcp_top_percent", 0.001)),
            dcp_guided_blur=int(gcfg.get("dcp_guided_blur", 9)),
            max_work_side=int(gcfg.get("max_work_side", 720)),
            debug=bool(gcfg.get("debug", False)),
        )

    def get_debug_info(self) -> Dict[str, Any]:
        return self._last_meta.to_dict()

    def apply(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        if image is None or image.size == 0:
            self._last_meta = GlareSuppressionMeta(enabled=self.enabled, applied=False, method=self.method, warnings=["empty_image"])
            return image, self._last_meta.to_dict()
        if not self.enabled or self.method in {"", "none", "off", "false"}:
            self._last_meta = GlareSuppressionMeta(enabled=False, applied=False, method="none", input_shape=list(image.shape), output_shape=list(image.shape))
            return image.copy(), self._last_meta.to_dict()

        original = image.copy()
        work, scale = _resize_for_work(image, self.max_work_side)
        method = self.method
        warnings: list[str] = []
        mask_ratio = 0.0
        dark_mean = 0.0
        trans_mean = 0.0

        try:
            out = work.copy()
            if method in {"inpaint", "inpaint_glare", "hybrid", "auto"}:
                out, mask_ratio = self._inpaint_glare(out)
            if method in {"dark_channel", "dcp", "hybrid", "auto"}:
                out, dmeta = self._dark_channel_dehaze(out)
                dark_mean = float(dmeta.get("dark_channel_mean", 0.0))
                trans_mean = float(dmeta.get("transmission_mean", 0.0))
            if method in {"clahe", "contrast", "hybrid", "auto"}:
                out = self._clahe_luminance(out)

            if scale != 1.0:
                out = cv2.resize(out, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_CUBIC)
            alpha = max(0.0, min(1.0, self.blend_alpha))
            if method in {"clahe", "contrast"}:
                alpha = max(alpha, 0.85)
            blended = cv2.addWeighted(out, alpha, original, 1.0 - alpha, 0.0)
            meta = GlareSuppressionMeta(
                enabled=True,
                applied=True,
                method=method,
                input_shape=list(image.shape),
                output_shape=list(blended.shape),
                glare_mask_ratio=round(float(mask_ratio), 6),
                dark_channel_mean=round(float(dark_mean), 6),
                transmission_mean=round(float(trans_mean), 6),
                blend_alpha=round(float(alpha), 4),
                warnings=warnings,
            )
            self._last_meta = meta
            return blended, meta.to_dict()
        except Exception as e:  # keep OCR pipeline alive
            warnings.append(f"glare_suppression_failed:{type(e).__name__}:{e}")
            meta = GlareSuppressionMeta(
                enabled=True,
                applied=False,
                method=method,
                input_shape=list(image.shape),
                output_shape=list(image.shape),
                warnings=warnings,
            )
            self._last_meta = meta
            return image.copy(), meta.to_dict()

    def _clahe_luminance(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        grid = max(2, int(self.clahe_tile_grid_size))
        clahe = cv2.createCLAHE(clipLimit=max(0.5, float(self.clahe_clip_limit)), tileGridSize=(grid, grid))
        l2 = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

    def _inpaint_glare(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        mask = ((v >= self.glare_v_threshold) & (s <= self.glare_s_threshold)).astype(np.uint8) * 255
        # Avoid inpainting large white price-tag background.  Keep only local blobs
        # whose area is not too large relative to the crop.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros_like(mask)
        area_total = max(1, mask.shape[0] * mask.shape[1])
        min_area = max(3, int(area_total * self.glare_min_area_ratio))
        max_area = max(min_area + 1, int(area_total * 0.18))
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if min_area <= area <= max_area:
                keep[labels == i] = 255
        if self.glare_dilate_px > 0 and np.any(keep):
            k = 2 * int(self.glare_dilate_px) + 1
            keep = cv2.dilate(keep, np.ones((k, k), np.uint8), iterations=1)
        ratio = float(np.count_nonzero(keep) / area_total)
        if ratio <= 0.0:
            return image, 0.0
        out = cv2.inpaint(image, keep, float(self.inpaint_radius), cv2.INPAINT_TELEA)
        return out, ratio

    def _dark_channel_dehaze(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        img = image.astype(np.float32) / 255.0
        dark = self._dark_channel(img, patch_size=self.dcp_patch_size)
        a = self._estimate_atmospheric_light(img, dark, top_percent=self.dcp_top_percent)
        a = np.maximum(a, 1e-3)
        normed = img / a.reshape(1, 1, 3)
        trans = 1.0 - float(self.dcp_omega) * self._dark_channel(normed, patch_size=self.dcp_patch_size)
        trans = np.clip(trans, float(self.dcp_t0), 1.0)
        k = int(self.dcp_guided_blur)
        if k > 1:
            if k % 2 == 0:
                k += 1
            trans = cv2.GaussianBlur(trans, (k, k), 0)
            trans = np.clip(trans, float(self.dcp_t0), 1.0)
        j = (img - a.reshape(1, 1, 3)) / trans[:, :, None] + a.reshape(1, 1, 3)
        j = np.clip(j, 0.0, 1.0)
        out = (j * 255.0 + 0.5).astype(np.uint8)
        return out, {"dark_channel_mean": float(np.mean(dark)), "transmission_mean": float(np.mean(trans))}

    @staticmethod
    def _dark_channel(img: np.ndarray, patch_size: int = 9) -> np.ndarray:
        patch = max(3, int(patch_size))
        if patch % 2 == 0:
            patch += 1
        min_rgb = np.min(img, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
        return cv2.erode(min_rgb, kernel)

    @staticmethod
    def _estimate_atmospheric_light(img: np.ndarray, dark: np.ndarray, top_percent: float = 0.001) -> np.ndarray:
        h, w = dark.shape[:2]
        n = max(1, int(h * w * max(1e-6, float(top_percent))))
        flat_dark = dark.reshape(-1)
        idx = np.argpartition(flat_dark, -n)[-n:]
        flat_img = img.reshape(-1, 3)
        # Among highest dark-channel pixels, choose highest RGB-intensity pixel.
        candidates = flat_img[idx]
        best = int(np.argmax(np.sum(candidates, axis=1)))
        return candidates[best].astype(np.float32)


def _resize_for_work(image: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    m = max(h, w)
    max_side = int(max_side or 0)
    if max_side <= 0 or m <= max_side:
        return image.copy(), 1.0
    scale = max_side / float(max(1, m))
    resized = cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    return resized, scale


def apply_glare_suppression_from_config(
    image: np.ndarray,
    cfg: Mapping[str, Any],
    *,
    debug_dir: Optional[Path] = None,
    stem: str = "image",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    suppressor = GlareSuppressor.from_config(cfg)
    out, meta = suppressor.apply(image)
    gcfg = cfg.get("glare_suppression", {}) if isinstance(cfg, Mapping) else {}
    if bool(gcfg.get("save_debug", False)) and debug_dir is not None and meta.get("applied"):
        debug_dir.mkdir(parents=True, exist_ok=True)
        imwrite_unicode(debug_dir / f"{stem}_glare_suppressed.jpg", out)
    return out, meta
