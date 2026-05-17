# -*- coding: utf-8 -*-
"""
Color Names image descriptor for price-tag template classification.

This module implements the Color Names descriptor as an image descriptor.
For every pixel it estimates an 11-dimensional vector of
basic color-name probabilities and then averages those probabilities over the
whole image or over fixed spatial regions.

Backends
--------
lab_soft
    Deterministic CIE-Lab soft assignment to 11 basic color-name prototypes.
    It does not require external files.

w2c_lut
    Uses the 32x32x32 -> 11 Color Names lookup table exported as .npy/.npz.
    The expected color order is:
      black, blue, brown, gray, green, orange, pink, purple, red, white, yellow

Important W2C indexing note
---------------------------
The original van de Weijer W2C MATLAB LUT is normally indexed as:
    idx = R_bin + 32 * G_bin + 32 * 32 * B_bin
where each bin is floor(channel / 8), and R is the fastest-varying channel.
If this is wrong, red/orange price-tag regions often appear as blue/purple.
Use w2c_index_order='rgb_fast' for the original W2C LUT.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import cv2
import numpy as np

ColorDescriptorMap = Dict[str, float]

@dataclass(frozen=True)
class RegionSpec:
    name: str
    x1: float
    y1: float
    x2: float
    y2: float

DEFAULT_REGIONS: Tuple[RegionSpec, ...] = (
    RegionSpec("full", 0.0, 0.0, 1.0, 1.0),
    RegionSpec("top", 0.0, 0.0, 1.0, 0.33),
    RegionSpec("middle", 0.0, 0.25, 1.0, 0.75),
    RegionSpec("bottom", 0.0, 0.55, 1.0, 1.0),
    RegionSpec("left", 0.0, 0.0, 0.5, 1.0),
    RegionSpec("right", 0.5, 0.0, 1.0, 1.0),
    RegionSpec("upper_left", 0.0, 0.0, 0.5, 0.5),
    RegionSpec("upper_right", 0.5, 0.0, 1.0, 0.5),
    RegionSpec("lower_left", 0.0, 0.5, 0.5, 1.0),
    RegionSpec("lower_right", 0.5, 0.5, 1.0, 1.0),
)

COLOR_NAMES_11: Tuple[str, ...] = (
    "black",
    "blue",
    "brown",
    "gray",
    "green",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
)

NAMED_COLOR_ORDER: Tuple[str, ...] = COLOR_NAMES_11

_SRGB_PROTOTYPES_RGB = np.array(
    [
        [0, 0, 0],        # black
        [0, 0, 255],      # blue
        [150, 75, 0],     # brown
        [128, 128, 128],  # gray
        [0, 128, 0],      # green
        [255, 165, 0],    # orange
        [255, 192, 203],  # pink
        [128, 0, 128],    # purple
        [255, 0, 0],      # red
        [255, 255, 255],  # white
        [255, 255, 0],    # yellow
    ],
    dtype=np.uint8,
)

def _rgb_to_lab_float(rgb: np.ndarray) -> np.ndarray:
    rgb_u8 = np.asarray(rgb, dtype=np.uint8)
    if rgb_u8.ndim == 2:
        rgb_u8 = rgb_u8.reshape(1, -1, 3)
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab.reshape(-1, 3)

_LAB_PROTOTYPES = _rgb_to_lab_float(_SRGB_PROTOTYPES_RGB.reshape(1, -1, 3))

def _region_pixels(image: np.ndarray, region: RegionSpec) -> Tuple[int, int, int, int]:
    h, w = image.shape[:2]
    x1 = max(0, min(w, int(round(region.x1 * w))))
    y1 = max(0, min(h, int(round(region.y1 * h))))
    x2 = max(0, min(w, int(round(region.x2 * w))))
    y2 = max(0, min(h, int(round(region.y2 * h))))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def _resize_for_descriptor(image_bgr: np.ndarray, max_pixels: int) -> np.ndarray:
    if max_pixels <= 0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    n = h * w
    if n <= max_pixels:
        return image_bgr
    scale = float(np.sqrt(max_pixels / max(1, n)))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _softmax_neg_dist2(dist2: np.ndarray, temperature: float) -> np.ndarray:
    temperature = max(float(temperature), 1e-6)
    logits = -dist2 / temperature
    logits = logits - np.max(logits, axis=1, keepdims=True)
    expv = np.exp(logits)
    denom = np.sum(expv, axis=1, keepdims=True)
    return expv / np.maximum(denom, 1e-12)

class ColorNamesDescriptor:

    def __init__(self,
                 backend: str = "lab_soft",
                 w2c_lut_path: str | Path | None = None,
                 temperature: float = 950.0,
                 max_pixels: int = 80_000,
                 w2c_index_order: str = "rgb_fast", ) -> None:

        self.backend = str(backend)
        self.temperature = float(temperature)
        self.max_pixels = int(max_pixels)
        self.w2c_index_order = str(w2c_index_order or "rgb_fast")
        self.w2c_lut: Optional[np.ndarray] = None

        if self.backend not in {"lab_soft", "w2c_lut"}:
            raise ValueError(f"Unsupported Color Names backend: {backend}")

        if self.w2c_index_order not in {"rgb_fast", "rgb_slow", "bgr_fast", "bgr_slow"}:
            raise ValueError(
                "w2c_index_order must be one of: rgb_fast, rgb_slow, bgr_fast, bgr_slow")

        if self.backend == "w2c_lut":
            if not w2c_lut_path:
                raise ValueError("backend='w2c_lut' requires color_names.lut_path")
            self.w2c_lut = self._load_w2c_lut(Path(w2c_lut_path))

    @staticmethod
    def _load_w2c_lut(path: Path) -> np.ndarray:

        if not path.exists():
            raise FileNotFoundError(str(path))

        if path.suffix.lower() == ".npz":
            data = np.load(str(path))
            if "w2c" in data:
                arr = data["w2c"]
            elif "lut" in data:
                arr = data["lut"]
            else:
                first_key = list(data.keys())[0]
                arr = data[first_key]
        else:
            arr = np.load(str(path))
        arr = np.asarray(arr, dtype=np.float32)

        if arr.shape[-1] == 11:
            arr = arr.reshape(-1, 11)

        elif arr.shape[0] == 11:
            arr = np.moveaxis(arr, 0, -1).reshape(-1, 11)
        else:
            arr = arr.reshape(-1, 11)
        if arr.shape[0] != 32 * 32 * 32 or arr.shape[1] != 11:
            raise ValueError(f"Expected W2C LUT with 32768x11 values, got {arr.shape}")

        arr = np.maximum(arr, 0.0)
        row_sum = np.sum(arr, axis=1, keepdims=True)
        arr = arr / np.maximum(row_sum, 1e-12)

        return arr.astype(np.float32)

    def _w2c_indices(self, image_bgr: np.ndarray) -> np.ndarray:

        if self.w2c_index_order.startswith("rgb"):
            arr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3)
        else:
            arr = image_bgr.reshape(-1, 3)

        bins = (arr.astype(np.uint16) // 8).clip(0, 31)
        c0 = bins[:, 0]
        c1 = bins[:, 1]
        c2 = bins[:, 2]
        if self.w2c_index_order.endswith("fast"):
           return c0 + 32 * c1 + 32 * 32 * c2

        return c0 * 32 * 32 + c1 * 32 + c2

    def pixel_probabilities(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr.size == 0:
            return np.zeros((0, len(COLOR_NAMES_11)), dtype=np.float32)

        img = _resize_for_descriptor(image_bgr, self.max_pixels)

        if self.backend == "w2c_lut":
            assert self.w2c_lut is not None
            idx = self._w2c_indices(img)
            return self.w2c_lut[idx].astype(np.float32)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lab = _rgb_to_lab_float(rgb.reshape(1, -1, 3))
        diff = lab[:, None, :] - _LAB_PROTOTYPES[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)

        return _softmax_neg_dist2(dist2, self.temperature).astype(np.float32)

    def describe(self, image_bgr: np.ndarray) -> ColorDescriptorMap:

        if image_bgr.size == 0:
            return {name: 0.0 for name in COLOR_NAMES_11}
        probs = self.pixel_probabilities(image_bgr)

        if probs.size == 0:
            return {name: 0.0 for name in COLOR_NAMES_11}
        vec = np.mean(probs, axis=0)

        return {name: float(vec[i]) for i, name in enumerate(COLOR_NAMES_11)}

    def describe_regions(self, image_bgr: np.ndarray, regions: Tuple[RegionSpec, ...] = DEFAULT_REGIONS,) -> Dict[str, ColorDescriptorMap]:
        feats: Dict[str, ColorDescriptorMap] = {}

        if image_bgr.size == 0:
            return feats

        for r in regions:
            x1, y1, x2, y2 = _region_pixels(image_bgr, r)
            crop = image_bgr[y1:y2, x1:x2]
            feats[r.name] = self.describe(crop)

        return feats

    def flattened_regions(self, image_bgr: np.ndarray, regions: Tuple[RegionSpec, ...] = DEFAULT_REGIONS, ) -> Dict[str, float]:
        region_desc = self.describe_regions(image_bgr, regions=regions)
        flat: Dict[str, float] = {}

        for region_name, desc in region_desc.items():
            for color_name, value in desc.items():
                flat[f"cn.{region_name}.{color_name}"] = float(value)

        return flat

_DEFAULT_DESCRIPTOR = ColorNamesDescriptor()

def color_name_descriptor(image_bgr: np.ndarray, descriptor: Optional[ColorNamesDescriptor] = None,) -> ColorDescriptorMap:
    return (descriptor or _DEFAULT_DESCRIPTOR).describe(image_bgr)

def region_color_features(image_bgr: np.ndarray, regions: Tuple[RegionSpec, ...] = DEFAULT_REGIONS, descriptor: Optional[ColorNamesDescriptor] = None,) -> Dict[str, ColorDescriptorMap]:
    return (descriptor or _DEFAULT_DESCRIPTOR).describe_regions(image_bgr, regions=regions)

def combined_ratio(region_feats: Mapping[str, Mapping[str, float]], region: str, names: Iterable[str]) -> float:
    vals = region_feats.get(region, {})
    return float(sum(float(vals.get(n, 0.0)) for n in names))

def dominant_colors(ratios: Mapping[str, float], top_k: int = 4) -> List[Tuple[str, float]]:
    return sorted([(k, float(v)) for k, v in ratios.items()], key=lambda x: x[1], reverse=True)[:top_k]

def named_color_masks(image_bgr: np.ndarray, descriptor: Optional[ColorNamesDescriptor] = None) -> Dict[str, np.ndarray]:
    desc = descriptor or _DEFAULT_DESCRIPTOR
    img = _resize_for_descriptor(image_bgr, desc.max_pixels)
    probs = desc.pixel_probabilities(img)
    if img.size == 0 or probs.size == 0:
        return {name: np.zeros((0, 0), dtype=bool) for name in COLOR_NAMES_11}
    labels = np.argmax(probs, axis=1).reshape(img.shape[:2])
    return {name: labels == i for i, name in enumerate(COLOR_NAMES_11)}

def color_ratios(image_bgr: np.ndarray, masks: Optional[Mapping[str, np.ndarray]] = None) -> ColorDescriptorMap:
    if masks is not None:
        return {name: float(np.mean(masks.get(name, np.zeros((0, 0), dtype=bool)))) for name in COLOR_NAMES_11}
    return color_name_descriptor(image_bgr)
