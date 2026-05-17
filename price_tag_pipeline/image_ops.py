"""
Common image operations
"""

from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np

from .pipeline_types import Box


def crop_box(image: np.ndarray, box: Box) -> np.ndarray:
    h, w = image.shape[:2]
    b = box.clip(w, h)
    return image[b.y1:b.y2, b.x1:b.x2].copy()

def normalize_for_ocr(image: np.ndarray, clahe: bool = True, sharpen: bool = True) -> np.ndarray:
    """Conservative OCR normalization. Returns BGR image."""

    out = image.copy()
    if out.size == 0:
        return out

    if clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe_obj.apply(l)
        out = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

    if sharpen:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        out = cv2.filter2D(out, -1, kernel)
    return out

def upscale_if_small(image: np.ndarray, min_side: int = 256, max_scale: int = 4) -> Tuple[np.ndarray, float]:

    h, w = image.shape[:2]
    side = min(h, w)

    if side >= min_side:
       return image, 1.0

    scale = min(max_scale, int(math.ceil(min_side / max(1, side))))
    out = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return out, float(scale)

def box_iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    return float(inter / union) if union > 0 else 0.0

def merge_same_class_boxes(boxes: list[Box], iou_thr: float = 0.65) -> list[Box]:

    out: list[Box] = []

    for b in sorted(boxes, key=lambda x: x.conf, reverse=True):
        keep = True
        for ob in out:
            if b.cls == ob.cls and box_iou(b, ob) > iou_thr:
                keep = False
                break
        if keep:
            out.append(b)
    return out
