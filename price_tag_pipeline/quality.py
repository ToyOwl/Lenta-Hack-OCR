"""
Image quality checks for already-cropped price tags.
"""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np


def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_stats(gray: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(gray)),
        "std": float(np.std(gray)),
        "underexposed_ratio": float(np.mean(gray < 25)),
        "overexposed_ratio": float(np.mean(gray > 245)),
    }


def compute_quality(
    image: np.ndarray,
    min_w: int = 120,
    min_h: int = 80,
    blur_warn: float = 60.0,
) -> Dict[str, Any]:
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = variance_of_laplacian(gray)
    bstats = brightness_stats(gray)
    warnings = []
    if w < min_w or h < min_h:
        warnings.append("low_resolution")
    if blur < blur_warn:
        warnings.append("blur_or_motion")
    if bstats["overexposed_ratio"] > 0.25:
        warnings.append("overexposed")
    if bstats["underexposed_ratio"] > 0.25:
        warnings.append("underexposed")
    return {
        "width": int(w),
        "height": int(h),
        "blur_score_laplacian_var": blur,
        "brightness": bstats,
        "status": "ok" if not warnings else "warning",
        "warnings": warnings,
    }
