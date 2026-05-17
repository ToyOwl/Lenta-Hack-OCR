"""
Template classifier based on the Color Names image descriptor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from .color_names import ColorNamesDescriptor, combined_ratio, dominant_colors
from .pipeline_types import TemplateResult


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _warm_promo_ratio(f: Mapping[str, Mapping[str, float]], region: str) -> float:
    # Lenta orange/red promo paper is often mapped to brown by W2C/Lab under store lighting.
    return combined_ratio(f, region, ["red", "orange", "brown", "pink"])

def _yellow_promo_ratio(f: Mapping[str, Mapping[str, float]], region: str) -> float:
    return combined_ratio(f, region, ["yellow", "orange", "brown"])

def _cool_artifact_ratio(f: Mapping[str, Mapping[str, float]], region: str) -> float:
    return combined_ratio(f, region, ["blue", "purple"])

def _score_shelf_red_promo(f: Mapping[str, Mapping[str, float]], aspect: float) -> float:
    bottom_warm = _warm_promo_ratio(f, "bottom")
    lower_left_warm = _warm_promo_ratio(f, "lower_left")
    lower_right_warm = _warm_promo_ratio(f, "lower_right")
    top_white_gray = combined_ratio(f, "top", ["white", "gray"])
    full_black = combined_ratio(f, "full", ["black"])
    top_yellow = combined_ratio(f, "top", ["yellow", "orange"])
    bottom_cool = _cool_artifact_ratio(f, "bottom")

    score = 0.0
    score += 1.45 * bottom_warm
    score += 0.75 * lower_left_warm
    score += 0.45 * lower_right_warm
    score += 0.22 * top_white_gray
    score += 0.15 * min(full_black * 2.5, 1.0)
    score -= 0.45 * top_yellow

    # If original W2C index order is wrong, red becomes blue/purple. Penalize it so it is visible.
    score -= 0.55 * max(0.0, bottom_cool - 0.25)
    if 0.55 <= aspect <= 2.20:
        score += 0.10
    return _clip01(score)

def _score_hanging_yellow_promo(f: Mapping[str, Mapping[str, float]], aspect: float) -> float:
    top_yellow = _yellow_promo_ratio(f, "top")
    upper_yellow = 0.5 * (_yellow_promo_ratio(f, "upper_left") + _yellow_promo_ratio(f, "upper_right"))
    full_white_gray = combined_ratio(f, "full", ["white", "gray"])
    bottom_warm = _warm_promo_ratio(f, "bottom")
    full_black = combined_ratio(f, "full", ["black"])

    score = 0.0
    score += 1.60 * top_yellow
    score += 0.70 * upper_yellow
    score += 0.18 * full_white_gray
    score += 0.18 * min(full_black * 2.5, 1.0)
    score -= 0.65 * bottom_warm
    if 0.45 <= aspect <= 4.2:
        score += 0.10
    return _clip01(score)

def _score_shelf_white_regular(f: Mapping[str, Mapping[str, float]], aspect: float) -> float:

    full_white_gray = combined_ratio(f, "full", ["white", "gray"])
    top_white_gray = combined_ratio(f, "top", ["white", "gray"])
    bottom_warm = _warm_promo_ratio(f, "bottom")
    top_yellow = combined_ratio(f, "top", ["yellow", "orange"])
    full_black = combined_ratio(f, "full", ["black"])

    score = 0.0
    score += 0.75 * full_white_gray
    score += 0.45 * top_white_gray
    score += 0.20 * min(full_black * 2.5, 1.0)
    score -= 1.20 * bottom_warm
    score -= 0.75 * top_yellow
    if 0.45 <= aspect <= 2.6:
        score += 0.10
    return _clip01(score)


def _score_progressive(f: Mapping[str, Mapping[str, float]], aspect: float) -> float:
    """Progressive shelf tag: white/gray top + warm/yellow price block below."""
    top_white_gray = combined_ratio(f, "top", ["white", "gray"])
    full_white_gray = combined_ratio(f, "full", ["white", "gray"])
    top_warm = _warm_promo_ratio(f, "top")
    middle_warm = _warm_promo_ratio(f, "middle")
    bottom_warm = _warm_promo_ratio(f, "bottom")
    lower_warm = 0.5 * (_warm_promo_ratio(f, "lower_left") + _warm_promo_ratio(f, "lower_right"))
    full_black = combined_ratio(f, "full", ["black"])

    score = 0.0
    score += 0.72 * top_white_gray
    score += 0.42 * full_white_gray
    score += 1.05 * bottom_warm
    score += 0.42 * lower_warm
    score += 0.22 * middle_warm
    score += 0.12 * min(full_black * 2.5, 1.0)
    score -= 0.55 * max(0.0, top_warm - 0.23)
    if 0.55 <= aspect <= 2.40:
        score += 0.12
    return _clip01(score)


def _score_progressive_yellow(f: Mapping[str, Mapping[str, float]], aspect: float) -> float:
    """Progressive yellow promo variant: wider yellow/orange body with large price."""
    full_yellow = _yellow_promo_ratio(f, "full")
    middle_yellow = _yellow_promo_ratio(f, "middle")
    bottom_yellow = _yellow_promo_ratio(f, "bottom")
    top_white_gray = combined_ratio(f, "top", ["white", "gray"])
    full_black = combined_ratio(f, "full", ["black"])
    score = 0.0
    score += 0.60 * full_yellow
    score += 0.70 * middle_yellow
    score += 0.95 * bottom_yellow
    score += 0.20 * top_white_gray
    score += 0.12 * min(full_black * 2.5, 1.0)
    if 0.45 <= aspect <= 3.20:
        score += 0.10
    return _clip01(score)

def _score_unknown_or_bad(f: Mapping[str, Mapping[str, float]], aspect: float, quality: Mapping[str, Any]) -> float:

    full = f.get("full", {})
    values = np.array([float(v) for v in full.values()], dtype=np.float32)

    if values.size:
        entropy = float(-np.sum(values * np.log(np.maximum(values, 1e-8))) / np.log(max(2, values.size)))
        max_prob = float(np.max(values))
    else:
        entropy = 1.0
        max_prob = 0.0

    warnings = quality.get("warnings") or []
    score = 0.12

    if warnings:
        score += 0.14

    if entropy > 0.82 and max_prob < 0.30:
        score += 0.18

    if aspect < 0.25 or aspect > 5.0:
        score += 0.15
    return _clip01(score)


class ColorNameTemplateClassifier:
    def __init__(self,
                 descriptor: Optional[ColorNamesDescriptor] = None,
                 cn_backend: str = "lab_soft",
                 cn_lut_path: str | Path | None = None,
                 cn_temperature: float = 950.0,
                 cn_max_pixels: int = 80_000,
                 cn_w2c_index_order: str = "rgb_fast",) -> None:

     self.descriptor = descriptor or\
        ColorNamesDescriptor(backend=cn_backend, w2c_lut_path=cn_lut_path, temperature=cn_temperature, max_pixels=cn_max_pixels, w2c_index_order=cn_w2c_index_order,)

    def classify(self, image_bgr: np.ndarray, quality: Mapping[str, Any] | None = None) -> TemplateResult:

        if quality is None:
            quality = {}

        h, w = image_bgr.shape[:2]
        aspect = float(w / max(1, h))

        f = self.descriptor.describe_regions(image_bgr)

        scores =\
        {"shelf_red_promo": _score_shelf_red_promo(f, aspect), "hanging_yellow_promo_large": _score_hanging_yellow_promo(f, aspect),
         "shelf_white_regular": _score_shelf_white_regular(f, aspect), "progressive": _score_progressive(f, aspect),
         "progressive_yellow": _score_progressive_yellow(f, aspect), "small_blurry_or_unknown": _score_unknown_or_bad(f, aspect, quality),}

        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_name = ordered[0][0]
        best_score = float(ordered[0][1])
        second_score = float(ordered[1][1]) if len(ordered) > 1 else 0.0
        margin = best_score - second_score

        notes: List[str] = []
        bottom_cool = _cool_artifact_ratio(f, "bottom")
        bottom_warm = _warm_promo_ratio(f, "bottom")

        if bottom_cool > 0.45 and bottom_warm < 0.15 and self.descriptor.backend == "w2c_lut":
            notes.append("w2c_possible_index_order_problem_bottom_is_blue_purple")

        if margin < 0.08:
            notes.append("low_template_margin")

        if quality.get("warnings"):
            notes.extend([f"quality:{x}" for x in quality.get("warnings", [])])

        confidence = _clip01(0.55 * best_score + 0.45 * min(1.0, margin * 4.0))

        if best_score < 0.26:
            best_name = "small_blurry_or_unknown"
            confidence = min(confidence, 0.42)
            notes.append("weak_color_names_descriptor_signal")

        color_features: Dict[str, Any] = {
            "descriptor": "color_names_11",
            "backend": self.descriptor.backend,
            "w2c_index_order": self.descriptor.w2c_index_order,
            "aspect_ratio": aspect,
            "regions": f,
            "dominant_full": dominant_colors(f.get("full", {}), top_k=5),
            "dominant_top": dominant_colors(f.get("top", {}), top_k=5),
            "dominant_bottom": dominant_colors(f.get("bottom", {}), top_k=5),
            "combined": {
                "full_white_gray": combined_ratio(f, "full", ["white", "gray"]),
                "top_yellow_orange": combined_ratio(f, "top", ["yellow", "orange"]),
                "top_yellow_orange_brown": _yellow_promo_ratio(f, "top"),
                "top_warm_promo": _warm_promo_ratio(f, "top"),
                "middle_warm_promo": _warm_promo_ratio(f, "middle"),
                "bottom_warm_promo": _warm_promo_ratio(f, "bottom"),
                "bottom_red_orange": combined_ratio(f, "bottom", ["red", "orange"]),
                "bottom_red_orange_brown_pink": bottom_warm,
                "bottom_cool_blue_purple": bottom_cool,
                "full_black": combined_ratio(f, "full", ["black"]),
            },
        }
        return TemplateResult(
            template_name=best_name,
            confidence=confidence,
            scores={k: float(v) for k, v in scores.items()},
            color_features=color_features,
            notes=notes,
        )


# Backward-compatible alias. Some pipeline revisions import TemplateClassifier.
#TemplateClassifier = ColorNameTemplateClassifier
