"""
Input-structure router: single cropped price tag vs shelf price rail.

This module  performs  geometric/color comparison before  OCR stage:

raw image -> rail hypothesis from PriceRailSplitter -> decision

The important distinction is that rail segmentation must not be the first
semantic assumption.  A wide progressive price tag can contain several price
columns, and a shelf photo can contain several physically separate tags.  The
router compares both hypotheses and only enters price-rail mode when the rail
hypothesis has enough evidence: multiple plausible cells, long rail band,
separator evidence or a very wide rail crop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .debug_vis import draw_label
from .pipeline_types import Box


@dataclass
class InputStructureDecision:
    """Decision made before OCR/layout parsing."""
    mode: str  # single_tag | price_rail
    confidence: float
    reason: str
    features: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InputStructureAnalyzer:
    """Decides whether the current input is a single tag or a rail of tags."""

    def __init__(
        self,
        mode: str = "auto",
        min_cells_for_rail: int = 2,
        decision_threshold: float = 0.58,
        min_rail_width_ratio: float = 0.42,
        min_rail_area_ratio: float = 0.045,
        max_single_like_aspect: float = 2.35,
        wide_crop_aspect: float = 2.65,
        require_separator_or_wide: bool = True,
        min_separator_count: int = 1,
        enable_single_tag_veto: bool = True,
        single_lower_band_y_ratio: float = 0.24,
        single_lower_band_max_height_ratio: float = 0.58,
        single_tag_veto_threshold: float = 0.62,
        full_height_separator_min_count: int = 1,
        full_height_separator_min_strength: float = 0.035,
        full_rail_top_y_ratio: float = 0.20,
        full_rail_min_height_ratio: float = 0.52,
        high_cell_width_cv_threshold: float = 0.34,
        lower_band_many_cells_veto_min_count: int = 4,
        lower_band_many_cells_max_aspect: float = 2.10,
        top_context_min_height_ratio: float = 0.30,
        top_context_min_paper_ratio: float = 0.42,
        suppress_stable_repeated_counter_on_lower_band: bool = True,
        debug: bool = False,
    ) -> None:
        self.mode = str(mode or "auto").lower().strip()
        self.min_cells_for_rail = int(min_cells_for_rail)
        self.decision_threshold = float(decision_threshold)
        self.min_rail_width_ratio = float(min_rail_width_ratio)
        self.min_rail_area_ratio = float(min_rail_area_ratio)
        self.max_single_like_aspect = float(max_single_like_aspect)
        self.wide_crop_aspect = float(wide_crop_aspect)
        self.require_separator_or_wide = bool(require_separator_or_wide)
        self.min_separator_count = int(min_separator_count)
        self.enable_single_tag_veto = bool(enable_single_tag_veto)
        self.single_lower_band_y_ratio = float(single_lower_band_y_ratio)
        self.single_lower_band_max_height_ratio = float(single_lower_band_max_height_ratio)
        self.single_tag_veto_threshold = float(single_tag_veto_threshold)
        self.full_height_separator_min_count = int(full_height_separator_min_count)
        self.full_height_separator_min_strength = float(full_height_separator_min_strength)
        self.full_rail_top_y_ratio = float(full_rail_top_y_ratio)
        self.full_rail_min_height_ratio = float(full_rail_min_height_ratio)
        self.high_cell_width_cv_threshold = float(high_cell_width_cv_threshold)
        self.lower_band_many_cells_veto_min_count = int(lower_band_many_cells_veto_min_count)
        self.lower_band_many_cells_max_aspect = float(lower_band_many_cells_max_aspect)
        self.top_context_min_height_ratio = float(top_context_min_height_ratio)
        self.top_context_min_paper_ratio = float(top_context_min_paper_ratio)
        self.suppress_stable_repeated_counter_on_lower_band = bool(suppress_stable_repeated_counter_on_lower_band)
        self.debug = bool(debug)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> "InputStructureAnalyzer":
        cfg = dict(cfg or {})
        return cls(
            mode=str(cfg.get("mode", "auto") or "auto"),
            min_cells_for_rail=int(cfg.get("min_cells_for_rail", 2)),
            decision_threshold=float(cfg.get("decision_threshold", 0.58)),
            min_rail_width_ratio=float(cfg.get("min_rail_width_ratio", 0.42)),
            min_rail_area_ratio=float(cfg.get("min_rail_area_ratio", 0.045)),
            max_single_like_aspect=float(cfg.get("max_single_like_aspect", 2.35)),
            wide_crop_aspect=float(cfg.get("wide_crop_aspect", 2.65)),
            require_separator_or_wide=bool(cfg.get("require_separator_or_wide", True)),
            min_separator_count=int(cfg.get("min_separator_count", 1)),
            enable_single_tag_veto=bool(cfg.get("enable_single_tag_veto", True)),
            single_lower_band_y_ratio=float(cfg.get("single_lower_band_y_ratio", 0.24)),
            single_lower_band_max_height_ratio=float(cfg.get("single_lower_band_max_height_ratio", 0.58)),
            single_tag_veto_threshold=float(cfg.get("single_tag_veto_threshold", 0.62)),
            full_height_separator_min_count=int(cfg.get("full_height_separator_min_count", 1)),
            full_height_separator_min_strength=float(cfg.get("full_height_separator_min_strength", 0.035)),
            full_rail_top_y_ratio=float(cfg.get("full_rail_top_y_ratio", 0.20)),
            full_rail_min_height_ratio=float(cfg.get("full_rail_min_height_ratio", 0.52)),
            high_cell_width_cv_threshold=float(cfg.get("high_cell_width_cv_threshold", 0.34)),
            lower_band_many_cells_veto_min_count=int(cfg.get("lower_band_many_cells_veto_min_count", 4)),
            lower_band_many_cells_max_aspect=float(cfg.get("lower_band_many_cells_max_aspect", 2.10)),
            top_context_min_height_ratio=float(cfg.get("top_context_min_height_ratio", 0.30)),
            top_context_min_paper_ratio=float(cfg.get("top_context_min_paper_ratio", 0.42)),
            suppress_stable_repeated_counter_on_lower_band=bool(cfg.get("suppress_stable_repeated_counter_on_lower_band", True)),
            debug=bool(cfg.get("debug", False)),
        )

    def decide(self, image: np.ndarray, rail_split: Mapping[str, Any] | None = None) -> InputStructureDecision:
        """Return routing decision.
        Args:
            image: raw image before OCR.
            rail_split: optional result from PriceRailSplitter.split(image).  The
                analyzer will not run the splitter by itself; it only compares
                the available rail hypothesis with the single-tag hypothesis.
        """
        if image is None or image.size == 0:
          return InputStructureDecision(mode="single_tag", confidence=1.0, reason="empty_or_invalid_image", features={"image_valid": False},)

        h, w = image.shape[:2]
        img_area = float(max(1, h * w))
        aspect = float(w / max(1, h))
        split = dict(rail_split or {})
        rails = list(split.get("rails", []) or [])
        cells = list(split.get("cells", []) or [])

        best_rail, best_cells = self._best_rail_and_cells(rails, cells)
        best_rail_box = self._box_from_dict((best_rail or {}).get("bbox") if isinstance(best_rail, Mapping) else None)

        cell_count_total = len(cells)
        best_cell_count = len(best_cells)
        rail_count = len(rails)
        rail_width_ratio = 0.0
        rail_height_ratio = 0.0
        rail_area_ratio = 0.0
        rail_source = ""
        rail_score = 0.0
        rail_y1_ratio = 0.0
        rail_y2_ratio = 0.0

        if best_rail_box is not None:
            rail_width_ratio = float(best_rail_box.width / max(1, w))
            rail_height_ratio = float(best_rail_box.height / max(1, h))
            rail_area_ratio = float(best_rail_box.area / img_area)
            rail_y1_ratio = float(best_rail_box.y1 / max(1, h))
            rail_y2_ratio = float(best_rail_box.y2 / max(1, h))
            rail_source = str(((best_rail or {}).get("horizontal_cluster") or {}).get("source", "")) if isinstance(best_rail, Mapping) else ""
            rail_score = float((best_rail or {}).get("score", 0.0)) if isinstance(best_rail, Mapping) else 0.0

        separator_count, separator_sources, equal_width_fallback_count = self._separator_stats(best_cells)
        full_sep_count, full_sep_xs, full_sep_strength = self._full_height_separator_stats(image, best_rail_box)
        top_ctx = self._above_rail_context_stats(image, best_rail_box)
        cell_widths = [self._box_width(c.get("bbox")) for c in best_cells if isinstance(c, Mapping)]
        cell_width_cv = self._cv(cell_widths)
        median_cell_width_ratio = float(np.median(cell_widths) / max(1, w)) if cell_widths else 0.0
        progressive_cells = sum(1 for c in best_cells if "progressive" in str(c.get("tag_type", "")))
        ordinary_cells = sum(1 for c in best_cells if str(c.get("tag_type", "")) == "ordinary")
        strong_separator = separator_count >= self.min_separator_count
        wide_crop = aspect >= self.wide_crop_aspect
        physically_wide_rail = rail_width_ratio >= self.min_rail_width_ratio and rail_area_ratio >= self.min_rail_area_ratio

        features: Dict[str, Any] = {
            "image_shape": [int(h), int(w)],
            "image_aspect": aspect,
            "rail_count": rail_count,
            "cell_count_total": cell_count_total,
            "best_cell_count": best_cell_count,
            "rail_width_ratio": rail_width_ratio,
            "rail_height_ratio": rail_height_ratio,
            "rail_area_ratio": rail_area_ratio,
            "rail_y1_ratio": rail_y1_ratio,
            "rail_y2_ratio": rail_y2_ratio,
            "rail_source": rail_source,
            "rail_score": rail_score,
            "separator_count": separator_count,
            "separator_sources": separator_sources,
            "full_height_separator_count": full_sep_count,
            "full_height_separator_xs": full_sep_xs,
            "full_height_separator_strength": full_sep_strength,
            "top_context_height_ratio": top_ctx["height_ratio"],
            "top_context_paper_ratio": top_ctx["paper_ratio"],
            "top_context_dark_ratio": top_ctx["dark_ratio"],
            "top_context_saturated_ratio": top_ctx["saturated_ratio"],
            "equal_width_fallback_cell_count": equal_width_fallback_count,
            "cell_width_cv": cell_width_cv,
            "median_cell_width_ratio": median_cell_width_ratio,
            "progressive_cell_count": progressive_cells,
            "ordinary_cell_count": ordinary_cells,
            "strong_separator": strong_separator,
            "strong_full_height_separator": full_sep_count >= self.full_height_separator_min_count,
            "wide_crop": wide_crop,
            "physically_wide_rail": physically_wide_rail,
        }

        if self.mode in {"single", "single_tag", "tag"}:
          return InputStructureDecision("single_tag", 1.0, "forced_single_tag", features)

        if self.mode in {"rail", "price_rail", "shelf_rail"}:
          return InputStructureDecision("price_rail", 1.0, "forced_price_rail", features)

        if best_cell_count < self.min_cells_for_rail:
          return InputStructureDecision(mode="single_tag", confidence=0.92, reason="rail_hypothesis_has_too_few_cells", features=features | {"rail_likelihood": 0.0},)

        rail_likelihood = self._score_rail_likelihood(aspect=aspect, best_cell_count=best_cell_count, rail_width_ratio=rail_width_ratio,
            rail_height_ratio=rail_height_ratio, rail_area_ratio=rail_area_ratio, rail_source=rail_source, rail_score=rail_score, separator_count=separator_count,
            equal_width_fallback_count=equal_width_fallback_count, cell_width_cv=cell_width_cv, progressive_cells=progressive_cells,ordinary_cells=ordinary_cells,)

        # A wide progressive single tag often contains several visual price
        # columns, but it usually lacks full-height separator evidence.  Avoid
        # turning such tags into a false rail unless the crop is very wide.
        if self.require_separator_or_wide and not strong_separator and not wide_crop:
          rail_likelihood -= 0.22
          features["penalty_no_separator_and_not_wide"] = 0.22

        if aspect <= self.max_single_like_aspect and not strong_separator and equal_width_fallback_count >= best_cell_count:
          rail_likelihood -= 0.18
          features["penalty_single_like_aspect_equal_fallback"] = 0.18

        # A common false positive: one progressive price tag is split into two
        # vertical price columns.  That usually has only two "cells", low image
        # aspect, and each half occupies a very large fraction of the image width.
        if best_cell_count == 2 and aspect < 1.80:
            rail_likelihood -= 0.32
            features["penalty_two_cells_low_aspect"] = 0.32
        if median_cell_width_ratio > 0.38 and aspect < self.wide_crop_aspect:
            rail_likelihood -= 0.22
            features["penalty_cells_too_wide_for_rail"] = 0.22
        if progressive_cells == best_cell_count and best_cell_count <= 2 and aspect < self.wide_crop_aspect:
            rail_likelihood -= 0.16
            features["penalty_two_progressive_halves"] = 0.16

        lower_band_many_cells_hint = (
            rail_y1_ratio >= self.single_lower_band_y_ratio
            and rail_y2_ratio >= 0.76
            and best_cell_count >= self.lower_band_many_cells_veto_min_count
            and progressive_cells == best_cell_count
            and aspect < self.lower_band_many_cells_max_aspect
        )
        if lower_band_many_cells_hint:
            rail_likelihood -= 0.24
            features["penalty_lower_band_many_progressive_fragments"] = 0.24

        single_veto_score, single_veto_reasons = self._score_single_tag_veto(
            aspect=aspect,
            best_cell_count=best_cell_count,
            rail_y1_ratio=rail_y1_ratio,
            rail_y2_ratio=rail_y2_ratio,
            rail_height_ratio=rail_height_ratio,
            rail_source=rail_source,
            rail_score=rail_score,
            separator_count=separator_count,
            cell_width_cv=cell_width_cv,
            progressive_cells=progressive_cells,
            ordinary_cells=ordinary_cells,
            full_height_separator_count=full_sep_count,
            top_context_height_ratio=top_ctx["height_ratio"],
            top_context_paper_ratio=top_ctx["paper_ratio"],
            top_context_dark_ratio=top_ctx["dark_ratio"],
            top_context_saturated_ratio=top_ctx["saturated_ratio"],
        )
        features["single_tag_veto_score"] = single_veto_score
        features["single_tag_veto_reasons"] = single_veto_reasons
        if self.enable_single_tag_veto and single_veto_score >= self.single_tag_veto_threshold:
            features["rail_likelihood_before_single_tag_veto"] = float(max(0.0, min(1.0, rail_likelihood)))
            features["rail_likelihood"] = 0.0
            features["decision_threshold"] = self.decision_threshold
            return InputStructureDecision(
                mode="single_tag",
                confidence=float(min(0.99, 0.55 + 0.45 * single_veto_score)),
                reason="single_physical_tag_veto_over_rail_split",
                features=features,
            )

        rail_likelihood = float(max(0.0, min(1.0, rail_likelihood)))
        features["rail_likelihood"] = rail_likelihood
        features["decision_threshold"] = self.decision_threshold

        if rail_likelihood >= self.decision_threshold:
            return InputStructureDecision(
                mode="price_rail",
                confidence=rail_likelihood,
                reason="multiple_price_cells_supported_by_rail_geometry",
                features=features,
            )

        return InputStructureDecision(
            mode="single_tag",
            confidence=1.0 - rail_likelihood,
            reason="single_tag_hypothesis_preferred_over_weak_rail_split",
            features=features,
        )

    def _score_single_tag_veto(
        self,
        *,
        aspect: float,
        best_cell_count: int,
        rail_y1_ratio: float,
        rail_y2_ratio: float,
        rail_height_ratio: float,
        rail_source: str,
        rail_score: float,
        separator_count: int,
        cell_width_cv: float,
        progressive_cells: int,
        ordinary_cells: int,
        full_height_separator_count: int,
        top_context_height_ratio: float,
        top_context_paper_ratio: float,
        top_context_dark_ratio: float,
        top_context_saturated_ratio: float,
    ) -> Tuple[float, List[str]]:
        """Score whether a rail split is probably just fields inside one tag.
        This is the guard for large single promotional shelf tags.  Such tags
        contain several price blocks/columns, so a pure vertical splitter often
        creates 2-6 false "cells".  A true rail should have either a full-tag
        band with physical full-height separators, or a strong repeated-cell
        geometry.  If the detected band is mostly the lower red price area, or
        if the image is not very wide and has no full-height separators, we keep
        the whole crop as one price tag.
        """
        if best_cell_count <= 0:
            return 0.0, []

        reasons: List[str] = []
        score = 0.0
        all_progressive = progressive_cells == best_cell_count and ordinary_cells == 0
        full_sep = full_height_separator_count >= self.full_height_separator_min_count

        lower_band_split = (rail_y1_ratio >= self.single_lower_band_y_ratio and rail_y2_ratio >= 0.78
            and rail_height_ratio <= self.single_lower_band_max_height_ratio)

        full_rail_geometry = (rail_y1_ratio <= self.full_rail_top_y_ratio and rail_height_ratio >= self.full_rail_min_height_ratio)

        if lower_band_split:
          score += 0.46
          reasons.append("rail_is_lower_price_band_not_full_tag")

        if all_progressive and best_cell_count >= 2:
          score += 0.16
          reasons.append("all_cells_are_progressive_fragments")

        if not full_sep:
          score += 0.16
          reasons.append("no_full_height_physical_separators")

        if aspect < self.wide_crop_aspect:
          score += 0.10
          reasons.append("crop_not_wide_enough_for_reliable_multi_tag_rail")

        if cell_width_cv >= self.high_cell_width_cv_threshold:
          score += 0.13
          reasons.append("cell_widths_are_unstable")

        if best_cell_count >= 5 and not full_sep:
          score += 0.10
          reasons.append("many_cells_without_physical_separators")

        if lower_band_split and all_progressive and best_cell_count >= self.lower_band_many_cells_veto_min_count and aspect < self.lower_band_many_cells_max_aspect:
          score += 0.28
          reasons.append("lower_price_band_split_into_many_internal_columns")

        if (lower_band_split and top_context_height_ratio >= self.top_context_min_height_ratio
            and top_context_paper_ratio >= self.top_context_min_paper_ratio
            and top_context_saturated_ratio < 0.55):
          score += 0.24
          reasons.append("large_paper_tag_body_above_detected_price_band")

        if (not full_sep) and all_progressive and best_cell_count <= 3 and aspect < 1.42:
          score += 0.36
          reasons.append("compact_single_tag_aspect_with_internal_price_columns")

        if "full_tag_expand" not in rail_source and lower_band_split:
          score += 0.08
          reasons.append("rail_source_did_not_expand_to_full_tag")

        # Strong counter-evidence: a full tag-height rail with physical vertical
        # boundaries is exactly what we expect for a real shelf rail.
        if full_sep and full_rail_geometry:
          score -= 0.38
          reasons.append("counterevidence_full_height_separators_in_full_rail")

        elif full_sep:
           score -= 0.22
           reasons.append("counterevidence_full_height_separators")

        if full_rail_geometry and not lower_band_split:
          score -= 0.12
          reasons.append("counterevidence_full_rail_geometry")

        # Repeated, stable, high-score cells with several separator cues are a
        # real rail pattern even when the horizontal crop starts at the red
        # price band. This protects wide shelf photos where the upper white
        # part is partially outside the crop/search window.
        stable_repeated_rail = (best_cell_count >= 4 and cell_width_cv <= 0.22 and separator_count >= 3 and rail_score >= 0.65)

        stable_repeated_is_reliable_rail = ( stable_repeated_rail
            and (aspect >= self.wide_crop_aspect
                or (full_sep and full_rail_geometry)
                or (full_rail_geometry and "full_tag_expand" in rail_source)
                or not self.suppress_stable_repeated_counter_on_lower_band)
        )

        if stable_repeated_is_reliable_rail:
            score -= 0.55
            reasons.append("counterevidence_stable_repeated_rail_cells")

        elif stable_repeated_rail and lower_band_split and aspect < self.lower_band_many_cells_max_aspect:
            score += 0.10
            reasons.append("stable_repeated_cells_are_only_lower_band_not_physical_rail")

        if "full_tag_expand" in rail_source and best_cell_count >= 4 and cell_width_cv <= 0.24:
            score -= 0.22
            reasons.append("counterevidence_splitter_full_tag_expand")

        return float(max(0.0, min(1.0, score))), reasons

    def _above_rail_context_stats(self, image: np.ndarray, rail_box: Optional[Box]) -> Dict[str, float]:

        """Summarize the area above the detected rail band.
        When a single physical shelf tag is split incorrectly, the splitter
        usually finds only the lower red/orange price band.  The area above that
        band is still the same paper tag body: mostly white/yellow, with product
        text, barcode and small fields.  In a real shelf photo the area above a
        lower rail candidate often contains cans/bottles/background instead.
        This OCR-free statistic is used only as additional evidence in the
        single-tag veto; it is not a standalone decision rule.
        """

        if image is None or image.size == 0 or rail_box is None:
          return {"height_ratio": 0.0, "paper_ratio": 0.0, "dark_ratio": 0.0, "saturated_ratio": 0.0}
        h, w = image.shape[:2]
        y2 = max(0, min(h, int(rail_box.y1)))

        if y2 < max(18, int(0.08 * h)) or w < 32:
           return {"height_ratio": float(y2 / max(1, h)), "paper_ratio": 0.0, "dark_ratio": 0.0, "saturated_ratio": 0.0}

        crop = image[:y2, :]
        # Ignore a thin outer border, which often contains shelf/camera padding
        # after tilt rotation.
        margin_x = max(0, int(0.025 * w))
        margin_y = max(0, int(0.025 * h))

        if crop.shape[0] > 2 * margin_y + 8 and crop.shape[1] > 2 * margin_x + 8:
            crop = crop[margin_y:crop.shape[0] - margin_y, margin_x:crop.shape[1] - margin_x]

        if crop.size == 0:
            return {"height_ratio": float(y2 / max(1, h)), "paper_ratio": 0.0, "dark_ratio": 0.0, "saturated_ratio": 0.0}

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV) if crop.ndim == 3 else cv2.cvtColor(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
        hch = hsv[:, :, 0].astype(np.int16)
        sat = hsv[:, :, 1].astype(np.float32)
        val = hsv[:, :, 2].astype(np.float32)

        # White/gray paper, slightly yellow paper and yellow promo/header areas.
        white_paper = (val >= 118) & (sat <= 95)
        yellow_paper = (val >= 105) & (sat >= 45) & (sat <= 210) & (hch >= 16) & (hch <= 45)
        light_gray_paper = (val >= 96) & (sat <= 55)
        paper = white_paper | yellow_paper | light_gray_paper
        dark = val <= 80
        saturated = (sat >= 130) & (val >= 80)

        return {"height_ratio": float(y2 / max(1, h)), "paper_ratio": float(np.mean(paper)), "dark_ratio": float(np.mean(dark)), "saturated_ratio": float(np.mean(saturated)),}

    def _full_height_separator_stats(self, image: np.ndarray, rail_box: Optional[Box]) -> Tuple[int, List[int], float]:
        """Detect physical vertical separators that cross most of the rail height.

        Separator evidence used by the splitter can be contaminated by digits,
        barcode strokes and field dividers.  For the single-vs-rail router we
        need a stricter cue: a vertical boundary must be visible both in the
        upper part and in the lower part of the detected rail band.
        """
        if image is None or image.size == 0 or rail_box is None:
            return 0, [], 0.0
        h, w = image.shape[:2]
        x1 = max(0, int(rail_box.x1))
        x2 = min(w, int(rail_box.x2))
        y1 = max(0, int(rail_box.y1))
        y2 = min(h, int(rail_box.y2))

        if x2 - x1 < 64 or y2 - y1 < 48:
            return 0, [], 0.0

        crop = image[y1:y2, x1:x2]
        rh, rw = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 45, 140).astype(np.float32) / 255.0

        # Emphasize long vertical strokes and suppress isolated text strokes.
        k_h = max(9, int(round(rh * 0.34)))
        vert = cv2.morphologyEx((edges * 255).astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_h)), iterations=1,).astype(np.float32) / 255.0

        upper = vert[int(0.08 * rh):int(0.45 * rh), :]
        lower = vert[int(0.55 * rh):int(0.94 * rh), :]
        mid = vert[int(0.28 * rh):int(0.76 * rh), :]

        if upper.size == 0 or lower.size == 0 or mid.size == 0:
            return 0, [], 0.0

        upper_col = upper.mean(axis=0)
        lower_col = lower.mean(axis=0)
        mid_col = mid.mean(axis=0)
        continuity = np.minimum(upper_col, lower_col) + 0.35 * mid_col
        continuity = self._smooth_1d(continuity, k=max(9, int(rw * 0.018)))

        if continuity.size == 0:
            return 0, [], 0.0

        thr = max(self.full_height_separator_min_strength, float(np.percentile(continuity, 96)), float(continuity.mean() + 2.1 * continuity.std()),)

        min_dist = max(28, int(rw * 0.055))
        candidates: List[Tuple[int, float]] = []
        for i in range(3, rw - 3):
            if i < rw * 0.04 or i > rw * 0.96:
                continue
            val = float(continuity[i])
            if val < thr:
                continue
            lo = max(0, i - 3)
            hi = min(rw, i + 4)
            if val >= float(np.max(continuity[lo:hi])):
                candidates.append((i, val))
        candidates.sort(key=lambda t: t[1], reverse=True)
        selected: List[Tuple[int, float]] = []
        for x, val in candidates:
            if all(abs(x - sx) >= min_dist for sx, _ in selected):
                selected.append((x, val))
        selected.sort(key=lambda t: t[0])
        xs = [int(x1 + x) for x, _ in selected]
        strength = float(max([v for _, v in selected], default=0.0))
        return len(xs), xs, strength

    @staticmethod
    def _smooth_1d(x: np.ndarray, k: int = 15) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x
        k = int(max(1, k))
        if k % 2 == 0:
            k += 1
        if x.size < k:
            k = int(x.size) if int(x.size) % 2 == 1 else max(1, int(x.size) - 1)
        if k <= 1:
            return x.astype(np.float32)
        return cv2.GaussianBlur(x.reshape(1, -1), (k, 1), 0).reshape(-1)

    def _score_rail_likelihood(
        self,
        *,
        aspect: float,
        best_cell_count: int,
        rail_width_ratio: float,
        rail_height_ratio: float,
        rail_area_ratio: float,
        rail_source: str,
        rail_score: float,
        separator_count: int,
        equal_width_fallback_count: int,
        cell_width_cv: float,
        progressive_cells: int,
        ordinary_cells: int,
    ) -> float:

        score = 0.0
        score += min(0.28, 0.10 + 0.06 * max(0, best_cell_count - 1))

        if rail_width_ratio >= self.min_rail_width_ratio:
            score += 0.16

        if rail_area_ratio >= self.min_rail_area_ratio:
            score += 0.08

        if 0.05 <= rail_height_ratio <= 0.38:
            score += 0.07

        if "hough" in rail_source:
            score += 0.12

        elif "row_projection" in rail_source:
            score += 0.07

        if separator_count >= self.min_separator_count:
            score += min(0.22, 0.12 + 0.04 * separator_count)

        elif aspect >= self.wide_crop_aspect:
            score += 0.10

        if cell_width_cv <= 0.55:
            score += 0.07

        if equal_width_fallback_count >= best_cell_count and aspect < self.wide_crop_aspect:
            score -= 0.12

        if progressive_cells >= 1 and ordinary_cells >= 1:
            # Mixed ordinary/progressive cells are common in real shelf rails.
            score += 0.06

        if rail_score >= 0.65:
            score += 0.05

        return score

    @staticmethod
    def _best_rail_and_cells(rails: Sequence[Any], cells: Sequence[Any]) -> Tuple[Optional[Mapping[str, Any]], List[Mapping[str, Any]]]:

        if not rails:
            return None, []

        by_rail: Dict[int, List[Mapping[str, Any]]] = {}
        for c in cells:
            if not isinstance(c, Mapping):
                continue
            ri = int(c.get("rail_index", 0))
            by_rail.setdefault(ri, []).append(c)

        best: Optional[Mapping[str, Any]] = None
        best_cells: List[Mapping[str, Any]] = []
        best_key = (-1, -1.0)

        for r in rails:
            if not isinstance(r, Mapping):
                continue
            ri = int(r.get("rail_index", 0))
            rcells = by_rail.get(ri, [])
            key = (len(rcells), float(r.get("score", 0.0)))
            if key > best_key:
                best = r
                best_cells = rcells
                best_key = key
        return best, best_cells

    @staticmethod
    def _box_from_dict(d: Any) -> Optional[Box]:
        if not isinstance(d, Mapping):
            return None
        try:
            return Box(
                cls=str(d.get("cls", "box")),
                x1=int(d.get("x1", 0)),
                y1=int(d.get("y1", 0)),
                x2=int(d.get("x2", 0)),
                y2=int(d.get("y2", 0)),
                conf=float(d.get("conf", 0.0)),
                source=str(d.get("source", "unknown")),
            )
        except Exception:
            return None

    @staticmethod
    def _box_width(d: Any) -> int:
        b = InputStructureAnalyzer._box_from_dict(d)
        return int(b.width) if b is not None else 0

    @staticmethod
    def _separator_stats(cells: Sequence[Mapping[str, Any]]) -> Tuple[int, List[str], int]:
        max_sep = 0
        sources: List[str] = []
        equal_width = 0
        for c in cells:
            feats = c.get("features", {}) if isinstance(c, Mapping) else {}
            if not isinstance(feats, Mapping):
                continue
            sep_count = int(feats.get("rail_separator_count", 0) or 0)
            max_sep = max(max_sep, sep_count)
            srcs = feats.get("rail_separator_sources", []) or []
            if isinstance(srcs, str):
                srcs = [srcs]
            for s in srcs:
                ss = str(s)
                if ss and ss not in sources:
                    sources.append(ss)
            if str(feats.get("rail_split_method", "")) == "equal_width_fallback":
                equal_width += 1
        return max_sep, sources, equal_width

    @staticmethod
    def _cv(values: Sequence[int | float]) -> float:
        vals = np.asarray([float(v) for v in values if float(v) > 0.0], dtype=np.float32)
        if vals.size <= 1:
            return 0.0
        return float(np.std(vals) / (np.mean(vals) + 1e-6))

def draw_structure_decision_debug(image: np.ndarray, decision: InputStructureDecision) -> np.ndarray:
    """Draw only the routing decision; rail boxes are drawn by PriceRailSplitter."""
    out = image.copy()
    label = f"input={decision.mode} conf={decision.confidence:.2f} {decision.reason}"
    color = (0, 255, 0) if decision.mode == "single_tag" else (0, 180, 255)
    draw_label(out, label, 5, 24, color)
    return out
