"""
Shelf price-rail segmentation and price-cell clustering.

The splitter is a detector-free preprocessing block for shelf photographs where a
single horizontal shelf rail contains several paper price tags.  It uses:

1. horizontal clustering to find long rail bands;
2. vertical clustering to split each rail band into individual price cells;
3. lightweight geometry/color statistics to classify cells as ordinary or
   progressive/multi-price tags.

The goal is not to replace a trained detector.  It provides a deterministic
fallback/bootstrapping stage for OCR experiments and dataset generation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .debug_vis import draw_label
from .image_ops import crop_box
from .pipeline_types import Box


@dataclass
class Cluster1D:
    start: int
    end: int
    score: float
    axis: str
    source: str

    @property
    def center(self) -> float:
        return (self.start + self.end) * 0.5

    @property
    def size(self) -> int:
        return max(0, self.end - self.start)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RailCell:
    rail_index: int
    cell_index: int
    bbox: Box
    local_bbox: Box
    tag_type: str
    score: float
    features: Dict[str, Any]
    horizontal_clusters: List[Cluster1D]
    vertical_clusters: List[Cluster1D]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rail_index": self.rail_index,
            "cell_index": self.cell_index,
            "bbox": self.bbox.to_dict(),
            "local_bbox": self.local_bbox.to_dict(),
            "tag_type": self.tag_type,
            "score": float(self.score),
            "features": self.features,
            "horizontal_clusters": [c.to_dict() for c in self.horizontal_clusters],
            "vertical_clusters": [c.to_dict() for c in self.vertical_clusters],
        }


@dataclass
class RailSegment:
    rail_index: int
    bbox: Box
    score: float
    horizontal_cluster: Cluster1D
    cells: List[RailCell]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rail_index": self.rail_index,
            "bbox": self.bbox.to_dict(),
            "score": float(self.score),
            "horizontal_cluster": self.horizontal_cluster.to_dict(),
            "cells": [c.to_dict() for c in self.cells],
        }




@dataclass
class BeamPath:
    template_name: str
    boundaries: List[int]
    cell_scores: List[float]
    total_score: float
    normalized_score: float
    features: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_name": self.template_name,
            "boundaries": [int(x) for x in self.boundaries],
            "cell_scores": [float(x) for x in self.cell_scores],
            "total_score": float(self.total_score),
            "normalized_score": float(self.normalized_score),
            "features": self.features,
        }


class PriceRailSplitter:
    def __init__(
        self,
        search_y_min_ratio: float = 0.25,
        search_y_max_ratio: float = 0.98,
        min_rail_width_ratio: float = 0.45,
        min_rail_height_ratio: float = 0.055,
        max_rail_height_ratio: float = 0.72,
        rail_expand_y_ratio: float = 0.020,
        min_cell_width: int = 80,
        max_cell_width: int = 520,
        min_cells: int = 2,
        max_rails: int = 3,
        cell_expand_px: int = 4,
        split_strategy: str = "beam",
        beam_size: int = 12,
        beam_boundary_step_px: int = 48,
        beam_min_cell_width_ratio: float = 0.11,
        beam_max_cell_width_ratio: float = 0.42,
        beam_min_path_score: float = 0.45,
        same_template_beam: bool = True,
        debug: bool = False,
    ) -> None:
        self.search_y_min_ratio = float(search_y_min_ratio)
        self.search_y_max_ratio = float(search_y_max_ratio)
        self.min_rail_width_ratio = float(min_rail_width_ratio)
        self.min_rail_height_ratio = float(min_rail_height_ratio)
        self.max_rail_height_ratio = float(max_rail_height_ratio)
        self.rail_expand_y_ratio = float(rail_expand_y_ratio)
        self.min_cell_width = int(min_cell_width)
        self.max_cell_width = int(max_cell_width)
        self.min_cells = int(min_cells)
        self.max_rails = int(max_rails)
        self.cell_expand_px = int(cell_expand_px)
        self.split_strategy = str(split_strategy or "beam").lower().strip()
        self.beam_size = int(beam_size)
        self.beam_boundary_step_px = int(beam_boundary_step_px)
        self.beam_min_cell_width_ratio = float(beam_min_cell_width_ratio)
        self.beam_max_cell_width_ratio = float(beam_max_cell_width_ratio)
        self.beam_min_path_score = float(beam_min_path_score)
        self.same_template_beam = bool(same_template_beam)
        self.debug = bool(debug)
        self.last_debug: Dict[str, Any] = {}

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "PriceRailSplitter":
        return cls(
            search_y_min_ratio=float(cfg.get("search_y_min_ratio", 0.25)),
            search_y_max_ratio=float(cfg.get("search_y_max_ratio", 0.98)),
            min_rail_width_ratio=float(cfg.get("min_rail_width_ratio", 0.45)),
            min_rail_height_ratio=float(cfg.get("min_rail_height_ratio", 0.055)),
            max_rail_height_ratio=float(cfg.get("max_rail_height_ratio", 0.72)),
            rail_expand_y_ratio=float(cfg.get("rail_expand_y_ratio", 0.020)),
            min_cell_width=int(cfg.get("min_cell_width", 80)),
            max_cell_width=int(cfg.get("max_cell_width", 520)),
            min_cells=int(cfg.get("min_cells", 2)),
            max_rails=int(cfg.get("max_rails", 3)),
            cell_expand_px=int(cfg.get("cell_expand_px", 4)),
            split_strategy=str(cfg.get("split_strategy", "beam")),
            beam_size=int(cfg.get("beam_size", 12)),
            beam_boundary_step_px=int(cfg.get("beam_boundary_step_px", 48)),
            beam_min_cell_width_ratio=float(cfg.get("beam_min_cell_width_ratio", 0.11)),
            beam_max_cell_width_ratio=float(cfg.get("beam_max_cell_width_ratio", 0.42)),
            beam_min_path_score=float(cfg.get("beam_min_path_score", 0.45)),
            same_template_beam=bool(cfg.get("same_template_beam", True)),
            debug=bool(cfg.get("debug", False)),
        )

    def split(self, image: np.ndarray) -> Dict[str, Any]:
        self.last_debug = {"enabled": True, "rails": 0, "cells": 0}
        if image is None or image.size == 0:
            self.last_debug["reason"] = "empty_image"
            return {"rails": [], "cells": [], "debug_boxes": []}

        h, w = image.shape[:2]
        rails: List[RailSegment] = []
        rail_clusters = self._find_horizontal_rail_clusters(image)
        for rail_idx, cl in enumerate(rail_clusters):
            y1 = max(0, cl.start - int(h * self.rail_expand_y_ratio))
            y2 = min(h, cl.end + int(h * self.rail_expand_y_ratio))
            rb = Box("price_rail", 0, y1, w, y2, cl.score, "horizontal_cluster")
            rail_crop = crop_box(image, rb)
            cells = self._split_rail_into_cells(rail_crop, rb, rail_idx=rail_idx)
            rails.append(RailSegment(rail_index=rail_idx, bbox=rb, score=cl.score, horizontal_cluster=cl, cells=cells))

        all_cells = [c for r in rails for c in r.cells]
        debug_boxes = [r.bbox for r in rails] + [c.bbox for c in all_cells]
        self.last_debug.update({"rails": len(rails), "cells": len(all_cells)})
        return {
            "rails": [r.to_dict() for r in rails],
            "cells": [c.to_dict() for c in all_cells],
            "debug_boxes": [b.to_dict() for b in debug_boxes],
            "_rail_objects": rails,
            "_cell_objects": all_cells,
        }

    # ------------------------------------------------------------------
    # Horizontal rail clustering
    # ------------------------------------------------------------------

    def _find_horizontal_rail_clusters(self, image: np.ndarray) -> List[Cluster1D]:
        h, w = image.shape[:2]
        min_h = max(18, int(h * self.min_rail_height_ratio))
        max_h = max(min_h + 1, int(h * self.max_rail_height_ratio))
        search_y1 = int(np.clip(h * self.search_y_min_ratio, 0, h - 1))
        search_y2 = int(np.clip(h * self.search_y_max_ratio, search_y1 + 1, h))

        horizontal_lines = self._detect_long_horizontal_lines(image, y1=search_y1, y2=search_y2)
        clusters: List[Cluster1D] = []

        # Strong prior: the upper edge of a shelf rail is often a long horizontal
        # plastic/shelf boundary.  Estimate the rail height below that line.
        for y, score in horizontal_lines:
            bottom = self._estimate_rail_bottom_from_top(image, y, max_h=max_h, min_h=min_h)
            if bottom - y < min_h:
                continue
            clusters.append(Cluster1D(start=int(y), end=int(bottom), score=float(score), axis="y", source="hough_horizontal_top"))

        # Projection fallback: catches already-cropped rails or shelves without a
        # long detected boundary line.
        row_score = self._row_activity_score(image)
        local = row_score[search_y1:search_y2]
        if local.size:
            thr = max(float(np.percentile(local, 78)), float(local.mean() + 0.20 * local.std()))
            mask = row_score >= thr
            for a, b, score in self._runs(mask, row_score, min_size=max(8, min_h // 2)):
                if b < search_y1 or a > search_y2:
                    continue
                a = max(search_y1, a)
                b = min(search_y2, b)
                # Merge close text/color rows into one rail band.
                a = max(0, a - int(0.03 * h))
                b = min(h, b + int(0.09 * h))
                if b - a < min_h:
                    continue
                if b - a > max_h:
                    # Keep the strongest sub-window of max_h height.
                    best_a = self._best_window_start(row_score, a, b, max_h)
                    a, b = best_a, min(h, best_a + max_h)
                clusters.append(Cluster1D(start=int(a), end=int(b), score=float(score), axis="y", source="row_projection"))

        if not clusters:
            # If the input already looks like one wide rail/crop, use it as a rail.
            aspect = w / max(1, h)
            if aspect >= 1.8:
                clusters.append(Cluster1D(start=0, end=h, score=0.35, axis="y", source="whole_image_wide_fallback"))

        clusters = [self._expand_rail_cluster_to_full_tag(image, c, min_h=min_h, max_h=max_h) for c in clusters]
        clusters = self._dedupe_y_clusters(clusters, image_h=h)
        clusters = [c for c in clusters if (c.end - c.start) >= min_h]
        clusters = sorted(clusters, key=lambda c: (c.score, min(c.size, int(h * 0.22))), reverse=True)[:max(1, self.max_rails)]
        clusters = sorted(clusters, key=lambda c: c.start)
        if self.debug:
            self.last_debug["horizontal_clusters"] = [c.to_dict() for c in clusters]
        return clusters

    def _expand_rail_cluster_to_full_tag(self, image: np.ndarray, cl: Cluster1D, min_h: int, max_h: int) -> Cluster1D:
        """Expand a rail band when the detected horizontal line is an internal color boundary.

        In Lenta shelf labels the strongest horizontal line is often not the top
        of the whole tag, but the boundary between the upper white product block
        and the lower red/orange progressive-price block.  If we crop from that
        boundary, downstream vertical splitting starts grouping lower price rows
        instead of whole price tags.  This method expands such bands upward to
        include the white/yellow header and product-name area.
        """
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return cl

        masks = _color_masks(image)
        warm = _smooth_1d(masks["warm"].astype(np.float32).mean(axis=1), k=15)
        yellow = _smooth_1d(masks["yellow"].astype(np.float32).mean(axis=1), k=15)
        white = _smooth_1d(masks["white"].astype(np.float32).mean(axis=1), k=15)
        dark = _smooth_1d(masks["dark"].astype(np.float32).mean(axis=1), k=15)

        start = int(np.clip(cl.start, 0, h - 1))
        end = int(np.clip(cl.end, start + 1, h))
        wh = max(4, int(0.025 * h))
        below = slice(start, min(h, start + 3 * wh))
        above = slice(max(0, start - 3 * wh), start)
        warm_below = float(np.mean(warm[below])) if warm[below].size else 0.0
        warm_above = float(np.mean(warm[above])) if warm[above].size else 0.0
        white_above = float(np.mean(white[above])) if white[above].size else 0.0

        # Only expand aggressively when the current top looks like the red-block
        # boundary: warm/red below, white paper above.
        looks_internal_red_boundary = warm_below > 0.28 and (white_above > 0.35 or warm_above < warm_below * 0.75)
        if not looks_internal_red_boundary:
            return cl

        # Paper/header score.  We require low/moderate darkness to avoid
        # expanding into bottles or black bottom clutter.  Yellow top stripes are
        # explicitly supported.
        paper = (0.78 * white + 0.72 * yellow + 0.08 * dark).astype(np.float32)
        search_lo = max(0, start - max_h)
        search_hi = start
        if search_hi <= search_lo + min_h // 3:
            return cl
        high = (paper > 0.48) & (dark < 0.62)
        runs = _runs(high[search_lo:search_hi], paper[search_lo:search_hi], min_size=max(6, int(0.035 * h)))
        if not runs:
            return cl

        # Choose the last paper run that reaches close to the detected internal
        # boundary.  In practice this is the upper white/yellow part of the same
        # tag, not the cans above the rail.
        best_top: Optional[int] = None
        best_score = -1.0
        for a0, b0, sc in runs:
            a = search_lo + a0
            b = search_lo + b0
            if b < start - max(8, int(0.12 * (end - start))):
                continue
            span = b - a
            score = float(sc + 0.001 * span - 0.0005 * max(0, start - b))
            if score > best_score:
                best_score = score
                best_top = int(a)
        if best_top is None:
            return cl

        # Keep the lower red/orange block but trim bottle caps / shelf clutter
        # below it.  The red block is the longest high-warm run after the
        # internal boundary.
        bottom = end
        red_runs = _runs((warm > 0.55) & (np.arange(h) >= max(0, start - wh)), warm, min_size=max(8, int(0.05 * h)))
        red_runs = [(a, b, sc) for a, b, sc in red_runs if b > start + max(8, int(0.05 * h))]
        if red_runs:
            a, b, _ = max(red_runs, key=lambda r: (r[1] - r[0], r[2]))
            bottom = min(end, int(b + max(4, int(0.018 * h))))
        if bottom - best_top < min_h:
            return cl

        return Cluster1D(
            start=int(best_top),
            end=int(bottom),
            score=float(cl.score + 0.08),
            axis=cl.axis,
            source=str(cl.source) + "+full_tag_expand",
        )

    def _detect_long_horizontal_lines(self, image: np.ndarray, y1: int, y2: int) -> List[Tuple[int, float]]:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
        roi = edges[y1:y2, :]
        if roi.size == 0:
            return []
        hough_min_line_ratio = min(0.25, max(0.18, self.min_rail_width_ratio * 0.55))
        lines = cv2.HoughLinesP(
            roi,
            1,
            np.pi / 180.0,
            threshold=max(40, int(w * 0.07)),
            minLineLength=max(40, int(w * hough_min_line_ratio)),
            maxLineGap=max(12, int(w * 0.055)),
        )
        if lines is None:
            return []
        ys: List[Tuple[float, float]] = []
        max_slope_px = max(3, int(0.012 * w))
        for x1, yy1, x2, yy2 in lines[:, 0]:
            if abs(int(yy2) - int(yy1)) > max_slope_px:
                continue
            length = abs(int(x2) - int(x1))
            if length < w * hough_min_line_ratio:
                continue
            y = y1 + (float(yy1) + float(yy2)) * 0.5
            ys.append((y, length / max(1.0, float(w))))
        if not ys:
            return []
        ys = sorted(ys, key=lambda t: t[0])
        clusters: List[List[Tuple[float, float]]] = []
        for y, s in ys:
            if not clusters or abs(y - np.mean([v[0] for v in clusters[-1]])) > 10:
                clusters.append([(y, s)])
            else:
                clusters[-1].append((y, s))
        out: List[Tuple[int, float]] = []
        for c in clusters:
            y = int(round(np.mean([v[0] for v in c])))
            score = float(max(v[1] for v in c) + 0.05 * len(c))
            out.append((y, score))
        return out

    def _estimate_rail_bottom_from_top(self, image: np.ndarray, top_y: int, max_h: int, min_h: int) -> int:
        h = image.shape[0]
        row_score = self._row_activity_score(image)
        lo = min(h, top_y + min_h)
        hi = min(h, top_y + max_h)
        if hi <= lo:
            return min(h, top_y + min_h)
        # Prefer a local activity valley after the minimal plausible tag height.
        segment = row_score[lo:hi]
        if segment.size < 3:
            return hi
        smooth = _smooth_1d(segment, k=21)
        valley_idx = int(np.argmin(smooth[int(0.25 * len(smooth)):])) + int(0.25 * len(smooth))
        valley_y = lo + valley_idx
        # But do not cut early if lower warm/promo areas are still active.
        if valley_y - top_y >= min_h and float(smooth[valley_idx]) < float(np.percentile(row_score, 45)):
            return int(valley_y)
        return int(hi)

    def _row_activity_score(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150).astype(np.float32) / 255.0
        masks = _color_masks(image)
        warm = masks["warm"].astype(np.float32)
        yellow = masks["yellow"].astype(np.float32)
        white = masks["white"].astype(np.float32)
        dark = masks["dark"].astype(np.float32)
        score = (
            0.60 * edges.mean(axis=1)
            + 0.45 * warm.mean(axis=1)
            + 0.25 * yellow.mean(axis=1)
            + 0.15 * white.mean(axis=1)
            + 0.18 * dark.mean(axis=1)
        ).astype(np.float32)
        return _smooth_1d(score, k=31)

    @staticmethod
    def _best_window_start(score: np.ndarray, a: int, b: int, size: int) -> int:
        if b - a <= size:
            return a
        s = score[a:b].astype(np.float32)
        kernel = np.ones(size, dtype=np.float32)
        conv = np.convolve(s, kernel, mode="valid")
        idx = int(np.argmax(conv)) if conv.size else 0
        return a + idx

    def _dedupe_y_clusters(self, clusters: List[Cluster1D], image_h: int) -> List[Cluster1D]:
        out: List[Cluster1D] = []
        for c in sorted(clusters, key=lambda x: x.score, reverse=True):
            duplicate = False
            for i, o in enumerate(out):
                ov = max(0, min(c.end, o.end) - max(c.start, o.start))
                den = max(1, min(c.size, o.size))
                if ov / den > 0.45 or abs(c.center - o.center) < image_h * 0.06:
                    # Merge, preserve the better source/score.
                    out[i] = Cluster1D(
                        start=min(c.start, o.start),
                        end=max(c.end, o.end),
                        score=max(c.score, o.score),
                        axis="y",
                        source=o.source if o.score >= c.score else c.source,
                    )
                    duplicate = True
                    break
            if not duplicate:
                out.append(c)
        return out

    # ------------------------------------------------------------------
    # Vertical cell clustering
    # ------------------------------------------------------------------

    def _split_rail_into_cells(self, rail_crop: np.ndarray, rail_box: Box, rail_idx: int) -> List[RailCell]:
        if self.split_strategy in {"beam", "same_template_beam", "template_beam"}:
            cells = self._split_rail_into_cells_beam(rail_crop, rail_box, rail_idx)
            if len(cells) >= self.min_cells:
                return cells
            # Deterministic fallback preserves the older behavior when the beam
            # cannot build a coherent same-template path.
        return self._split_rail_into_cells_projection(rail_crop, rail_box, rail_idx)

    def _split_rail_into_cells_projection(self, rail_crop: np.ndarray, rail_box: Box, rail_idx: int) -> List[RailCell]:
        rh, rw = rail_crop.shape[:2]
        if rh < 16 or rw < self.min_cell_width:
            return []

        separators = self._find_vertical_separators(rail_crop)
        raw_boundaries = self._boundaries_from_separators(separators, rw)
        boundaries = self._regularize_boundaries(raw_boundaries, rw)
        if not separators and len(boundaries) > 2:
            split_method = "equal_width_fallback"
        elif len(boundaries) > len(raw_boundaries):
            split_method = "regularized_overwide_cell"
        else:
            split_method = "separator_projection"
        separator_sources = sorted({str(s.source) for s in separators})

        cells: List[RailCell] = []
        for i, (x1, x2) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if x2 - x1 < self.min_cell_width:
                continue
            x1e = max(0, x1 - self.cell_expand_px)
            x2e = min(rw, x2 + self.cell_expand_px)
            local = Box("price_cell", x1e, 0, x2e, rh, 0.65, "vertical_cluster")
            global_box = Box(
                "price_cell",
                rail_box.x1 + local.x1,
                rail_box.y1 + local.y1,
                rail_box.x1 + local.x2,
                rail_box.y1 + local.y2,
                0.65,
                "vertical_cluster",
            )
            crop = rail_crop[local.y1:local.y2, local.x1:local.x2]
            tag_type, type_score, features, hcl, vcl = self._classify_cell(crop)
            features.update({
                "rail_split_method": split_method,
                "rail_separator_count": len(separators),
                "rail_separator_sources": separator_sources,
                "rail_raw_boundaries": [int(x) for x in raw_boundaries],
                "rail_final_boundaries": [int(x) for x in boundaries],
                "cell_width_ratio_in_rail": float((x2 - x1) / max(1, rw)),
            })
            global_box.cls = f"price_cell_{tag_type}"
            local.cls = f"price_cell_{tag_type}"
            global_box.conf = type_score
            local.conf = type_score
            cells.append(
                RailCell(
                    rail_index=rail_idx,
                    cell_index=i,
                    bbox=global_box,
                    local_bbox=local,
                    tag_type=tag_type,
                    score=type_score,
                    features=features,
                    horizontal_clusters=hcl,
                    vertical_clusters=vcl,
                )
            )

        if len(cells) < self.min_cells:
            return []
        return cells

    def _split_rail_into_cells_beam(self, rail_crop: np.ndarray, rail_box: Box, rail_idx: int) -> List[RailCell]:
        """Split a rail by sequentially filling same-template price tags.

        This is deliberately different from a separator-only heuristic.  It
        builds hypotheses left-to-right: place one tag, require a price-like
        block inside it, then place the next tag of the same template.  The best
        path is selected by normalized beam score.
        """
        rh, rw = rail_crop.shape[:2]
        adaptive_min_w = max(self.min_cell_width, int(round(rw * self.beam_min_cell_width_ratio)))
        adaptive_max_w = min(max(self.max_cell_width, adaptive_min_w), max(adaptive_min_w + 1, int(round(rw * self.beam_max_cell_width_ratio))))
        if rh < 16 or rw < adaptive_min_w * max(1, self.min_cells):
            return []

        separators = self._find_vertical_separators(rail_crop)
        sep_xs = [int(round(s.center)) for s in separators]
        main_price_centers = self._main_price_centers(rail_crop)
        price_centers = main_price_centers if len(main_price_centers) >= 2 else self._price_block_centers(rail_crop)
        midpoint_xs = [int(round((a + b) * 0.5)) for a, b in zip(price_centers[:-1], price_centers[1:])]
        valley_xs = self._low_content_valleys(rail_crop)
        expected_w = self._expected_cell_width(rw, price_centers)

        # If we have several stable main-price anchors, use a constrained beam
        # over boundaries between adjacent anchors.  This prevents the splitter
        # from treating stacked progressive prices inside one tag as separate
        # price cells.
        anchor_best: Optional[BeamPath] = None
        if len(main_price_centers) >= 2:
            anchor_best = self._run_anchor_guided_beam(
                rail_crop=rail_crop,
                template_name="progressive" if self._candidate_beam_templates(rail_crop)[0].startswith("progressive") else self._candidate_beam_templates(rail_crop)[0],
                main_price_centers=main_price_centers,
                cue_xs={"separator": sep_xs, "price_midpoint": midpoint_xs, "low_content_valley": valley_xs},
                expected_w=expected_w,
            )
            if anchor_best is not None and anchor_best.normalized_score >= self.beam_min_path_score:
                return self._cells_from_beam_path(
                    rail_crop=rail_crop,
                    rail_box=rail_box,
                    rail_idx=rail_idx,
                    path=anchor_best,
                    separators=separators,
                    candidate_boundaries=anchor_best.boundaries,
                    cue_xs={"separator": sep_xs, "price_midpoint": midpoint_xs, "low_content_valley": valley_xs},
                    expected_w=expected_w,
                    price_centers=main_price_centers,
                    split_method="main_price_anchor_beam",
                )

        boundary_step = max(8, int(self.beam_boundary_step_px))
        grid_xs = list(range(0, rw + 1, boundary_step))
        if grid_xs[-1] != rw:
            grid_xs.append(rw)
        candidate_boundaries = sorted(set(
            [0, rw]
            + grid_xs
            + [x for x in sep_xs if adaptive_min_w // 2 <= x <= rw - adaptive_min_w // 2]
            + [x for x in midpoint_xs if adaptive_min_w // 2 <= x <= rw - adaptive_min_w // 2]
            + [x for x in valley_xs if adaptive_min_w // 2 <= x <= rw - adaptive_min_w // 2]
        ))

        cue_xs = {
            "separator": sep_xs,
            "price_midpoint": midpoint_xs,
            "low_content_valley": valley_xs,
        }
        templates = self._candidate_beam_templates(rail_crop)
        paths: List[BeamPath] = []
        for template_name in templates:
            path = self._run_same_template_beam(
                rail_crop=rail_crop,
                template_name=template_name,
                candidate_boundaries=candidate_boundaries,
                cue_xs=cue_xs,
                min_w=adaptive_min_w,
                max_w=adaptive_max_w,
                expected_w=expected_w,
            )
            if path is not None:
                paths.append(path)

        if not paths:
            return []
        best = max(paths, key=lambda p: (p.normalized_score, len(p.boundaries)))
        if best.normalized_score < self.beam_min_path_score:
            if self.debug:
                self.last_debug.setdefault("beam_paths", []).extend([p.to_dict() for p in paths])
                self.last_debug["beam_reject_reason"] = "score_below_threshold"
            return []

        if self.debug:
            self.last_debug.setdefault("beam_paths", []).extend([p.to_dict() for p in paths])
            self.last_debug["beam_best_path"] = best.to_dict()

        return self._cells_from_beam_path(
            rail_crop=rail_crop,
            rail_box=rail_box,
            rail_idx=rail_idx,
            path=best,
            separators=separators,
            candidate_boundaries=candidate_boundaries,
            cue_xs=cue_xs,
            expected_w=expected_w,
            price_centers=price_centers,
            split_method="same_template_beam",
        )

    def _cells_from_beam_path(
        self,
        *,
        rail_crop: np.ndarray,
        rail_box: Box,
        rail_idx: int,
        path: BeamPath,
        separators: Sequence[Cluster1D],
        candidate_boundaries: Sequence[int],
        cue_xs: Dict[str, Sequence[int]],
        expected_w: Optional[float],
        price_centers: Sequence[float],
        split_method: str,
    ) -> List[RailCell]:
        rh, rw = rail_crop.shape[:2]
        cells: List[RailCell] = []
        for i, (x1, x2) in enumerate(zip(path.boundaries[:-1], path.boundaries[1:])):
            if x2 - x1 < max(16, int(self.min_cell_width * 0.60)):
                continue
            x1e = max(0, int(x1) - self.cell_expand_px)
            x2e = min(rw, int(x2) + self.cell_expand_px)
            local = Box("price_cell", x1e, 0, x2e, rh, 0.65, "template_beam")
            global_box = Box(
                "price_cell",
                rail_box.x1 + local.x1,
                rail_box.y1 + local.y1,
                rail_box.x1 + local.x2,
                rail_box.y1 + local.y2,
                0.65,
                "template_beam",
            )
            crop = rail_crop[local.y1:local.y2, local.x1:local.x2]
            observed_type, type_score, features, hcl, vcl = self._classify_cell(crop)
            beam_score, beam_features = self._score_template_cell(
                crop=crop,
                template_name=path.template_name,
                x1=x1,
                x2=x2,
                rw=rw,
                cue_xs=cue_xs,
                expected_w=expected_w,
                main_price_centers=price_centers,
            )
            tag_type = path.template_name
            if tag_type == "progressive_yellow" and observed_type == "progressive":
                tag_type = "progressive_yellow"
            elif tag_type == "progressive" and observed_type == "progressive_yellow":
                tag_type = "progressive_yellow"
            conf = float(max(0.05, min(0.99, 0.50 * type_score + 0.50 * beam_score)))
            features.update(beam_features)
            features.update({
                "rail_split_method": split_method,
                "rail_separator_count": len(separators),
                "rail_separator_sources": sorted({str(s.source) for s in separators}),
                "rail_raw_boundaries": [int(x) for x in candidate_boundaries],
                "rail_final_boundaries": [int(x) for x in path.boundaries],
                "cell_width_ratio_in_rail": float((x2 - x1) / max(1, rw)),
                "beam_template": path.template_name,
                "beam_path_score": path.normalized_score,
                "beam_observed_type": observed_type,
                "beam_expected_width": expected_w,
                "beam_price_centers": [float(x) for x in price_centers],
            })
            global_box.cls = f"price_cell_{tag_type}"
            local.cls = f"price_cell_{tag_type}"
            global_box.conf = conf
            local.conf = conf
            cells.append(
                RailCell(
                    rail_index=rail_idx,
                    cell_index=i,
                    bbox=global_box,
                    local_bbox=local,
                    tag_type=tag_type,
                    score=conf,
                    features=features,
                    horizontal_clusters=hcl,
                    vertical_clusters=vcl,
                )
            )
        return cells if len(cells) >= self.min_cells else []

    def _run_same_template_beam(
        self,
        *,
        rail_crop: np.ndarray,
        template_name: str,
        candidate_boundaries: Sequence[int],
        cue_xs: Dict[str, Sequence[int]],
        min_w: int,
        max_w: int,
        expected_w: Optional[float],
    ) -> Optional[BeamPath]:
        h, w = rail_crop.shape[:2]
        boundaries = sorted(set(int(x) for x in candidate_boundaries if 0 <= int(x) <= w))
        by_x = {x: i for i, x in enumerate(boundaries)}
        if 0 not in by_x or w not in by_x:
            return None

        # state: (current_x, path_boundaries, per_cell_scores, total_score)
        states: List[Tuple[int, List[int], List[float], float]] = [(0, [0], [], 0.0)]
        finished: List[Tuple[List[int], List[float], float]] = []
        max_cells = max(self.min_cells, min(14, max(2, w // max(1, min_w))))
        trailing_slack = max(8, int(0.04 * w))
        score_cache: Dict[Tuple[int, int], float] = {}

        def candidate_ends_for(x: int) -> List[int]:
            possible: List[int] = []
            for end in boundaries:
                if end <= x:
                    continue
                span = end - x
                if end != w and (span < min_w or span > max_w):
                    continue
                if end == w and span < int(min_w * 0.72):
                    continue
                if span > max_w * 1.15:
                    continue
                possible.append(end)
            if not possible:
                return []
            def rank(end: int) -> float:
                span = end - x
                width_penalty = 0.0 if expected_w is None else abs(float(span) - float(expected_w)) / max(1.0, float(expected_w))
                cue_bonus = self._boundary_cue_bonus(end, w, cue_xs)
                end_bonus = 0.20 if end == w else 0.0
                return width_penalty - cue_bonus - end_bonus
            possible.sort(key=rank)
            return possible[:max(6, min(10, self.beam_size // 2))]

        for _ in range(max_cells):
            next_states: List[Tuple[int, List[int], List[float], float]] = []
            for x, path, scores, total in states:
                remaining = w - x
                if remaining <= trailing_slack:
                    finished.append((path[:-1] + [w] if path[-1] != w else path, scores, total))
                    continue
                for end in candidate_ends_for(x):
                    span = end - x
                    key = (x, end)
                    if key in score_cache:
                        score = score_cache[key]
                    else:
                        cell = rail_crop[:, x:end]
                        score, _ = self._score_template_cell(
                            crop=cell,
                            template_name=template_name,
                            x1=x,
                            x2=end,
                            rw=w,
                            cue_xs=cue_xs,
                            expected_w=expected_w,
                        )
                        score_cache[key] = score
                    if score < 0.18:
                        continue
                    boundary_bonus = self._boundary_cue_bonus(end, w, cue_xs)
                    width_bonus = self._width_prior_score(span, expected_w) * 0.18
                    step_score = float(score + boundary_bonus + width_bonus)
                    next_states.append((end, path + [end], scores + [score], total + step_score))
            if not next_states:
                break
            next_states.sort(key=lambda st: (st[3] / max(1, len(st[2])), st[0]), reverse=True)
            states = next_states[:max(1, self.beam_size)]

        for x, path, scores, total in states:
            if w - x <= trailing_slack:
                finished.append((path[:-1] + [w] if path[-1] != w else path, scores, total))

        if not finished:
            return None
        candidates: List[BeamPath] = []
        for path, scores, total in finished:
            if len(path) < self.min_cells + 1:
                continue
            coverage = (path[-1] - path[0]) / max(1.0, float(w))
            n_cells = max(1, len(path) - 1)
            normalized = float(total / n_cells)
            # Penalize very many tiny cells and large uncovered margins.  The
            # adaptive min width already prevents most pathological splits.
            normalized -= max(0.0, 1.0 - coverage) * 0.25
            spans = [b - a for a, b in zip(path[:-1], path[1:])]
            if spans:
                cv = float(np.std(spans) / (np.mean(spans) + 1e-6))
                normalized -= min(0.12, cv * 0.12)
            candidates.append(BeamPath(
                template_name=template_name,
                boundaries=[int(x) for x in path],
                cell_scores=[float(x) for x in scores],
                total_score=float(total),
                normalized_score=normalized,
                features={"coverage": coverage, "cell_count": n_cells},
            ))
        if not candidates:
            return None
        return max(candidates, key=lambda p: (p.normalized_score, p.features.get("coverage", 0.0)))

    def _run_anchor_guided_beam(
        self,
        *,
        rail_crop: np.ndarray,
        template_name: str,
        main_price_centers: Sequence[float],
        cue_xs: Dict[str, Sequence[int]],
        expected_w: Optional[float],
    ) -> Optional[BeamPath]:
        """Beam over internal boundaries between adjacent main-price anchors.

        The state space is intentionally small: for N detected main-price blocks,
        we create N cells and only vary the N-1 internal boundaries around
        midpoints/cues.  This models the retail prior that one visible main price
        belongs to one price tag and avoids splitting one progressive tag into
        stacked price rows.
        """
        h, w = rail_crop.shape[:2]
        centers = [float(c) for c in sorted(main_price_centers) if 0.035 * w <= float(c) <= 0.965 * w]
        if len(centers) < 2:
            return None
        if expected_w is None:
            expected_w = self._expected_cell_width(w, centers)
        gap_candidates: List[List[int]] = []
        min_gap = max(18, int(0.10 * (expected_w or self.min_cell_width)))
        for a, b in zip(centers[:-1], centers[1:]):
            mid = 0.5 * (a + b)
            local: List[int] = [int(round(mid))]
            # Nearby physical cues are allowed, but only if they stay between
            # the adjacent anchors.  This rejects vertical strokes inside digits.
            lo = a + min_gap
            hi = b - min_gap
            for xs in cue_xs.values():
                for x in xs:
                    if lo <= float(x) <= hi and abs(float(x) - mid) <= max(42.0, 0.22 * (expected_w or (b - a))):
                        local.append(int(round(float(x))))
            for delta in (-28, -16, 16, 28):
                x = int(round(mid + delta))
                if lo <= x <= hi:
                    local.append(x)
            # Deduplicate and keep closest candidates to the midpoint.
            uniq = sorted(set(max(1, min(w - 1, x)) for x in local), key=lambda x: abs(float(x) - mid))
            gap_candidates.append(uniq[:max(3, min(7, self.beam_size // 2))])

        states: List[Tuple[List[int], List[float], float]] = [([0], [], 0.0)]
        for gi, cands in enumerate(gap_candidates):
            next_states: List[Tuple[List[int], List[float], float]] = []
            for path, scores, total in states:
                prev = path[-1]
                for x in cands:
                    if x <= prev + max(16, int(self.min_cell_width * 0.45)):
                        continue
                    # Ensure current cell contains exactly current anchor.
                    if not (prev <= centers[gi] <= x):
                        continue
                    if centers[gi + 1] <= x:
                        continue
                    crop = rail_crop[:, prev:x]
                    score, _ = self._score_template_cell(
                        crop=crop,
                        template_name=template_name,
                        x1=prev,
                        x2=x,
                        rw=w,
                        cue_xs=cue_xs,
                        expected_w=expected_w,
                        main_price_centers=centers,
                    )
                    midpoint = 0.5 * (centers[gi] + centers[gi + 1])
                    midpoint_prior = max(0.0, 1.0 - abs(float(x) - midpoint) / max(1.0, 0.35 * (expected_w or (centers[gi + 1] - centers[gi]))))
                    step_score = float(score + 0.16 * midpoint_prior + self._boundary_cue_bonus(x, w, cue_xs))
                    next_states.append((path + [int(x)], scores + [float(score)], total + step_score))
            if not next_states:
                return None
            next_states.sort(key=lambda st: st[2] / max(1, len(st[1])), reverse=True)
            states = next_states[:max(1, self.beam_size)]

        finished: List[BeamPath] = []
        for path, scores, total in states:
            prev = path[-1]
            if not (prev <= centers[-1] <= w):
                continue
            crop = rail_crop[:, prev:w]
            score, _ = self._score_template_cell(
                crop=crop,
                template_name=template_name,
                x1=prev,
                x2=w,
                rw=w,
                cue_xs=cue_xs,
                expected_w=expected_w,
                main_price_centers=centers,
            )
            all_scores = scores + [float(score)]
            path2 = path + [w]
            spans = [b - a for a, b in zip(path2[:-1], path2[1:])]
            cv = float(np.std(spans) / (np.mean(spans) + 1e-6)) if spans else 0.0
            total2 = float(total + score + 0.20)  # right edge prior
            normalized = float(total2 / max(1, len(all_scores)))
            normalized -= min(0.18, cv * 0.16)
            finished.append(BeamPath(
                template_name=template_name,
                boundaries=[int(x) for x in path2],
                cell_scores=all_scores,
                total_score=total2,
                normalized_score=normalized,
                features={"coverage": 1.0, "cell_count": len(path2) - 1, "anchor_guided": True, "main_price_centers": centers},
            ))
        if not finished:
            return None
        return max(finished, key=lambda p: p.normalized_score)

    def _candidate_beam_templates(self, rail_crop: np.ndarray) -> List[str]:
        masks = _color_masks(rail_crop)
        h = rail_crop.shape[0]
        bottom_warm = float(np.mean(masks["warm"][int(h * 0.45):, :])) if h > 1 else 0.0
        yellow = float(np.mean(masks["yellow"]))
        warm = float(np.mean(masks["warm"]))
        out: List[str] = []
        if yellow > 0.16:
            out.append("progressive_yellow")
        if bottom_warm > 0.09 or warm > 0.11:
            out.append("progressive")
        out.append("ordinary")
        # Preserve order and uniqueness.
        uniq: List[str] = []
        for t in out:
            if t not in uniq:
                uniq.append(t)
        return uniq

    def _score_template_cell(
        self,
        *,
        crop: np.ndarray,
        template_name: str,
        x1: int,
        x2: int,
        rw: int,
        cue_xs: Dict[str, Sequence[int]],
        expected_w: Optional[float],
        main_price_centers: Optional[Sequence[float]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        if crop.size == 0:
            return 0.0, {"price_block_score": 0.0}
        h, w = crop.shape[:2]
        masks = _color_masks(crop)
        warm = float(np.mean(masks["warm"]))
        yellow = float(np.mean(masks["yellow"]))
        dark = float(np.mean(masks["dark"]))
        bottom_warm = float(np.mean(masks["warm"][int(h * 0.45):, :])) if h > 1 else 0.0
        price_score = self._price_block_score(crop)
        structure = 0.0
        if template_name == "ordinary":
            white = float(np.mean(masks["white"]))
            structure += min(0.35, white * 0.45)
            structure += min(0.20, dark * 0.55)
            # Ordinary cells may still have a red strip, but progressive cells get
            # better score under the progressive template.
            structure -= max(0.0, bottom_warm - 0.20) * 0.20
        elif template_name == "progressive_yellow":
            structure += min(0.32, yellow * 0.95)
            structure += min(0.24, bottom_warm * 0.65)
            structure += min(0.16, warm * 0.35)
        else:  # progressive
            structure += min(0.34, bottom_warm * 0.80)
            structure += min(0.22, warm * 0.45)
            structure += min(0.12, yellow * 0.25)

        width_score = self._width_prior_score(w, expected_w)
        boundary_score = 0.5 * self._boundary_cue_bonus(x1, rw, cue_xs) + self._boundary_cue_bonus(x2, rw, cue_xs)
        anchor_count = 0
        anchor_score = 0.55
        if main_price_centers:
            margin = max(3.0, min(24.0, 0.06 * float(max(1, w))))
            anchor_count = sum(1 for c in main_price_centers if float(x1) + margin <= float(c) <= float(x2) - margin)
            if anchor_count == 1:
                anchor_score = 1.0
            elif anchor_count == 0:
                touches_edge = int(x1) <= 2 or int(x2) >= int(rw) - 2
                anchor_score = 0.38 if touches_edge else 0.05
            else:
                # Several main-price anchors inside one crop means the cell is
                # probably over-merged.
                anchor_score = 0.18
        score = 0.42 * price_score + structure + 0.13 * width_score + 0.05 * boundary_score + 0.26 * anchor_score
        score = float(max(0.0, min(1.20, score)))
        return score, {
            "price_block_score": float(price_score),
            "template_structure_score": float(structure),
            "template_width_score": float(width_score),
            "template_boundary_score": float(boundary_score),
            "main_price_anchor_count": int(anchor_count),
            "main_price_anchor_score": float(anchor_score),
        }

    def _price_block_score(self, crop: np.ndarray) -> float:
        h, w = crop.shape[:2]
        if h < 8 or w < 8:
            return 0.0
        masks = _color_masks(crop)
        dark = masks["dark"].astype(np.uint8) * 255
        kx = max(3, int(w * 0.020))
        ky = max(3, int(h * 0.025))
        m = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)), iterations=1)
        num, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        big = 0
        digit_area = 0.0
        for i in range(1, num):
            x, y, bw, bh, area = stats[i]
            if area < max(20, int(h * w * 0.00045)):
                continue
            if bh >= h * 0.11 and bw >= w * 0.025:
                big += 1
                digit_area += float(area)
        row = _smooth_1d((dark.mean(axis=1) / 255.0).astype(np.float32), k=9)
        col = _smooth_1d((dark.mean(axis=0) / 255.0).astype(np.float32), k=11)
        row_peak = float(np.max(row)) if row.size else 0.0
        col_peak = float(np.max(col)) if col.size else 0.0
        score = 0.0
        score += min(0.40, big * 0.13)
        score += min(0.30, (digit_area / max(1.0, float(h * w))) * 3.8)
        score += min(0.20, row_peak * 0.55)
        score += min(0.18, col_peak * 0.65)
        return float(max(0.0, min(1.0, score)))

    def _main_price_centers(self, rail_crop: np.ndarray) -> List[float]:
        """Detect centers of the upper/main price blocks in a full price-rail crop.

        Unlike ``_price_block_centers`` this deliberately ignores most lower
        red progressive rows.  The detected centers are used as anchors: one
        main price ≈ one visible price tag.
        """
        h, w = rail_crop.shape[:2]
        if h < 24 or w < 32:
            return []
        masks = _color_masks(rail_crop)
        warm_row = _smooth_1d(masks["warm"].astype(np.float32).mean(axis=1), k=15)
        # First large warm run in the lower half is usually the red/orange block.
        red_top = int(h * 0.55)
        runs = _runs((warm_row > 0.35) & (np.arange(h) > int(0.30 * h)), warm_row, min_size=max(5, int(0.035 * h)))
        if runs:
            red_top = int(runs[0][0])
        y1 = max(0, int(red_top - 0.36 * h))
        y2 = min(h, int(red_top + 0.13 * h))
        if y2 <= y1 + 8:
            y1, y2 = max(0, int(0.25 * h)), min(h, int(0.68 * h))

        dark = masks["dark"][y1:y2, :].astype(np.uint8) * 255
        if dark.size == 0:
            return []
        kx = max(2, int(w * 0.006))
        ky = max(4, int(h * 0.018))
        m = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)), iterations=1)
        num, _, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
        raw: List[Tuple[float, float]] = []
        for i in range(1, num):
            x, y, bw, bh, area = stats[i]
            cx = float(centroids[i][0])
            # Drop border artifacts and tiny text. Main price digits are tall.
            if cx < 0.035 * w or cx > 0.965 * w:
                continue
            if area < max(60, int(w * h * 0.00035)):
                continue
            if bh < h * 0.075 or bw < w * 0.010:
                continue
            if bw > w * 0.24 or bh > h * 0.42:
                continue
            # Prefer components in the lower part of the white block, where the
            # main price normally sits, not product-name text at the top.
            cy_global = y1 + float(centroids[i][1])
            if cy_global < y1 + 0.18 * (y2 - y1):
                continue
            raw.append((cx, float(area)))
        if not raw:
            return []
        raw.sort(key=lambda t: t[0])
        clusters: List[List[Tuple[float, float]]] = []
        # Digits inside one price are typically separated by < 70-90 px at this
        # scale; different tags are separated by substantially more.
        max_digit_gap = max(55.0, min(115.0, 0.16 * float(w)))
        for cx, wt in raw:
            if not clusters:
                clusters.append([(cx, wt)])
                continue
            prev_center = float(np.average([v[0] for v in clusters[-1]], weights=[v[1] for v in clusters[-1]]))
            if cx - prev_center > max_digit_gap:
                clusters.append([(cx, wt)])
            else:
                clusters[-1].append((cx, wt))
        centers: List[float] = []
        for cl in clusters:
            if not cl:
                continue
            vals = [v[0] for v in cl]
            wts = [v[1] for v in cl]
            total_wt = float(np.sum(wts))
            if total_wt < max(80.0, w * h * 0.00045):
                continue
            centers.append(float(np.average(vals, weights=wts)))
        # Merge very close centers that survived as separate digits.
        merged: List[float] = []
        for c in centers:
            if not merged or c - merged[-1] > max(85.0, 0.12 * w):
                merged.append(c)
            else:
                merged[-1] = 0.5 * (merged[-1] + c)
        return merged

    def _price_block_centers(self, rail_crop: np.ndarray) -> List[float]:
        h, w = rail_crop.shape[:2]
        masks = _color_masks(rail_crop)
        dark = masks["dark"].astype(np.uint8) * 255
        kx = max(4, int(w * 0.010))
        ky = max(3, int(h * 0.025))
        m = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)), iterations=1)
        num, _, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
        centers: List[float] = []
        weights: List[float] = []
        for i in range(1, num):
            x, y, bw, bh, area = stats[i]
            if area < max(35, int(w * h * 0.00055)):
                continue
            if bh < h * 0.12 or bw < w * 0.015:
                continue
            if bw > w * 0.30 or bh > h * 0.80:
                continue
            centers.append(float(centroids[i][0]))
            weights.append(float(area))
        if not centers:
            return []
        order = np.argsort(np.asarray(centers))
        centers_sorted = [centers[int(i)] for i in order]
        weights_sorted = [weights[int(i)] for i in order]
        clusters: List[List[Tuple[float, float]]] = []
        max_gap = max(35.0, float(self.min_cell_width) * 0.45)
        for c, wt in zip(centers_sorted, weights_sorted):
            if not clusters or c - np.average([v[0] for v in clusters[-1]], weights=[v[1] for v in clusters[-1]]) > max_gap:
                clusters.append([(c, wt)])
            else:
                clusters[-1].append((c, wt))
        out: List[float] = []
        for cl in clusters:
            if not cl:
                continue
            vals = [v[0] for v in cl]
            wts = [v[1] for v in cl]
            out.append(float(np.average(vals, weights=wts)))
        return out

    def _low_content_valleys(self, rail_crop: np.ndarray) -> List[int]:
        h, w = rail_crop.shape[:2]
        gray = cv2.cvtColor(rail_crop, cv2.COLOR_BGR2GRAY) if rail_crop.ndim == 3 else rail_crop.copy()
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 45, 140).astype(np.float32) / 255.0
        masks = _color_masks(rail_crop)
        dark = masks["dark"].astype(np.float32)
        warm = masks["warm"].astype(np.float32)
        score = _smooth_1d((0.55 * dark.mean(axis=0) + 0.25 * edges.mean(axis=0) + 0.20 * warm.mean(axis=0)).astype(np.float32), k=21)
        if score.size == 0:
            return []
        thr = float(np.percentile(score, 28))
        valleys: List[int] = []
        min_dist = max(28, self.min_cell_width // 2)
        last = -10_000
        for i in range(3, w - 3):
            if score[i] > thr:
                continue
            if score[i] <= float(np.min(score[max(0, i - 3):min(w, i + 4)])) and i - last >= min_dist:
                valleys.append(int(i))
                last = i
        return valleys

    @staticmethod
    def _expected_cell_width(rw: int, price_centers: Sequence[float]) -> Optional[float]:
        if len(price_centers) >= 2:
            diffs = np.diff(np.asarray(price_centers, dtype=np.float32))
            diffs = diffs[(diffs > max(40.0, rw * 0.06)) & (diffs < rw * 0.55)]
            if diffs.size:
                return float(np.median(diffs))
        return None

    @staticmethod
    def _width_prior_score(width: int | float, expected_w: Optional[float]) -> float:
        if expected_w is None or expected_w <= 1.0:
            return 0.55
        sigma = max(20.0, expected_w * 0.35)
        z = (float(width) - expected_w) / sigma
        return float(np.exp(-0.5 * z * z))

    @staticmethod
    def _boundary_cue_bonus(x: int | float, rw: int, cue_xs: Dict[str, Sequence[int]]) -> float:
        x = float(x)
        if x <= 2 or x >= rw - 2:
            return 0.20
        best = 0.0
        for name, xs in cue_xs.items():
            for cx in xs:
                d = abs(float(cx) - x)
                if d <= 5:
                    val = 0.22 if name == "separator" else 0.16
                elif d <= 14:
                    val = 0.12 if name == "separator" else 0.08
                else:
                    continue
                if val > best:
                    best = val
        return float(best)

    def _find_vertical_separators(self, rail_crop: np.ndarray) -> List[Cluster1D]:
        h, w = rail_crop.shape[:2]
        gray = cv2.cvtColor(rail_crop, cv2.COLOR_BGR2GRAY) if rail_crop.ndim == 3 else rail_crop.copy()
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 45, 140)
        sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        masks = _color_masks(rail_crop)

        # Vertical border cues.  QR codes and large digits also produce vertical
        # edges, so we require either line-like height coverage or a separator-like
        # narrow peak after smoothing.
        edge_score = (edges.astype(np.float32) / 255.0).mean(axis=0)
        sobel_score = sobel_x.mean(axis=0) / (float(np.max(sobel_x)) + 1e-6)
        warm_score = masks["warm"].astype(np.float32).mean(axis=0)
        dark_score = masks["dark"].astype(np.float32).mean(axis=0)
        score = 0.55 * edge_score + 0.30 * sobel_score + 0.10 * warm_score + 0.05 * dark_score
        score = _smooth_1d(score.astype(np.float32), k=25)

        # Height coverage: split image into bins and count bins containing edges
        # near this x.  True plastic/paper separators often span several bins.
        bins = max(4, min(8, h // 24))
        coverage = np.zeros(w, dtype=np.float32)
        for bi in range(bins):
            y1 = int(round(bi * h / bins))
            y2 = int(round((bi + 1) * h / bins))
            band = edges[y1:y2, :]
            if band.size == 0:
                continue
            band_score = _smooth_1d((band.mean(axis=0) / 255.0).astype(np.float32), k=11)
            coverage += (band_score > max(0.02, float(np.percentile(band_score, 80)))).astype(np.float32)
        coverage /= float(max(1, bins))

        thr = max(float(np.percentile(score, 86)), float(score.mean() + 0.55 * score.std()))
        candidates: List[Cluster1D] = []
        min_dist = max(self.min_cell_width // 2, 36)
        last_x = -10_000
        for x in _local_maxima(score, thr=thr, min_distance=min_dist):
            if x < self.min_cell_width // 2 or x > w - self.min_cell_width // 2:
                continue
            # Reject peaks that are only local text/digit clutter.
            cov = float(np.mean(coverage[max(0, x - 3):min(w, x + 4)]))
            if cov < 0.28 and score[x] < thr * 1.20:
                continue
            if x - last_x < min_dist:
                continue
            candidates.append(Cluster1D(start=max(0, x - 2), end=min(w, x + 3), score=float(score[x] + cov), axis="x", source="vertical_projection"))
            last_x = x

        # If separators are weak, use price/content x-clusters and split between
        # their centers.  This fallback is intentionally conservative.
        if len(candidates) < 1:
            content_centers = self._content_x_centers(rail_crop)
            for a, b in zip(content_centers[:-1], content_centers[1:]):
                if b - a >= self.min_cell_width:
                    x = int(round((a + b) * 0.5))
                    candidates.append(Cluster1D(start=x - 2, end=x + 3, score=0.35, axis="x", source="content_center_midpoint"))

        return sorted(candidates, key=lambda c: c.center)

    def _content_x_centers(self, rail_crop: np.ndarray) -> List[float]:
        h, w = rail_crop.shape[:2]
        gray = cv2.cvtColor(rail_crop, cv2.COLOR_BGR2GRAY) if rail_crop.ndim == 3 else rail_crop.copy()
        masks = _color_masks(rail_crop)
        dark = masks["dark"].astype(np.uint8) * 255
        # Join digits/text within one price block but avoid merging whole rail.
        kx = max(5, int(w * 0.012))
        ky = max(3, int(h * 0.025))
        m = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)), iterations=1)
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
        centers: List[float] = []
        for i in range(1, num):
            x, y, bw, bh, area = stats[i]
            if area < max(25, int(w * h * 0.0004)):
                continue
            if bh < h * 0.10 and bw < w * 0.03:
                continue
            if bw > w * 0.45:
                continue
            centers.append(float(centroids[i][0]))
        if not centers:
            return []
        centers = sorted(centers)
        clustered: List[List[float]] = []
        for c in centers:
            if not clustered or c - np.mean(clustered[-1]) > max(45, self.min_cell_width * 0.55):
                clustered.append([c])
            else:
                clustered[-1].append(c)
        return [float(np.mean(c)) for c in clustered]

    def _boundaries_from_separators(self, separators: Sequence[Cluster1D], w: int) -> List[int]:
        xs = [0]
        for s in separators:
            x = int(round(s.center))
            if self.min_cell_width // 2 <= x <= w - self.min_cell_width // 2:
                xs.append(x)
        xs.append(w)
        xs = sorted(set(xs))
        # Drop separators that create tiny adjacent cells.
        cleaned = [xs[0]]
        for x in xs[1:-1]:
            if x - cleaned[-1] < self.min_cell_width:
                continue
            if w - x < self.min_cell_width:
                continue
            cleaned.append(x)
        cleaned.append(xs[-1])
        return cleaned

    def _regularize_boundaries(self, boundaries: List[int], w: int) -> List[int]:
        if len(boundaries) <= 2:
            # Equal-width fallback for a very wide rail.
            if w >= self.min_cell_width * self.min_cells:
                n = max(self.min_cells, int(round(w / max(self.min_cell_width * 1.7, 1))))
                n = max(2, min(n, max(2, w // max(1, self.min_cell_width))))
                return [int(round(i * w / n)) for i in range(n + 1)]
            return boundaries

        out = [boundaries[0]]
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            span = b - a
            if span <= self.max_cell_width:
                if b != out[-1]:
                    out.append(b)
                continue
            # Overly wide cell -> split into roughly regular subcells.
            n = int(round(span / max(self.max_cell_width * 0.72, self.min_cell_width)))
            n = max(2, min(n, span // max(1, self.min_cell_width)))
            for j in range(1, n + 1):
                x = int(round(a + span * j / n))
                if x - out[-1] >= self.min_cell_width or j == n:
                    out.append(x)
        if out[-1] != w:
            out.append(w)
        return sorted(set(out))

    # ------------------------------------------------------------------
    # Ordinary/progressive classification inside each cell
    # ------------------------------------------------------------------

    def _classify_cell(self, crop: np.ndarray) -> Tuple[str, float, Dict[str, Any], List[Cluster1D], List[Cluster1D]]:
        h, w = crop.shape[:2]
        masks = _color_masks(crop)
        warm_ratio = float(np.mean(masks["warm"]))
        yellow_ratio = float(np.mean(masks["yellow"]))
        dark_ratio = float(np.mean(masks["dark"]))
        bottom_warm = float(np.mean(masks["warm"][int(h * 0.50):, :])) if h > 1 else 0.0

        dark = masks["dark"].astype(np.uint8) * 255
        # Horizontal clusters: rows with text/price activity.  Progressive labels
        # usually have several stacked price rows in the lower colored area.
        row = _smooth_1d((dark.mean(axis=1) / 255.0).astype(np.float32), k=9)
        row_thr = max(float(np.percentile(row, 72)), float(row.mean() + 0.35 * row.std())) if row.size else 0.0
        hclusters = [Cluster1D(a, b, s, "y", "cell_dark_row_projection") for a, b, s in self._runs(row >= row_thr, row, min_size=max(3, int(h * 0.035)))]

        # Vertical clusters: columns with large dark connected components / prices.
        col = _smooth_1d((dark.mean(axis=0) / 255.0).astype(np.float32), k=11)
        col_thr = max(float(np.percentile(col, 75)), float(col.mean() + 0.45 * col.std())) if col.size else 0.0
        vclusters = [Cluster1D(a, b, s, "x", "cell_dark_col_projection") for a, b, s in self._runs(col >= col_thr, col, min_size=max(4, int(w * 0.035)))]

        big_digit_like = self._count_big_digit_components(dark, crop_h=h, crop_w=w)
        lower_hclusters = [c for c in hclusters if c.center > h * 0.42]
        right_or_multi_columns = len([c for c in vclusters if c.size >= max(8, int(w * 0.04))])

        progressive_score = 0.0
        progressive_score += 0.32 if bottom_warm > 0.16 else 0.0
        progressive_score += 0.18 if warm_ratio > 0.13 else 0.0
        progressive_score += 0.12 if yellow_ratio > 0.10 else 0.0
        progressive_score += 0.18 if len(lower_hclusters) >= 2 else 0.0
        progressive_score += 0.15 if big_digit_like >= 2 else 0.0
        progressive_score += 0.10 if right_or_multi_columns >= 3 else 0.0

        if progressive_score >= 0.42:
            tag_type = "progressive"
            score = min(0.95, 0.50 + progressive_score)
        else:
            tag_type = "ordinary"
            score = min(0.90, 0.55 + (0.20 if dark_ratio > 0.035 else 0.0) + (0.10 if len(hclusters) >= 1 else 0.0))

        if yellow_ratio > 0.22 and progressive_score >= 0.34:
            tag_type = "progressive_yellow"
            score = max(score, 0.72)

        features: Dict[str, Any] = {
            "warm_ratio": warm_ratio,
            "yellow_ratio": yellow_ratio,
            "dark_ratio": dark_ratio,
            "bottom_warm_ratio": bottom_warm,
            "horizontal_cluster_count": len(hclusters),
            "lower_horizontal_cluster_count": len(lower_hclusters),
            "vertical_cluster_count": len(vclusters),
            "big_digit_like_components": big_digit_like,
            "progressive_score": progressive_score,
        }
        return tag_type, float(score), features, hclusters, vclusters

    @staticmethod
    def _count_big_digit_components(dark_mask_u8: np.ndarray, crop_h: int, crop_w: int) -> int:
        if dark_mask_u8.size == 0:
            return 0
        kx = max(2, int(crop_w * 0.010))
        ky = max(3, int(crop_h * 0.020))
        m = cv2.morphologyEx(dark_mask_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)), iterations=1)
        num, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        count = 0
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < max(20, int(crop_h * crop_w * 0.0007)):
                continue
            if h >= crop_h * 0.12 and w >= crop_w * 0.035:
                count += 1
        return count

    @staticmethod
    def _runs(mask: np.ndarray, score: np.ndarray, min_size: int) -> List[Tuple[int, int, float]]:
        return _runs(mask, score, min_size)

    def draw_debug(self, image: np.ndarray, split_result: Dict[str, Any]) -> np.ndarray:
        out = image.copy()
        rails: List[RailSegment] = split_result.get("_rail_objects", []) or []
        for rail in rails:
            rb = rail.bbox
            cv2.rectangle(out, (rb.x1, rb.y1), (rb.x2, rb.y2), (255, 180, 0), 2)
            draw_label(out, f"rail {rail.rail_index} score={rail.score:.2f}", rb.x1 + 3, rb.y1 + 18, (255, 180, 0))
            for cell in rail.cells:
                b = cell.bbox
                color = (0, 255, 0) if cell.tag_type.startswith("ordinary") else (0, 128, 255)
                if "yellow" in cell.tag_type:
                    color = (0, 220, 255)
                cv2.rectangle(out, (b.x1, b.y1), (b.x2, b.y2), color, 2)
                label = f"{cell.cell_index}:{cell.tag_type} {cell.score:.2f}"
                draw_label(out, label, b.x1 + 2, b.y1 + 36, color)
        return out

    def get_debug_info(self) -> Dict[str, Any]:
        return dict(self.last_debug)


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------


def _smooth_1d(x: np.ndarray, k: int = 15) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32)
    k = int(max(1, k))
    if k % 2 == 0:
        k += 1
    if x.size < k:
        k = int(x.size) if int(x.size) % 2 == 1 else max(1, int(x.size) - 1)
    if k <= 1:
        return x.astype(np.float32)
    return cv2.GaussianBlur(x.astype(np.float32).reshape(1, -1), (k, 1), 0).reshape(-1)


def _runs(mask: np.ndarray, score: np.ndarray, min_size: int) -> List[Tuple[int, int, float]]:
    mask = np.asarray(mask).astype(bool).reshape(-1)
    score = np.asarray(score).astype(np.float32).reshape(-1)
    out: List[Tuple[int, int, float]] = []
    start: Optional[int] = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if (not v or i == len(mask) - 1) and start is not None:
            end = i if not v else i + 1
            if end - start >= min_size:
                out.append((int(start), int(end), float(np.mean(score[start:end]))))
            start = None
    return out


def _local_maxima(score: np.ndarray, thr: float, min_distance: int) -> List[int]:
    score = np.asarray(score, dtype=np.float32).reshape(-1)
    if score.size == 0:
        return []
    raw: List[int] = []
    radius = max(2, int(min_distance // 4))
    for i in range(radius, score.size - radius):
        if score[i] < thr:
            continue
        lo = max(0, i - radius)
        hi = min(score.size, i + radius + 1)
        if score[i] >= float(np.max(score[lo:hi])):
            raw.append(i)
    if not raw:
        return []
    raw = sorted(raw, key=lambda i: float(score[i]), reverse=True)
    selected: List[int] = []
    for i in raw:
        if all(abs(i - j) >= min_distance for j in selected):
            selected.append(i)
    return sorted(selected)


def _color_masks(image: np.ndarray) -> Dict[str, np.ndarray]:
    if image.ndim == 2:
        gray = image
        hsv = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    b, g, r = cv2.split(bgr)

    warm_hsv = (((h <= 25) | (h >= 168)) & (s > 35) & (v > 45)) | ((h >= 5) & (h <= 45) & (s > 28) & (v > 50))
    warm_rgb = (r.astype(np.int16) > g.astype(np.int16) + 8) & (r.astype(np.int16) > b.astype(np.int16) + 10) & (r > 70)
    yellow = (h >= 15) & (h <= 45) & (s > 35) & (v > 80)
    white = (s < 75) & (v > 135)
    dark = gray < 115
    return {
        "warm": (warm_hsv | warm_rgb),
        "yellow": yellow,
        "white": white,
        "dark": dark,
    }
