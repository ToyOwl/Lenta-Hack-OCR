"""
Track-level aggregation for price-tag OCR results.

The pipeline often sees the same shelf label in several consecutive frames.  A
single frame can be blurred, overexposed, partially outside the crop, or confused
with a neighboring label.  This module keeps observations separated into tracks,
selects the best frame per track, and performs weighted voting over OCR fields.

It does not require a dedicated detector/tracker.  It consumes the existing
pipeline result structure:
- single-tag result -> one observation with the full image as bbox;
- price-rail result -> one observation per rail cell, using rail-cell bbox;
- optional child result fields are used when available.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .io_utils import safe_stem
from .product_card import clean_product_name, extract_price_only_candidate_from_result, normalize_price_value


@dataclass
class TrackObservation:
    frame_index: int
    image_path: str
    observation_id: str
    bbox: List[int]
    mode: str = "single_tag"
    saved_crop: str = ""
    track_hint: str = ""
    final: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)
    template: Dict[str, Any] = field(default_factory=dict)
    prices: Dict[str, Any] = field(default_factory=dict)
    parsed: Dict[str, Any] = field(default_factory=dict)
    csv_correction: Dict[str, Any] = field(default_factory=dict)
    llm_correction: Dict[str, Any] = field(default_factory=dict)
    raw_summary: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrackState:
    track_id: int
    observations: List[TrackObservation] = field(default_factory=list)
    last_bbox: List[int] = field(default_factory=list)
    last_frame_index: int = -1
    warnings: List[str] = field(default_factory=list)

    def add(self, obs: TrackObservation) -> None:
        self.observations.append(obs)
        self.last_bbox = list(obs.bbox)
        self.last_frame_index = int(obs.frame_index)


class PriceTagTrackAggregator:
    def __init__(
        self,
        enabled: bool = False,
        min_iou: float = 0.04,
        max_center_jump_ratio: float = 0.45,
        assignment_threshold: float = 0.38,
        max_frame_gap: int = 12,
        prevent_two_observations_same_frame: bool = True,
        best_blur_reference: float = 160.0,
        price_vote_boost: float = 1.30,
        product_vote_boost: float = 1.10,
        ambiguity_ratio: float = 0.84,
        copy_best_frames: bool = True,
        min_product_chars: int = 4,
        price_only_ok: bool = True,
        prefer_larger_decimal_shift: bool = True,
        decimal_shift_alias_enabled: bool = True,
        decimal_shift_min_larger_price: float = 5.0,
        digit_flip_alias_enabled: bool = True,
        split_mixed_tracks: bool = True,
        split_min_segment_observations: int = 2,
        split_min_price_support: int = 2,
        split_min_reliable_score: float = 0.72,
        split_price_gap_ratio: float = 0.16,
        split_product_token_overlap: float = 0.34,
        false_tag_filter_enabled: bool = True,
        false_tag_reject_score: float = 0.68,
        false_tag_review_score: float = 0.45,
        false_tag_max_tall_aspect: float = 0.62,
        false_tag_large_area: int = 120000,
        false_tag_min_reasonable_price: float = 2.0,
        unstable_price_consistency_review: float = 0.58,
        unstable_price_unique_reject: int = 4,
        catalog_gate_enabled: bool = True,
        catalog_reject_on_review: bool = True,
        catalog_min_text_score_for_track_accept: float = 0.50,
        catalog_min_price_consistency_for_track_accept: float = 0.62,
        catalog_min_product_consistency_for_track_accept: float = 0.62,
        debug: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.min_iou = float(min_iou)
        self.max_center_jump_ratio = float(max_center_jump_ratio)
        self.assignment_threshold = float(assignment_threshold)
        self.max_frame_gap = int(max_frame_gap)
        self.prevent_two_observations_same_frame = bool(prevent_two_observations_same_frame)
        self.best_blur_reference = float(best_blur_reference)
        self.price_vote_boost = float(price_vote_boost)
        self.product_vote_boost = float(product_vote_boost)
        self.ambiguity_ratio = float(ambiguity_ratio)
        self.copy_best_frames = bool(copy_best_frames)
        self.min_product_chars = int(min_product_chars)
        self.price_only_ok = bool(price_only_ok)
        self.prefer_larger_decimal_shift = bool(prefer_larger_decimal_shift)
        self.decimal_shift_alias_enabled = bool(decimal_shift_alias_enabled)
        self.decimal_shift_min_larger_price = float(decimal_shift_min_larger_price)
        self.digit_flip_alias_enabled = bool(digit_flip_alias_enabled)
        self.split_mixed_tracks = bool(split_mixed_tracks)
        self.split_min_segment_observations = int(split_min_segment_observations)
        self.split_min_price_support = int(split_min_price_support)
        self.split_min_reliable_score = float(split_min_reliable_score)
        self.split_price_gap_ratio = float(split_price_gap_ratio)
        self.split_product_token_overlap = float(split_product_token_overlap)
        self.false_tag_filter_enabled = bool(false_tag_filter_enabled)
        self.false_tag_reject_score = float(false_tag_reject_score)
        self.false_tag_review_score = float(false_tag_review_score)
        self.false_tag_max_tall_aspect = float(false_tag_max_tall_aspect)
        self.false_tag_large_area = int(false_tag_large_area)
        self.false_tag_min_reasonable_price = float(false_tag_min_reasonable_price)
        self.unstable_price_consistency_review = float(unstable_price_consistency_review)
        self.unstable_price_unique_reject = int(unstable_price_unique_reject)
        self.catalog_gate_enabled = bool(catalog_gate_enabled)
        self.catalog_reject_on_review = bool(catalog_reject_on_review)
        self.catalog_min_text_score_for_track_accept = float(catalog_min_text_score_for_track_accept)
        self.catalog_min_price_consistency_for_track_accept = float(catalog_min_price_consistency_for_track_accept)
        self.catalog_min_product_consistency_for_track_accept = float(catalog_min_product_consistency_for_track_accept)
        self.debug = bool(debug)
        self.tracks: List[TrackState] = []
        self._next_track_id = 0
        self._global_warnings: List[str] = []

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "PriceTagTrackAggregator":
        tcfg = cfg.get("track_aggregation", {}) if isinstance(cfg, Mapping) else {}
        return cls(
            enabled=bool(tcfg.get("enabled", False)),
            min_iou=float(tcfg.get("min_iou", 0.04)),
            max_center_jump_ratio=float(tcfg.get("max_center_jump_ratio", 0.45)),
            assignment_threshold=float(tcfg.get("assignment_threshold", 0.38)),
            max_frame_gap=int(tcfg.get("max_frame_gap", 12)),
            prevent_two_observations_same_frame=bool(tcfg.get("prevent_two_observations_same_frame", True)),
            best_blur_reference=float(tcfg.get("best_blur_reference", 160.0)),
            price_vote_boost=float(tcfg.get("price_vote_boost", 1.30)),
            product_vote_boost=float(tcfg.get("product_vote_boost", 1.10)),
            ambiguity_ratio=float(tcfg.get("ambiguity_ratio", 0.84)),
            copy_best_frames=bool(tcfg.get("copy_best_frames", True)),
            min_product_chars=int(tcfg.get("min_product_chars", 4)),
            price_only_ok=bool(tcfg.get("price_only_ok", True)),
            prefer_larger_decimal_shift=bool(tcfg.get("prefer_larger_decimal_shift", True)),
            decimal_shift_alias_enabled=bool(tcfg.get("decimal_shift_alias_enabled", True)),
            decimal_shift_min_larger_price=float(tcfg.get("decimal_shift_min_larger_price", 5.0)),
            digit_flip_alias_enabled=bool(tcfg.get("digit_flip_alias_enabled", True)),
            split_mixed_tracks=bool(tcfg.get("split_mixed_tracks", True)),
            split_min_segment_observations=int(tcfg.get("split_min_segment_observations", 2)),
            split_min_price_support=int(tcfg.get("split_min_price_support", 2)),
            split_min_reliable_score=float(tcfg.get("split_min_reliable_score", 0.72)),
            split_price_gap_ratio=float(tcfg.get("split_price_gap_ratio", 0.16)),
            split_product_token_overlap=float(tcfg.get("split_product_token_overlap", 0.34)),
            false_tag_filter_enabled=bool(tcfg.get("false_tag_filter_enabled", True)),
            false_tag_reject_score=float(tcfg.get("false_tag_reject_score", 0.68)),
            false_tag_review_score=float(tcfg.get("false_tag_review_score", 0.45)),
            false_tag_max_tall_aspect=float(tcfg.get("false_tag_max_tall_aspect", 0.62)),
            false_tag_large_area=int(tcfg.get("false_tag_large_area", 120000)),
            false_tag_min_reasonable_price=float(tcfg.get("false_tag_min_reasonable_price", 2.0)),
            unstable_price_consistency_review=float(tcfg.get("unstable_price_consistency_review", 0.58)),
            unstable_price_unique_reject=int(tcfg.get("unstable_price_unique_reject", 4)),
            catalog_gate_enabled=bool(tcfg.get("catalog_gate_enabled", True)),
            catalog_reject_on_review=bool(tcfg.get("catalog_reject_on_review", True)),
            catalog_min_text_score_for_track_accept=float(tcfg.get("catalog_min_text_score_for_track_accept", 0.50)),
            catalog_min_price_consistency_for_track_accept=float(tcfg.get("catalog_min_price_consistency_for_track_accept", 0.62)),
            catalog_min_product_consistency_for_track_accept=float(tcfg.get("catalog_min_product_consistency_for_track_accept", 0.62)),
            debug=bool(tcfg.get("debug", False)),
        )

    def add_result(self, result: Mapping[str, Any], frame_index: int) -> None:
        if not self.enabled:
            return
        observations = extract_observations(result, frame_index, best_blur_reference=self.best_blur_reference)
        if len(observations) > 1:
            self._global_warnings.append(f"frame_{frame_index}:multiple_price_tag_observations={len(observations)}")
        assigned_tracks: set[int] = set()
        for obs in observations:
            tr = self._match_track(obs, assigned_tracks)
            if tr is None:
                tr = self._new_track()
            else:
                assigned_tracks.add(tr.track_id)
            tr.add(obs)

    def _new_track(self) -> TrackState:
        tr = TrackState(track_id=self._next_track_id)
        self._next_track_id += 1
        self.tracks.append(tr)
        return tr

    def _match_track(self, obs: TrackObservation, assigned_tracks: set[int]) -> Optional[TrackState]:
        best_track: Optional[TrackState] = None
        best_score = -1.0
        for tr in self.tracks:
            if tr.last_frame_index < 0 or not tr.last_bbox:
                continue
            if obs.frame_index - tr.last_frame_index > self.max_frame_gap:
                continue
            if self.prevent_two_observations_same_frame and tr.track_id in assigned_tracks:
                continue
            if self.prevent_two_observations_same_frame and tr.last_frame_index == obs.frame_index:
                continue
            geom_score = _geometry_assignment_score(obs.bbox, tr.last_bbox, self.min_iou, self.max_center_jump_ratio)
            if geom_score <= 0.0:
                continue
            sig_score = _signature_score(obs, tr.observations[-1] if tr.observations else None)
            score = 0.72 * geom_score + 0.28 * sig_score
            if score > best_score:
                best_score = score
                best_track = tr
        if best_track is not None and best_score >= self.assignment_threshold:
            return best_track
        return None

    def aggregate(self, out_dir: Optional[Path] = None) -> Dict[str, Any]:
        tracks: List[Dict[str, Any]] = []
        for tr in self.tracks:
            split_states = self.split_track_state(tr)
            split_count = len(split_states)
            for split_index, (sub_state, split_meta) in enumerate(split_states):
                agg = self._aggregate_track(sub_state, out_dir=out_dir)
                agg["split"] = {**dict(split_meta), "split_index": split_index, "split_count": split_count}
                if split_count > 1:
                    agg.setdefault("warnings", [])
                    agg["warnings"] = list(dict.fromkeys(list(agg.get("warnings") or []) + ["source_track_was_split_by_aggregation"]))
                tracks.append(agg)
        tracks.sort(key=lambda x: (x.get("best_observation", {}).get("frame_index", 10**9), x.get("track_id", 0)))
        return {
            "enabled": self.enabled,
            "track_count": len(tracks),
            "tracks": tracks,
            "warnings": list(dict.fromkeys(self._global_warnings)),
        }


    def split_track_state(self, tr: TrackState) -> List[Tuple[TrackState, Dict[str, Any]]]:
        """Split a filesystem track when OCR evidence shows several different tags.

        The detector/tracker sometimes glues two neighboring shelf labels into one
        folder.  We split only when the new price/product hypothesis has repeated
        future support; single-frame OCR glitches stay inside the current segment.
        """
        if not self.split_mixed_tracks or len(tr.observations) < 2:
            return [(tr, {"split": False, "reason": "disabled_or_short"})]

        obs_sorted = sorted(tr.observations, key=lambda o: (int(o.frame_index), str(o.observation_id)))
        segments: List[List[TrackObservation]] = []
        metas: List[Dict[str, Any]] = []
        cur: List[TrackObservation] = []
        cur_prices: List[str] = []
        cur_products: List[str] = []

        def close_segment(reason: str, before_obs: TrackObservation) -> None:
            nonlocal cur, cur_prices, cur_products
            if cur:
                segments.append(cur)
                metas.append({
                    "split": True,
                    "reason": reason,
                    "break_before_frame": int(before_obs.frame_index),
                    "price_votes_before": _counter_top(cur_prices),
                    "product_votes_before": _counter_top(cur_products),
                })
            cur = []
            cur_prices = []
            cur_products = []

        for i, obs in enumerate(obs_sorted):
            price = _reliable_split_price(obs, min_score=self.split_min_reliable_score)
            product = _reliable_split_product(obs, min_score=max(0.50, self.split_min_reliable_score - 0.10), min_chars=self.min_product_chars)
            dom_price = _dominant_string(cur_prices)
            dom_product = _dominant_string(cur_products)

            should_split = False
            reason = ""
            if price and dom_price and not _prices_compatible(price, dom_price, gap_ratio=self.split_price_gap_ratio, digit_flip_alias_enabled=self.digit_flip_alias_enabled):
                old_support = sum(1 for p in cur_prices if _prices_compatible(p, dom_price, gap_ratio=self.split_price_gap_ratio, digit_flip_alias_enabled=self.digit_flip_alias_enabled))
                new_support = _future_price_support(obs_sorted, i, price, min_score=self.split_min_reliable_score, gap_ratio=self.split_price_gap_ratio, digit_flip_alias_enabled=self.digit_flip_alias_enabled)
                if old_support >= self.split_min_price_support and new_support >= self.split_min_price_support:
                    should_split = True
                    reason = f"price_change:{dom_price}->{price};old_support={old_support};new_support={new_support}"

            if not should_split and product and dom_product:
                overlap = _token_overlap(_norm_text(product), _norm_text(dom_product))
                old_support = sum(1 for p in cur_products if _token_overlap(_norm_text(p), _norm_text(dom_product)) >= self.split_product_token_overlap)
                new_support = _future_product_support(obs_sorted, i, product, min_score=max(0.50, self.split_min_reliable_score - 0.10), min_chars=self.min_product_chars, overlap_thr=self.split_product_token_overlap)
                if overlap < self.split_product_token_overlap and old_support >= self.split_min_price_support and new_support >= self.split_min_price_support:
                    should_split = True
                    reason = f"product_change:{truncate_for_meta(dom_product)}->{truncate_for_meta(product)};old_support={old_support};new_support={new_support}"

            if should_split and len(cur) >= self.split_min_segment_observations:
                close_segment(reason, obs)

            cur.append(obs)
            if price:
                cur_prices.append(price)
            if product:
                cur_products.append(product)

        if cur:
            segments.append(cur)
            metas.append({"split": len(segments) > 1, "reason": "tail", "price_votes_before": _counter_top(cur_prices), "product_votes_before": _counter_top(cur_products)})

        if len(segments) <= 1:
            return [(tr, {"split": False, "reason": "no_stable_change"})]

        out: List[Tuple[TrackState, Dict[str, Any]]] = []
        for idx, seg in enumerate(segments):
            st = TrackState(track_id=tr.track_id * 100 + idx)
            st.warnings = list(tr.warnings)
            for o in seg:
                st.add(o)
            meta = dict(metas[idx] if idx < len(metas) else {})
            meta.update({
                "split": True,
                "split_index": idx,
                "split_count": len(segments),
                "source_track_id": tr.track_id,
                "frame_span": [int(min(o.frame_index for o in seg)), int(max(o.frame_index for o in seg))],
            })
            out.append((st, meta))
        return out

    def _aggregate_track(self, tr: TrackState, out_dir: Optional[Path] = None) -> Dict[str, Any]:
        if not tr.observations:
            return {"track_id": tr.track_id, "status": "empty"}
        price_vote = _weighted_price_vote(
            ((_observation_price_candidate(o), _field_vote_weight(o, "main_price") * self.price_vote_boost) for o in tr.observations),
            ambiguity_ratio=self.ambiguity_ratio,
            prefer_larger_decimal_shift=self.prefer_larger_decimal_shift,
            alias_enabled=self.decimal_shift_alias_enabled,
            min_larger_price=self.decimal_shift_min_larger_price,
            digit_flip_alias_enabled=self.digit_flip_alias_enabled,
        )
        product_vote = _weighted_text_vote(
            ((_observation_product_candidate(o, min_chars=self.min_product_chars), _field_vote_weight(o, "product_name") * self.product_vote_boost) for o in tr.observations),
            ambiguity_ratio=self.ambiguity_ratio,
        )
        unit_vote = _weighted_vote(((_get_final_value(o, "unit"), o.score) for o in tr.observations), ambiguity_ratio=self.ambiguity_ratio)
        item_id_vote = _weighted_vote((_product_match_item_id(o, accepted_only=True), o.score) for o in tr.observations)
        discount_vote = _weighted_vote(((_get_final_value(o, "discount_percent_raw"), o.score) for o in tr.observations), ambiguity_ratio=self.ambiguity_ratio)
        stockout_vote = _weighted_vote(((_observation_stock_status(o), o.score) for o in tr.observations), ambiguity_ratio=self.ambiguity_ratio)
        best = _select_best_observation(tr.observations, preferred_price=price_vote.get("value"), preferred_product=product_vote.get("value"))

        agg_final = dict(best.final or {})
        if price_vote["value"]:
            agg_final["main_price"] = price_vote["value"]
        if product_vote["value"]:
            agg_final["product_name"] = product_vote["value"]
        if unit_vote["value"]:
            agg_final["unit"] = unit_vote["value"]
        if item_id_vote["value"]:
            pm = dict(agg_final.get("product_match") or {})
            pm.setdefault("status", "matched")
            pm["item_id"] = item_id_vote["value"]
            agg_final["product_match"] = pm
        if discount_vote.get("value"):
            agg_final["discount_percent_raw"] = discount_vote["value"]
        stockout_diag = _stockout_diagnostics(tr.observations, stockout_vote)
        if stockout_vote.get("value") == "out_of_stock" and stockout_diag.get("dominant"):
            agg_final["stock_status"] = "out_of_stock"
            agg_final["stock_status_label"] = "товар закончился"
            agg_final["stock_status_confidence"] = stockout_diag.get("confidence", stockout_vote.get("weight", 0.0))

        warnings = list(tr.warnings)
        if price_vote.get("ambiguous"):
            warnings.append("track_price_vote_ambiguous")
        if price_vote.get("decimal_shift_resolved"):
            warnings.append("track_price_decimal_shift_resolved_prefer_larger")
        if price_vote.get("digit_flip_resolved"):
            warnings.append("track_price_digit_flip_alias_resolved")
        if product_vote.get("ambiguous"):
            warnings.append("track_product_vote_ambiguous")
        if len(tr.observations) >= 2 and _track_has_large_bbox_jump(tr.observations):
            warnings.append("track_bbox_jump_possible_tracking_error")
        if self.price_only_ok and price_vote.get("value") and not product_vote.get("value"):
            warnings.append("price_only_track_missing_product_name")

        votes = {
            "main_price": price_vote,
            "product_name": product_vote,
            "unit": unit_vote,
            "item_id": item_id_vote,
            "discount_percent_raw": discount_vote,
            "stock_status": stockout_vote,
        }
        diagnostics = _track_diagnostics(tr.observations, votes=votes)
        diagnostics["stock_status"] = stockout_diag
        validation = self._validate_track(tr.observations, agg_final, votes, diagnostics, best)
        warnings.extend(validation.get("warnings", []))
        status = validation.get("status") or "ok"
        if status == "ok" and (self.price_only_ok and price_vote.get("value") and not product_vote.get("value")):
            status = "price_only_ok"
        if validation.get("needs_review"):
            agg_final["needs_review"] = True
        if agg_final.get("stock_status") == "out_of_stock":
            warnings.append("track_stockout_service_tag_detected")
            if status in {"ok", "price_only_ok", "needs_review", "rejected_false_candidate"}:
                status = "out_of_stock"

        catalog_gate = _gate_catalog_match(
            agg_final,
            warnings=warnings,
            status=status,
            validation=validation,
            diagnostics=diagnostics,
            min_text_score=self.catalog_min_text_score_for_track_accept,
            min_price_consistency=self.catalog_min_price_consistency_for_track_accept,
            min_product_consistency=self.catalog_min_product_consistency_for_track_accept,
            reject_on_review=self.catalog_reject_on_review,
            enabled=self.catalog_gate_enabled,
        )
        if catalog_gate.get("rejected"):
            warnings.append("track_catalog_match_rejected_by_evidence")
            if status == "ok":
                status = "ok_db_rejected"
            agg_final["needs_review"] = True

        best_copy = ""
        if out_dir is not None and self.copy_best_frames:
            best_copy = _copy_best_observation_image(best, out_dir / "tracks")

        return {
            "track_id": tr.track_id,
            "status": status,
            "num_observations": len(tr.observations),
            "frame_span": [int(min(o.frame_index for o in tr.observations)), int(max(o.frame_index for o in tr.observations))],
            "best_observation": _compact_observation(best),
            "best_frame_copy": best_copy,
            "aggregated_final": agg_final,
            "votes": votes,
            "diagnostics": diagnostics,
            "validation": validation,
            "catalog_gate": catalog_gate,
            "warnings": list(dict.fromkeys(warnings)),
            "observations": [_compact_observation(o) for o in tr.observations],
        }


    def _validate_track(
        self,
        observations: Sequence[TrackObservation],
        final: Mapping[str, Any],
        votes: Mapping[str, Any],
        diagnostics: Mapping[str, Any],
        best: TrackObservation,
    ) -> Dict[str, Any]:
        warnings: List[str] = []
        reasons: List[Dict[str, Any]] = []
        score = 0.0
        needs_review = False
        status = "ok"

        price_diag = diagnostics.get("price") if isinstance(diagnostics.get("price"), Mapping) else {}
        product_diag = diagnostics.get("product") if isinstance(diagnostics.get("product"), Mapping) else {}
        price_consistency = _to_float(price_diag.get("winner_ratio"), 1.0)
        price_unique = int(price_diag.get("unique_count") or 0)
        product_consistency = _to_float(product_diag.get("winner_ratio"), 1.0)
        product_unique = int(product_diag.get("unique_count") or 0)

        if price_unique >= self.unstable_price_unique_reject and price_consistency < self.unstable_price_consistency_review:
            warnings.append("track_price_vote_unstable_many_unique_values")
            reasons.append({"reason": "price_vote_unstable_many_unique", "score": 0.42, "unique_count": price_unique, "winner_ratio": round(price_consistency, 4)})
            score += 0.42
            needs_review = True
        elif price_unique >= 2 and price_consistency < self.unstable_price_consistency_review:
            warnings.append("track_price_vote_low_consistency")
            reasons.append({"reason": "price_vote_low_consistency", "score": 0.24, "unique_count": price_unique, "winner_ratio": round(price_consistency, 4)})
            score += 0.24
            needs_review = True

        if product_unique >= 2 and product_consistency < 0.72:
            warnings.append("track_product_vote_low_consistency")
            reasons.append({"reason": "product_vote_low_consistency", "score": 0.18, "unique_count": product_unique, "winner_ratio": round(product_consistency, 4)})
            score += 0.18
            needs_review = True

        if self.false_tag_filter_enabled:
            visual = _visual_false_tag_evidence(best, max_tall_aspect=self.false_tag_max_tall_aspect, large_area=self.false_tag_large_area)
            if visual.get("score", 0.0) > 0:
                score += float(visual.get("score", 0.0))
                reasons.append(visual)
                warnings.extend(visual.get("warnings", []))

            price = normalize_price_value(final.get("main_price"))
            if price:
                try:
                    pf = float(price)
                    if pf < self.false_tag_min_reasonable_price:
                        score += 0.22
                        warnings.append("false_tag_price_below_min_reasonable")
                        reasons.append({"reason": "price_below_min_reasonable", "score": 0.22, "price": price, "min_price": self.false_tag_min_reasonable_price})
                except Exception:
                    pass

            cat = _catalog_match_evidence(observations, final)
            if cat.get("catalog_available") and cat.get("max_score", 0.0) < cat.get("min_score", 0.35):
                score += 0.26
                warnings.append("false_tag_product_far_from_catalog")
                reasons.append({"reason": "product_far_from_catalog", "score": 0.26, **cat})

            if product_unique >= 2 and (votes.get("product_name") or {}).get("ambiguous"):
                score += 0.12
                warnings.append("false_tag_product_vote_ambiguous")
                reasons.append({"reason": "product_vote_ambiguous", "score": 0.12})

        if score >= self.false_tag_reject_score:
            status = "rejected_false_candidate"
            needs_review = True
        elif price_unique >= self.unstable_price_unique_reject and price_consistency < self.unstable_price_consistency_review:
            status = "mixed_track_review"
            needs_review = True
        elif score >= self.false_tag_review_score:
            status = "needs_review"
            needs_review = True

        return {
            "status": status,
            "score": round(float(min(1.0, score)), 4),
            "needs_review": bool(needs_review),
            "warnings": list(dict.fromkeys(warnings)),
            "reasons": reasons,
        }


def extract_observations(result: Mapping[str, Any], frame_index: int, best_blur_reference: float = 160.0) -> List[TrackObservation]:
    mode = str(result.get("mode", "single_tag") or "single_tag")
    if mode == "price_rail":
        observations: List[TrackObservation] = []
        for i, child in enumerate(result.get("cell_results", []) or []):
            if not isinstance(child, Mapping):
                continue
            rail_cell = child.get("rail_cell") if isinstance(child.get("rail_cell"), Mapping) else {}
            bbox = _safe_bbox(rail_cell.get("bbox"), default=[0, 0, 1, 1])
            obs_final = child.get("final") if isinstance(child.get("final"), Mapping) else {}
            if not obs_final:
                obs_final = _summary_to_final(child)
            obs = TrackObservation(
                frame_index=int(frame_index),
                image_path=str(result.get("image_path", "") or ""),
                observation_id=f"frame{frame_index:06d}_rail{rail_cell.get('rail_index', 0)}_cell{rail_cell.get('cell_index', i)}",
                bbox=bbox,
                mode="price_rail_cell",
                saved_crop=str(child.get("saved_cell_crop", "") or ""),
                track_hint=f"rail={rail_cell.get('rail_index', 0)} cell={rail_cell.get('cell_index', i)} type={rail_cell.get('tag_type', '')}",
                final=dict(obs_final),
                quality=dict(child.get("quality") or {}),
                template=dict(child.get("template") or {}),
                prices=dict(child.get("prices") or {}),
                parsed=dict(child.get("parsed") or {}),
                csv_correction=dict(child.get("csv_correction") or {}),
                llm_correction=dict(child.get("llm_correction") or {}),
                raw_summary=dict(child),
            )
            obs.score = _frame_quality_score(obs, best_blur_reference=best_blur_reference)
            observations.append(obs)
        return observations

    q = result.get("quality") if isinstance(result.get("quality"), Mapping) else {}
    w = int(q.get("width", 1) or 1)
    h = int(q.get("height", 1) or 1)
    obs = TrackObservation(
        frame_index=int(frame_index),
        image_path=str(result.get("image_path", "") or ""),
        observation_id=f"frame{frame_index:06d}_single",
        bbox=[0, 0, max(1, w), max(1, h)],
        mode="single_tag",
        final=_build_observation_final(result),
        quality=dict(q),
        template=dict(result.get("template") or {}),
        prices=dict(result.get("prices") or {}),
        parsed=dict(result.get("parsed") or {}),
        csv_correction=dict(result.get("csv_correction") or {}),
        llm_correction=dict(result.get("llm_correction") or {}),
        raw_summary={
            "status": result.get("status"),
            "mode": result.get("mode"),
            "image_path": result.get("image_path"),
            "tilt": result.get("tilt"),
            "glare_suppression": result.get("glare_suppression"),
            "ocr_all_text_joined": ((result.get("ocr") or {}).get("all_text_joined") if isinstance(result.get("ocr"), Mapping) else ""),
            "codes_count": len(result.get("codes") or []),
            "codes": result.get("codes") or [],
            "layout_classes": [b.get("cls") for b in (result.get("layout") or []) if isinstance(b, Mapping)],
            "price_only_candidate": extract_price_only_candidate_from_result(result),
        },
    )
    obs.score = _frame_quality_score(obs, best_blur_reference=best_blur_reference)
    return [obs]


def write_track_outputs(track_result: Mapping[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "track_aggregation.json", "w", encoding="utf-8") as f:
        json.dump(track_result, f, ensure_ascii=False, indent=2)
    cols = ["track_id", "num_observations", "frame_span", "best_image", "best_score", "main_price", "product_name", "unit", "warnings"]
    with open(out_dir / "track_summary.tsv", "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for tr in track_result.get("tracks", []) or []:
            best = tr.get("best_observation", {}) if isinstance(tr, Mapping) else {}
            final = tr.get("aggregated_final", {}) if isinstance(tr, Mapping) else {}
            row = {
                "track_id": tr.get("track_id"),
                "num_observations": tr.get("num_observations"),
                "frame_span": "-".join(str(x) for x in tr.get("frame_span", [])),
                "best_image": best.get("image_path") or best.get("saved_crop"),
                "best_score": best.get("score"),
                "main_price": final.get("main_price"),
                "product_name": final.get("product_name"),
                "unit": final.get("unit"),
                "warnings": ",".join(tr.get("warnings", []) or []),
            }
            f.write("\t".join(str(row.get(c, "") or "") for c in cols) + "\n")


def _summary_to_final(child: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "main_price": normalize_price_value(child.get("final_main_price") or child.get("main_price")),
        "product_name": child.get("final_product_name") or None,
        "needs_review": child.get("needs_review"),
    }



def _build_observation_final(result: Mapping[str, Any]) -> Dict[str, Any]:
    final = dict(result.get("final") or {})
    if not normalize_price_value(final.get("main_price")):
        price_only = extract_price_only_candidate_from_result(result)
        if price_only and price_only.get("value"):
            final["main_price"] = price_only.get("value")
            final["price_source"] = price_only.get("zone")
            final["price_confidence"] = price_only.get("confidence")
    # Remove OCR garbage product names.  In price-only tracks a one/two-letter
    # fragment like "чс" must not win the product vote.
    pn = _clean_product_candidate(final.get("product_name"))
    if pn:
        final["product_name"] = pn
    else:
        final.pop("product_name", None)
    return final


def _field_vote_weight(obs: TrackObservation, field_name: str) -> float:
    w = max(0.0, float(obs.score))
    if field_name == "main_price":
        if obs.final.get("price_confidence") not in (None, ""):
            try:
                w *= 0.85 + 0.25 * max(0.0, min(1.0, float(obs.final.get("price_confidence"))))
            except Exception:
                pass
        if (obs.raw_summary or {}).get("price_only_candidate"):
            w *= 1.08
    elif field_name == "product_name":
        pn = _observation_product_candidate(obs)
        if not pn:
            return 0.0
        # Longer product strings are usually more informative, but keep the
        # multiplier bounded so one hallucinated long line cannot dominate.
        w *= min(1.25, 0.85 + len(pn) / 80.0)
        # Catalog/DB is not a field-vote amplifier.  It is gated later at the
        # track level.  Keeping product votes OCR-only prevents false stable
        # products when DB matched only by price/noisy text.
    return w


def _select_best_observation(
    observations: Sequence[TrackObservation],
    *,
    preferred_price: Any = None,
    preferred_product: Any = None,
) -> TrackObservation:
    pref_price = normalize_price_value(preferred_price)
    pref_product = _norm_text(str(preferred_product or ""))

    def score(o: TrackObservation) -> float:
        s = float(o.score)
        if pref_price and _observation_price_candidate(o) == pref_price:
            s += 0.12
        op = _norm_text(str(_observation_product_candidate(o) or ""))
        if pref_product and op and _token_overlap(op, pref_product) >= 0.68:
            s += 0.06
        if (o.final or {}).get("needs_review"):
            s -= 0.06
        return s

    return max(observations, key=score)


def _observation_price_candidate(obs: TrackObservation) -> Optional[str]:
    # Prefer final field already corrected by product_card/CSV, but resolve
    # decimal-point ambiguity using stored price-only evidence when available.
    v = normalize_price_value(_get_final_value(obs, "main_price"))
    poc = (obs.raw_summary or {}).get("price_only_candidate")
    if v:
        resolved = _resolve_decimal_shift_from_price_only(v, poc)
        return resolved or v
    # Then explicit parser output if present.
    main = (obs.prices or {}).get("main") if isinstance(obs.prices, Mapping) else None
    if isinstance(main, Mapping):
        v = normalize_price_value(main.get("value"))
        if v:
            resolved = _resolve_decimal_shift_from_price_only(v, poc)
            return resolved or v
    # Finally price-only candidate stored at extraction time.
    if isinstance(poc, Mapping):
        v = normalize_price_value(poc.get("value"))
        if v:
            return _resolve_decimal_shift_from_price_only(v, poc) or v
    return None

def _observation_product_candidate(obs: TrackObservation, min_chars: int = 4) -> Optional[str]:
    direct = _clean_product_candidate(_get_final_value(obs, "product_name"), min_chars=min_chars)
    if direct:
        return direct
    # Do not use CSV/DB as a product vote fallback.  The catalog is an
    # enrichment/prior source; using it as product text makes price-only or
    # weak-OCR frames hallucinate stable product votes.
    return None


def _observation_catalog_name_score(obs: TrackObservation) -> float:
    _name, score = _observation_catalog_name(obs)
    return float(score or 0.0)


def _observation_catalog_name(obs: TrackObservation) -> Tuple[Optional[str], float]:
    """Return a per-frame catalog name candidate for track-level voting.

    The CSV corrector may leave a frame as weak/ambiguous because one frame is
    noisy.  At track level the same weak catalog candidate repeated over several
    frames is valuable evidence, so we expose it to the product-name vote with a
    conservative confidence threshold.
    """
    pm = (obs.final or {}).get("product_match") if isinstance((obs.final or {}).get("product_match"), Mapping) else {}
    name = str(pm.get("catalog_name") or "").strip()
    score = _to_float(pm.get("score"), 0.0)
    if name and (score >= 0.42 or str(pm.get("status") or "") == "matched"):
        return name, max(0.42, score)

    cc = obs.csv_correction or {}
    selected = cc.get("selected") if isinstance(cc.get("selected"), Mapping) else None
    if selected is not None:
        name = str(selected.get("name") or "").strip()
        score = _to_float(selected.get("score"), 0.0)
        if name and score >= 0.42:
            return name, score

    best_name = ""
    best_score = 0.0
    for cand in cc.get("candidates") or []:
        if not isinstance(cand, Mapping):
            continue
        name = str(cand.get("name") or "").strip()
        if not name:
            continue
        score = _to_float(cand.get("score"), 0.0)
        text_score = _to_float(cand.get("text_score"), 0.0)
        price_score = _to_float(cand.get("price_score"), 0.0)
        # Price-only candidates are not safe for autofill, but when they also
        # carry partial text evidence they can participate in a multi-frame vote.
        acceptable = score >= 0.42 or (price_score >= 0.90 and text_score >= 0.24)
        if acceptable and score > best_score:
            best_name = name
            best_score = score
    if best_name:
        return best_name, best_score
    return None, 0.0


def _observation_stock_status(obs: TrackObservation) -> Optional[str]:
    status = str(_get_final_value(obs, "stock_status") or "").strip().lower()
    if status in {"out_of_stock", "sold_out", "нет_товара", "товар_закончился"}:
        return "out_of_stock"
    text = " | ".join(
        str(x or "")
        for x in [
            _get_final_value(obs, "stock_status_label"),
            _get_final_value(obs, "stock_status_text"),
            (obs.raw_summary or {}).get("ocr_all_text_joined"),
        ]
    ).lower().replace("ё", "е")
    if re.search(r"товар\w*\s+законч\w*|скоро\s+привез\w*|\bупс\b.*\bтовар", text, flags=re.I):
        return "out_of_stock"
    return None


def _clean_product_candidate(value: Any, min_chars: int = 4) -> Optional[str]:
    text = clean_product_name(str(value or "").strip())
    if not text:
        return None
    low = text.lower().replace("ё", "е")
    if low in {"none", "null", "unknown", "не определен", "не определён"}:
        return None
    if any(x in low for x in ("qr", "ean", "штрих", "карта", "ценник", "цена", "руб")):
        return None
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text)
    if len(letters) < int(min_chars):
        return None
    if len(text) <= 3:
        return None
    # Reject mostly numeric/noise strings.
    num = len(re.findall(r"\d", text))
    if num > 0 and num >= len(letters):
        return None
    return text

def _frame_quality_score(obs: TrackObservation, best_blur_reference: float) -> float:
    q = obs.quality or {}
    template = obs.template or {}
    blur = _to_float(q.get("blur_score_laplacian_var"), 0.0)
    blur_score = max(0.0, min(1.0, blur / max(1.0, float(best_blur_reference))))
    b = q.get("brightness") if isinstance(q.get("brightness"), Mapping) else {}
    over = _to_float(b.get("overexposed_ratio"), 0.0)
    under = _to_float(b.get("underexposed_ratio"), 0.0)
    exposure_score = max(0.0, 1.0 - 1.6 * max(over, under))
    template_score = max(0.0, min(1.0, _to_float(template.get("confidence"), 0.55)))
    price_bonus = 0.20 if _observation_price_candidate(obs) else 0.0
    product_bonus = 0.10 if _observation_product_candidate(obs) else 0.0
    # DB status must not make a visually weak frame become the best frame.
    csv_bonus = 0.0
    review_penalty = 0.12 if bool((obs.final or {}).get("needs_review")) else 0.0
    score = 0.42 * blur_score + 0.22 * exposure_score + 0.20 * template_score + price_bonus + product_bonus + csv_bonus - review_penalty
    return round(max(0.0, min(1.0, score)), 4)


def _geometry_assignment_score(a: Sequence[int], b: Sequence[int], min_iou: float, max_center_jump_ratio: float) -> float:
    iou = _bbox_iou(a, b)
    ca = _center(a)
    cb = _center(b)
    diag = max(1.0, math.sqrt(max(_area(a), _area(b))))
    center_dist = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
    center_score = max(0.0, 1.0 - center_dist / max(1.0, max_center_jump_ratio * diag))
    if iou < min_iou and center_score < 0.22:
        return 0.0
    return float(max(iou, 0.65 * center_score + 0.35 * iou))


def _signature_score(obs: TrackObservation, prev: Optional[TrackObservation]) -> float:
    if prev is None:
        return 0.0
    p1 = normalize_price_value(_get_final_value(obs, "main_price"))
    p2 = normalize_price_value(_get_final_value(prev, "main_price"))
    price_score = 1.0 if p1 and p2 and p1 == p2 else 0.0
    n1 = _norm_text(str(_get_final_value(obs, "product_name") or ""))
    n2 = _norm_text(str(_get_final_value(prev, "product_name") or ""))
    if n1 and n2:
        name_score = _token_overlap(n1, n2)
    else:
        name_score = 0.0
    return max(price_score, name_score)


def _weighted_price_vote(
    values: Iterable[Tuple[Any, float]],
    ambiguity_ratio: float = 0.84,
    *,
    prefer_larger_decimal_shift: bool = True,
    alias_enabled: bool = True,
    min_larger_price: float = 5.0,
    digit_flip_alias_enabled: bool = True,
) -> Dict[str, Any]:
    """Weighted price vote with decimal-point ambiguity resolution.

    OCR of price labels often alternates between ``179`` and ``1.79`` for the
    same visual digits when the decimal/kopeck zone is unclear.  For shelf tags
    with only the large readable number we prefer the larger ruble interpretation
    and merge decimal-shift aliases into one bucket.
    """
    raw: Dict[str, float] = {}
    for v, w in values:
        norm = normalize_price_value(v)
        if not norm:
            continue
        raw[norm] = raw.get(norm, 0.0) + max(0.0, float(w))
    if not raw:
        return {"value": None, "weight": 0.0, "ambiguous": False, "candidates": []}

    if not alias_enabled:
        ordered = sorted(raw.items(), key=lambda x: x[1], reverse=True)
        ambiguous = len(ordered) > 1 and ordered[1][1] >= ordered[0][1] * float(ambiguity_ratio)
        return {
            "value": ordered[0][0],
            "weight": round(float(ordered[0][1]), 4),
            "ambiguous": ambiguous,
            "candidates": [{"value": k, "weight": round(float(v), 4)} for k, v in ordered[:5]],
        }

    groups: Dict[str, Dict[str, Any]] = {}
    for price, weight in raw.items():
        key = _price_alias_key(price, digit_flip_alias_enabled=digit_flip_alias_enabled)
        g = groups.setdefault(key, {"members": {}, "weight": 0.0})
        g["members"][price] = g["members"].get(price, 0.0) + float(weight)
        g["weight"] += float(weight)

    merged: List[Dict[str, Any]] = []
    decimal_shift_resolved = False
    digit_flip_resolved = False
    for key, g in groups.items():
        members = dict(g["members"])
        member_prices = sorted(members.keys(), key=lambda x: _to_float(x, 0.0), reverse=True)
        rep = max(member_prices, key=lambda x: (members[x], _to_float(x, 0.0)))
        if _has_digit_flip_aliases(member_prices):
            digit_flip_resolved = True
        if prefer_larger_decimal_shift and len(member_prices) > 1:
            large = [p for p in member_prices if _to_float(p, 0.0) >= float(min_larger_price) and _is_integer_rubles_price(p)]
            small = [p for p in member_prices if 0.0 < _to_float(p, 0.0) < float(min_larger_price)]
            if large and small:
                rep = max(large, key=lambda x: (_to_float(x, 0.0), members.get(x, 0.0)))
                decimal_shift_resolved = True
        merged.append({
            "value": rep,
            "weight": float(g["weight"]),
            "raw_members": [{"value": p, "weight": round(float(members[p]), 4)} for p in sorted(members, key=lambda x: members[x], reverse=True)],
        })

    merged.sort(key=lambda x: x["weight"], reverse=True)
    ambiguous = len(merged) > 1 and merged[1]["weight"] >= merged[0]["weight"] * float(ambiguity_ratio)
    candidates = []
    for g in merged[:5]:
        item = {"value": g["value"], "weight": round(float(g["weight"]), 4)}
        if len(g.get("raw_members") or []) > 1:
            item["aliases"] = g["raw_members"]
        candidates.append(item)
    return {
        "value": merged[0]["value"],
        "weight": round(float(merged[0]["weight"]), 4),
        "ambiguous": ambiguous,
        "decimal_shift_resolved": bool(decimal_shift_resolved),
        "digit_flip_resolved": bool(digit_flip_resolved),
        "candidates": candidates,
    }


def _weighted_vote(values: Iterable[Tuple[Any, float]], ambiguity_ratio: float = 0.84) -> Dict[str, Any]:
    buckets: Dict[str, float] = {}
    for v, w in values:
        if v in (None, ""):
            continue
        key = str(v).strip()
        if not key:
            continue
        buckets[key] = buckets.get(key, 0.0) + max(0.0, float(w))
    if not buckets:
        return {"value": None, "weight": 0.0, "ambiguous": False, "candidates": []}
    ordered = sorted(buckets.items(), key=lambda x: x[1], reverse=True)
    ambiguous = len(ordered) > 1 and ordered[1][1] >= ordered[0][1] * float(ambiguity_ratio)
    return {
        "value": ordered[0][0],
        "weight": round(float(ordered[0][1]), 4),
        "ambiguous": ambiguous,
        "candidates": [{"value": k, "weight": round(float(v), 4)} for k, v in ordered[:5]],
    }


def _weighted_text_vote(values: Iterable[Tuple[Any, float]], ambiguity_ratio: float = 0.84) -> Dict[str, Any]:
    clusters: List[Dict[str, Any]] = []
    for v, w in values:
        text = str(v or "").strip()
        if not text:
            continue
        norm = _norm_text(text)
        if not norm:
            continue
        weight = max(0.0, float(w))
        matched = False
        for c in clusters:
            if _token_overlap(norm, c["norm"]) >= 0.68:
                c["weight"] += weight
                if weight + 0.01 * len(text) > c["best_score"]:
                    c["best_text"] = text
                    c["best_score"] = weight + 0.01 * len(text)
                matched = True
                break
        if not matched:
            clusters.append({"norm": norm, "best_text": text, "best_score": weight + 0.01 * len(text), "weight": weight})
    if not clusters:
        return {"value": None, "weight": 0.0, "ambiguous": False, "candidates": []}
    clusters.sort(key=lambda c: c["weight"], reverse=True)
    ambiguous = len(clusters) > 1 and clusters[1]["weight"] >= clusters[0]["weight"] * float(ambiguity_ratio)
    return {
        "value": clusters[0]["best_text"],
        "weight": round(float(clusters[0]["weight"]), 4),
        "ambiguous": ambiguous,
        "candidates": [{"value": c["best_text"], "weight": round(float(c["weight"]), 4)} for c in clusters[:5]],
    }


def _compact_observation(obs: TrackObservation) -> Dict[str, Any]:
    visual_consensus = (obs.raw_summary or {}).get("visual_consensus") if isinstance(obs.raw_summary, Mapping) else None
    if not isinstance(visual_consensus, Mapping):
        visual_consensus = {}
    return {
        "frame_index": obs.frame_index,
        "image_path": obs.image_path,
        "observation_id": obs.observation_id,
        "bbox": obs.bbox,
        "mode": obs.mode,
        "saved_crop": obs.saved_crop,
        "track_hint": obs.track_hint,
        "score": obs.score,
        "score_original": visual_consensus.get("original_score", obs.score),
        "visual_consensus_selected": visual_consensus.get("selected", ""),
        "visual_consensus_cluster_id": visual_consensus.get("cluster_id", ""),
        "visual_consensus_sim_to_reference": visual_consensus.get("sim_to_reference", ""),
        "ocr_consensus_sim_to_reference": visual_consensus.get("ocr_sim_to_reference", ""),
        "ocr_consensus_cluster_centrality": visual_consensus.get("ocr_cluster_centrality", ""),
        "ocr_consensus_knn_support": visual_consensus.get("ocr_knn_support", ""),
        "ocr_consensus_text_short": visual_consensus.get("ocr_text_short", ""),
        "main_price": _observation_price_candidate(obs),
        "product_name": _observation_product_candidate(obs),
        "unit": _get_final_value(obs, "unit"),
        "discount_percent_raw": _get_final_value(obs, "discount_percent_raw"),
        "old_price": _get_final_value(obs, "old_price"),
        "promo_condition": _get_final_value(obs, "promo_condition"),
        "stock_status": _observation_stock_status(obs),
        "stock_status_text": _get_final_value(obs, "stock_status_text"),
        "db_match_status": _observation_product_match_status(obs),
        "db_match_item_id": _product_match_item_id(obs),
        "db_match_score": _observation_product_match_score(obs),
        "needs_review": bool((obs.final or {}).get("needs_review")),
        "template_name": str((obs.template or {}).get("template_name") or ""),
        "template_confidence": (obs.template or {}).get("confidence", ""),
        "glare_applied": ((obs.raw_summary or {}).get("glare_suppression") or {}).get("applied") if isinstance((obs.raw_summary or {}).get("glare_suppression"), Mapping) else "",
        "glare_method": ((obs.raw_summary or {}).get("glare_suppression") or {}).get("method") if isinstance((obs.raw_summary or {}).get("glare_suppression"), Mapping) else "",
        "ocr_all_text_joined": str((obs.raw_summary or {}).get("ocr_all_text_joined") or ""),
    }


def _copy_best_observation_image(obs: TrackObservation, out_dir: Path) -> str:
    src = Path(obs.saved_crop) if obs.saved_crop else Path(str(obs.image_path).split("#", 1)[0])
    if not src.exists() or not src.is_file():
        return ""
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix or ".jpg"
    dst = out_dir / f"track_best_{safe_stem(Path(obs.observation_id))}{suffix}"
    try:
        shutil.copy2(src, dst)
        return str(dst)
    except Exception:
        return ""


def _track_has_large_bbox_jump(obs: Sequence[TrackObservation]) -> bool:
    if len(obs) < 2:
        return False
    for a, b in zip(obs[:-1], obs[1:]):
        score = _geometry_assignment_score(a.bbox, b.bbox, min_iou=0.0, max_center_jump_ratio=0.85)
        if score < 0.12:
            return True
    return False


def _get_final_value(obs: TrackObservation, key: str) -> Any:
    if obs.final and obs.final.get(key) not in (None, ""):
        return obs.final.get(key)
    if obs.parsed and obs.parsed.get(key) not in (None, ""):
        return obs.parsed.get(key)
    return None


def _product_match_item_id(obs: TrackObservation, *, accepted_only: bool = False) -> Optional[str]:
    pm = (obs.final or {}).get("product_match")
    if isinstance(pm, Mapping):
        status = str(pm.get("status") or "").lower()
        if accepted_only and status not in {"matched", "accepted", "csv_catalog_corrected"}:
            return None
        v = pm.get("item_id")
        return str(v) if v not in (None, "") else None
    return None



def _observation_product_match_status(obs: TrackObservation) -> str:
    pm = (obs.final or {}).get("product_match")
    if isinstance(pm, Mapping):
        return str(pm.get("status") or pm.get("source") or "")
    return ""


def _observation_product_match_score(obs: TrackObservation) -> Any:
    pm = (obs.final or {}).get("product_match")
    if isinstance(pm, Mapping):
        return pm.get("score", "")
    return ""


def _reliable_split_price(obs: TrackObservation, *, min_score: float) -> Optional[str]:
    if float(obs.score) < float(min_score):
        return None
    return _observation_price_candidate(obs)


def _reliable_split_product(obs: TrackObservation, *, min_score: float, min_chars: int) -> Optional[str]:
    if float(obs.score) < float(min_score):
        return None
    return _observation_product_candidate(obs, min_chars=min_chars)


def _resolve_decimal_shift_from_price_only(primary: Any, price_only_candidate: Any) -> Optional[str]:
    primary_norm = normalize_price_value(primary)
    if not primary_norm or not isinstance(price_only_candidate, Mapping):
        return primary_norm
    try:
        primary_f = float(primary_norm)
    except Exception:
        primary_f = 0.0
    candidates = price_only_candidate.get("candidates") if isinstance(price_only_candidate.get("candidates"), list) else []
    if not candidates:
        return primary_norm
    primary_key = _decimal_shift_key(primary_norm)
    alternatives: List[str] = []
    for c in candidates:
        if not isinstance(c, Mapping):
            continue
        val = normalize_price_value(c.get("value"))
        if not val:
            continue
        if _decimal_shift_key(val) == primary_key and _to_float(val, 0.0) > primary_f:
            if ".integer_rubles" in str(c.get("source") or "") or _is_integer_rubles_price(val):
                alternatives.append(val)
    if alternatives:
        return max(alternatives, key=lambda x: _to_float(x, 0.0))
    return primary_norm


def _decimal_shift_key(price: Any) -> str:
    norm = normalize_price_value(price)
    if not norm:
        return ""
    ip, fp = norm.split(".", 1)
    ip_clean = ip.lstrip("0") or "0"
    fp = fp[:2]
    f = _to_float(norm, 0.0)
    # 179.00 -> 179; 1.79 -> 179; 12.90 -> 1290.  This intentionally groups
    # only decimal-position variants that share the same visible digit sequence.
    if fp == "00" and f >= 5.0:
        return ip_clean
    if 0.0 < f < 100.0 and fp != "00":
        return (ip_clean + fp).lstrip("0") or "0"
    return (ip_clean + ("" if fp == "00" else fp)).lstrip("0") or "0"


def _is_integer_rubles_price(price: Any) -> bool:
    norm = normalize_price_value(price)
    return bool(norm and norm.endswith(".00") and _to_float(norm, 0.0) >= 1.0)


def _price_alias_key(price: Any, *, digit_flip_alias_enabled: bool = True) -> str:
    """Return a visual-price key for temporal grouping.

    It first removes decimal point ambiguity (179.00 and 1.79 share key 179).
    Optionally it also merges 6/9 flip OCR variants (66.00 and 99.00 share key gg).
    The flip alias is deliberately used only at aggregation/split level; the final
    displayed price is still selected from real observed candidates.
    """
    key = _decimal_shift_key(price)
    if not key or not digit_flip_alias_enabled:
        return key
    return _digit_flip_key(key)


def _digit_flip_key(digits: Any) -> str:
    s = re.sub(r"\D", "", str(digits or ""))
    if not s:
        return ""
    # Keep aliasing conservative: prices with at least one 6/9 and up to four
    # visible digits. This covers the common 66<->99 / 169<->199 cases without
    # merging arbitrary long barcodes or article numbers.
    if len(s) <= 4 and any(ch in "69" for ch in s):
        return "".join("g" if ch in "69" else ch for ch in s)
    return s


def _has_digit_flip_aliases(prices: Sequence[Any]) -> bool:
    raw_keys = {_decimal_shift_key(p) for p in prices if _decimal_shift_key(p)}
    if len(raw_keys) <= 1:
        return False
    flip_keys = {_digit_flip_key(k) for k in raw_keys}
    return len(flip_keys) == 1 and any("g" in k for k in flip_keys)


def _prices_compatible(a: Any, b: Any, *, gap_ratio: float = 0.16, digit_flip_alias_enabled: bool = True) -> bool:
    pa = normalize_price_value(a)
    pb = normalize_price_value(b)
    if not pa or not pb:
        return False
    if pa == pb:
        return True
    if _price_alias_key(pa, digit_flip_alias_enabled=digit_flip_alias_enabled) == _price_alias_key(pb, digit_flip_alias_enabled=digit_flip_alias_enabled):
        return True
    try:
        fa, fb = float(pa), float(pb)
        denom = max(1.0, min(abs(fa), abs(fb)))
        if abs(fa - fb) / denom <= float(gap_ratio):
            return True
    except Exception:
        pass
    da = re.sub(r"\D", "", pa.split(".", 1)[0])
    db = re.sub(r"\D", "", pb.split(".", 1)[0])
    if da and db and da == db:
        return True
    return False


def _future_price_support(obs: Sequence[TrackObservation], start: int, price: str, *, min_score: float, gap_ratio: float, digit_flip_alias_enabled: bool = True) -> int:
    n = 0
    for o in obs[start:]:
        p = _reliable_split_price(o, min_score=min_score)
        if p and _prices_compatible(p, price, gap_ratio=gap_ratio, digit_flip_alias_enabled=digit_flip_alias_enabled):
            n += 1
    return n


def _future_product_support(obs: Sequence[TrackObservation], start: int, product: str, *, min_score: float, min_chars: int, overlap_thr: float) -> int:
    n = 0
    target = _norm_text(product)
    for o in obs[start:]:
        p = _reliable_split_product(o, min_score=min_score, min_chars=min_chars)
        if p and _token_overlap(_norm_text(p), target) >= overlap_thr:
            n += 1
    return n


def _dominant_string(values: Sequence[str]) -> Optional[str]:
    if not values:
        return None
    counts: Dict[str, int] = {}
    for v in values:
        counts[str(v)] = counts.get(str(v), 0) + 1
    return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[0][0]


def _counter_top(values: Sequence[str], limit: int = 5) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for v in values:
        if v:
            counts[str(v)] = counts.get(str(v), 0) + 1
    return [{"value": k, "count": int(v)} for k, v in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:limit]]


def truncate_for_meta(value: Any, n: int = 32) -> str:
    s = str(value or "")
    return s if len(s) <= n else s[: max(1, n - 1)] + "…"


def _track_diagnostics(observations: Sequence[TrackObservation], *, votes: Mapping[str, Any]) -> Dict[str, Any]:
    price_values = [_observation_price_candidate(o) for o in observations]
    product_values = [_observation_product_candidate(o) for o in observations]
    return {
        "price": _vote_diagnostics(votes.get("main_price") if isinstance(votes.get("main_price"), Mapping) else {}, price_values),
        "product": _vote_diagnostics(votes.get("product_name") if isinstance(votes.get("product_name"), Mapping) else {}, product_values),
        "frame_score": {
            "min": round(min([float(o.score) for o in observations] or [0.0]), 4),
            "max": round(max([float(o.score) for o in observations] or [0.0]), 4),
            "mean": round(sum(float(o.score) for o in observations) / max(1, len(observations)), 4),
        },
        "templates": _counter_top([str((o.template or {}).get("template_name") or "") for o in observations if (o.template or {}).get("template_name")]),
    }


def _vote_diagnostics(vote: Mapping[str, Any], raw_values: Sequence[Any]) -> Dict[str, Any]:
    vals = [str(v) for v in raw_values if v not in (None, "")]
    unique = len(set(vals))
    candidates = vote.get("candidates") if isinstance(vote.get("candidates"), list) else []
    total_w = 0.0
    for c in candidates:
        if isinstance(c, Mapping):
            total_w += _to_float(c.get("weight"), 0.0)
    winner_w = _to_float(vote.get("weight"), 0.0)
    winner_ratio = (winner_w / total_w) if total_w > 1e-9 else (1.0 if winner_w <= 0 else 1.0)
    return {
        "observed_count": len(vals),
        "unique_count": unique,
        "winner_ratio": round(float(winner_ratio), 4),
        "winner_weight": round(float(winner_w), 4),
        "total_top_weight": round(float(total_w), 4),
    }


def _stockout_diagnostics(observations: Sequence[TrackObservation], vote: Mapping[str, Any]) -> Dict[str, Any]:
    flags = [_observation_stock_status(o) == "out_of_stock" for o in observations]
    count = int(sum(1 for f in flags if f))
    n = int(len(observations))
    ratio = float(count / max(1, n))
    weight = _to_float(vote.get("weight"), 0.0) if isinstance(vote, Mapping) else 0.0
    dominant = bool(count > 0 and (ratio >= 0.50 or (n <= 2 and count >= 1)))
    return {
        "observed_count": count,
        "total_observations": n,
        "frame_ratio": round(ratio, 4),
        "winner_weight": round(float(weight), 4),
        "dominant": dominant,
        "confidence": round(float(min(1.0, 0.50 + 0.45 * ratio)), 4) if count else 0.0,
    }


def _gate_catalog_match(
    final: Dict[str, Any],
    *,
    warnings: Sequence[str],
    status: str,
    validation: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    min_text_score: float,
    min_price_consistency: float,
    min_product_consistency: float,
    reject_on_review: bool,
    enabled: bool,
) -> Dict[str, Any]:
    """Reject CSV/DB product autofill when track-level evidence is unstable.

    The catalog is an enrichment source, not ground truth.  A track with an
    orange banner/not-shelf-label warning, low price consistency, or ambiguous
    product vote must not inherit a product from CSV merely because one price or
    one noisy OCR fragment matched a row.
    """
    pm = final.get("product_match") if isinstance(final.get("product_match"), Mapping) else None
    if not enabled:
        return {"enabled": False, "rejected": False, "reason": "disabled"}
    if not isinstance(pm, Mapping) or not pm.get("item_id"):
        return {"enabled": True, "rejected": False, "reason": "no_catalog_item"}
    pm_status = str(pm.get("status") or "").lower()
    if pm_status not in {"matched", "accepted", "csv_catalog_corrected"}:
        # Weak/ambiguous catalog results are already marked as priors and do not
        # overwrite OCR product text.  Keep them visible in debug output instead
        # of converting them to rejected_product_match/not_found.
        return {
            "enabled": True,
            "rejected": False,
            "reason": "catalog_prior_not_autofilled",
            "catalog_status": pm_status,
            "catalog_text_score": round(_to_float(pm.get("text_score"), 0.0), 4),
            "catalog_score": round(_to_float(pm.get("score"), 0.0), 4),
        }

    warning_set = {str(w) for w in warnings}
    hard_warning_prefixes = (
        "false_tag_",
        "track_price_vote_unstable",
        "track_price_vote_low_consistency",
        "track_product_vote_low_consistency",
        "track_product_vote_ambiguous",
        "track_bbox_jump_possible_tracking_error",
    )
    reject_reasons: List[str] = []
    if reject_on_review and str(status) in {"needs_review", "mixed_track_review", "rejected_false_candidate", "out_of_stock"}:
        reject_reasons.append(f"status:{status}")
    for w in sorted(warning_set):
        if any(w.startswith(prefix) for prefix in hard_warning_prefixes):
            reject_reasons.append(f"warning:{w}")

    price_diag = diagnostics.get("price") if isinstance(diagnostics.get("price"), Mapping) else {}
    product_diag = diagnostics.get("product") if isinstance(diagnostics.get("product"), Mapping) else {}
    price_ratio = _to_float(price_diag.get("winner_ratio"), 1.0)
    product_ratio = _to_float(product_diag.get("winner_ratio"), 1.0)
    if price_ratio < float(min_price_consistency):
        reject_reasons.append(f"price_consistency:{price_ratio:.3f}")
    if product_ratio < float(min_product_consistency):
        reject_reasons.append(f"product_consistency:{product_ratio:.3f}")

    text_score = _to_float(pm.get("text_score"), 1.0)
    if text_score < float(min_text_score):
        reject_reasons.append(f"catalog_text_score:{text_score:.3f}")

    if not reject_reasons:
        return {
            "enabled": True,
            "rejected": False,
            "reason": "accepted_by_track_gate",
            "catalog_text_score": round(text_score, 4),
            "price_consistency": round(price_ratio, 4),
            "product_consistency": round(product_ratio, 4),
        }

    rejected_pm = dict(pm)
    final["rejected_product_match"] = rejected_pm
    final["product_match"] = {
        "status": "rejected_by_track_evidence",
        "item_id": None,
        "catalog_name": rejected_pm.get("catalog_name"),
        "score": rejected_pm.get("score", 0.0),
        "source": rejected_pm.get("source", "structured_csv"),
        "reject_reasons": reject_reasons,
    }
    reasons = list(final.get("review_reasons") or []) if isinstance(final.get("review_reasons"), list) else []
    if "csv_catalog_rejected_by_track_evidence" not in reasons:
        reasons.append("csv_catalog_rejected_by_track_evidence")
    final["review_reasons"] = reasons
    return {
        "enabled": True,
        "rejected": True,
        "reject_reasons": reject_reasons,
        "rejected_product_match": rejected_pm,
        "catalog_text_score": round(text_score, 4),
        "price_consistency": round(price_ratio, 4),
        "product_consistency": round(product_ratio, 4),
    }


def _visual_false_tag_evidence(obs: TrackObservation, *, max_tall_aspect: float, large_area: int) -> Dict[str, Any]:
    q = obs.quality or {}
    w = int(q.get("width") or 0)
    h = int(q.get("height") or 0)
    if w <= 0 or h <= 0:
        x1, y1, x2, y2 = _safe_bbox(obs.bbox, [0, 0, 0, 0])
        w, h = max(1, x2 - x1), max(1, y2 - y1)
    aspect = float(w / max(1, h))
    area = int(w * h)
    tmpl = obs.template or {}
    cf = tmpl.get("color_features") if isinstance(tmpl.get("color_features"), Mapping) else {}
    regions = cf.get("regions") if isinstance(cf.get("regions"), Mapping) else {}
    def warm(region: str) -> float:
        r = regions.get(region) if isinstance(regions.get(region), Mapping) else {}
        return sum(_to_float(r.get(k), 0.0) for k in ("red", "orange", "brown", "pink", "yellow"))
    top_warm = warm("top")
    bottom_warm = warm("bottom")
    full_warm = warm("full")
    white_gray = 0.0
    full = regions.get("full") if isinstance(regions.get("full"), Mapping) else {}
    white_gray = _to_float(full.get("white"), 0.0) + _to_float(full.get("gray"), 0.0)
    score = 0.0
    warnings: List[str] = []
    details = {
        "reason": "visual_false_tag_evidence",
        "score": 0.0,
        "aspect": round(aspect, 4),
        "area": area,
        "top_warm": round(top_warm, 4),
        "bottom_warm": round(bottom_warm, 4),
        "full_warm": round(full_warm, 4),
        "full_white_gray": round(white_gray, 4),
        "template_name": str(tmpl.get("template_name") or ""),
        "warnings": warnings,
    }
    if aspect < float(max_tall_aspect) and area >= int(large_area):
        score += 0.34
        warnings.append("false_tag_tall_large_crop")
    if full_warm > 0.42 and top_warm > 0.32 and abs(bottom_warm - top_warm) < 0.18:
        score += 0.32
        warnings.append("false_tag_poster_like_uniform_warm_background")
    if full_warm > 0.50 and white_gray < 0.36 and aspect < 0.72:
        score += 0.20
        warnings.append("false_tag_orange_banner_not_shelf_label")
    details["score"] = round(float(min(0.72, score)), 4)
    return details


def _catalog_match_evidence(observations: Sequence[TrackObservation], final: Mapping[str, Any]) -> Dict[str, Any]:
    max_score = 0.0
    catalog_available = False
    min_score = 0.35
    for o in observations:
        cc = o.csv_correction or {}
        catalog = cc.get("catalog") if isinstance(cc.get("catalog"), Mapping) else {}
        if catalog.get("enabled"):
            catalog_available = True
            min_score = max(min_score, _to_float(catalog.get("min_text_score"), 0.26))
        for c in cc.get("candidates") or []:
            if isinstance(c, Mapping):
                max_score = max(max_score, _to_float(c.get("score"), 0.0))
    pm = final.get("product_match") if isinstance(final.get("product_match"), Mapping) else {}
    max_score = max(max_score, _to_float(pm.get("score"), 0.0))
    return {"catalog_available": bool(catalog_available), "max_score": round(max_score, 4), "min_score": round(min_score, 4)}

def _safe_bbox(v: Any, default: List[int]) -> List[int]:
    if isinstance(v, Sequence) and not isinstance(v, (str, bytes)) and len(v) >= 4:
        try:
            return [int(float(v[0])), int(float(v[1])), int(float(v[2])), int(float(v[3]))]
        except Exception:
            return list(default)
    return list(default)


def _bbox_iou(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = _safe_bbox(a, [0, 0, 0, 0])
    bx1, by1, bx2, by2 = _safe_bbox(b, [0, 0, 0, 0])
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = _area(a) + _area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def _area(a: Sequence[int]) -> float:
    x1, y1, x2, y2 = _safe_bbox(a, [0, 0, 0, 0])
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def _center(a: Sequence[int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = _safe_bbox(a, [0, 0, 0, 0])
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _norm_text(s: str) -> str:
    s = str(s or "").lower().replace("ё", "е")
    s = re.sub(r"[^0-9a-zа-я]+", " ", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()


def _token_overlap(a: str, b: str) -> float:
    ta = set(re.findall(r"[0-9a-zа-я]{2,}", a.lower(), flags=re.I))
    tb = set(re.findall(r"[0-9a-zа-я]{2,}", b.lower(), flags=re.I))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except Exception:
        return default
