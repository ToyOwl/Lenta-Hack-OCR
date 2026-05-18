"""
Track-image fusion and visual-consensus scoring for noisy video crops.

The detected-track runner receives several crops for the same physical label.
A naive top-N fusion is fragile: one high-quality but wrong crop, or a frame
that belongs to a neighboring label, can dominate the representative image and
OCR voting.  This module first finds the visual/evidence consensus cluster and
then gives higher weight to the best frame plus frames that converge to it.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gzip
import hashlib
import re
from typing import Any, Mapping, Sequence, Tuple

import cv2
import numpy as np

from .io_utils import imread_unicode
from .product_card import normalize_price_value
from .track_aggregator import TrackObservation
from .tag_qr_enhancement import super_resolve_crop
from .focus_metrics import compute_focus_quality, normalize_focus_scores, summarize_focus


@dataclass
class _FrameCandidate:
    obs: TrackObservation
    image: np.ndarray
    feature: np.ndarray
    original_score: float
    price: str
    product_norm: str
    template_name: str
    ocr_text: str = ""
    ocr_feature: np.ndarray | None = None
    cluster_id: int = -1
    sim_to_reference: float = 0.0
    cluster_centrality: float = 0.0
    ocr_sim_to_reference: float = 0.0
    ocr_cluster_centrality: float = 0.0
    ocr_knn_support: float = 0.0
    focus_score: float = 0.0
    focus_norm: float = 0.0
    focus_metrics: dict[str, Any] | None = None
    fusion_score: float = 0.0


def apply_track_consensus_scores(
    observations: Sequence[TrackObservation],
    *,
    enabled: bool = True,
    max_candidates: int = 32,
    feature_size: int = 64,
    visual_similarity_threshold: float = 0.66,
    evidence_similarity_threshold: float = 0.50,
    min_cluster_size: int = 2,
    ocr_similarity_enabled: bool = True,
    ocr_embedding_dim: int = 384,
    ocr_knn_k: int = 4,
    ocr_similarity_threshold: float = 0.56,
    ocr_strong_similarity_threshold: float = 0.70,
    ocr_gzip_weight: float = 0.55,
    focus_selection_enabled: bool = True,
    focus_score_weight: float = 0.18,
    focus_roi_policy: str = "price_tag",
    selected_score_boost: float = 0.22,
    outlier_score_penalty: float = 0.32,
    update_raw_summary: bool = True,
) -> dict[str, Any]:
    """Boost frame scores inside the dominant visual consensus cluster.

    The aggregator uses ``TrackObservation.score`` for price/product votes and
    best-frame selection.  This pass modifies those scores in-place:

    * frames in the selected visual cluster are boosted;
    * non-selected visual outliers are down-weighted;
    * all decisions are recorded in ``raw_summary['visual_consensus']``.

    Splitting by OCR price/product should be executed before this function when
    possible.  That way, visual consensus works inside a candidate segment rather
    than suppressing a legitimate second tag too early.
    """
    if not enabled:
        return {"enabled": False, "status": "disabled"}

    candidates, meta = _build_visual_consensus(
        observations,
        max_candidates=max_candidates,
        feature_size=feature_size,
        visual_similarity_threshold=visual_similarity_threshold,
        evidence_similarity_threshold=evidence_similarity_threshold,
        min_cluster_size=min_cluster_size,
        ocr_similarity_enabled=ocr_similarity_enabled,
        ocr_embedding_dim=ocr_embedding_dim,
        ocr_knn_k=ocr_knn_k,
        ocr_similarity_threshold=ocr_similarity_threshold,
        ocr_strong_similarity_threshold=ocr_strong_similarity_threshold,
        ocr_gzip_weight=ocr_gzip_weight,
        focus_selection_enabled=focus_selection_enabled,
        focus_score_weight=focus_score_weight,
        focus_roi_policy=focus_roi_policy,
    )
    if not candidates:
        return meta

    selected_cluster_id = meta.get("selected_cluster_id")
    selected_count = int(meta.get("selected_cluster_size") or 0)
    selected_active = selected_cluster_id is not None and selected_count >= int(min_cluster_size)
    cluster_count = int(meta.get("cluster_count") or 0)

    by_obs_id = {id(c.obs): c for c in candidates}
    changed = 0
    for obs in observations:
        cand = by_obs_id.get(id(obs))
        old_score = float(obs.score)
        new_score = old_score
        vc: dict[str, Any] = {
            "enabled": True,
            "original_score": round(old_score, 4),
            "readable": cand is not None,
            "selected": False,
        }
        if cand is not None:
            is_selected = selected_active and int(cand.cluster_id) == int(selected_cluster_id)
            vc.update(
                {
                    "cluster_id": int(cand.cluster_id),
                    "selected": bool(is_selected),
                    "sim_to_reference": round(float(cand.sim_to_reference), 4),
                    "cluster_centrality": round(float(cand.cluster_centrality), 4),
                    "fusion_score": round(float(cand.fusion_score), 4),
                    "ocr_text_available": bool(cand.ocr_text),
                    "ocr_sim_to_reference": round(float(cand.ocr_sim_to_reference), 4),
                    "ocr_cluster_centrality": round(float(cand.ocr_cluster_centrality), 4),
                    "ocr_knn_support": round(float(cand.ocr_knn_support), 4),
                    "focus_score": round(float(cand.focus_score), 4),
                    "focus_norm": round(float(cand.focus_norm), 4),
                    "ocr_text_short": cand.ocr_text[:96],
                }
            )
            if is_selected:
                # Similar frames get the largest boost.  The additive term lets
                # a slightly weaker but very stable frame overtake a sharp outlier.
                sim = max(0.0, min(1.0, float(cand.sim_to_reference)))
                central = max(0.0, min(1.0, float(cand.cluster_centrality)))
                focus = max(0.0, min(1.0, float(cand.focus_norm)))
                multiplier = 1.0 + float(selected_score_boost) * (0.28 + 0.38 * sim + 0.18 * central + 0.16 * focus)
                new_score = min(1.0, old_score * multiplier + 0.030 * sim + 0.018 * focus)
            elif selected_active and cluster_count > 1:
                new_score = max(0.0, old_score * max(0.05, 1.0 - float(outlier_score_penalty)))
        elif selected_active and cluster_count > 1:
            vc["reason"] = "image_unreadable_or_not_in_candidate_pool"
            new_score = max(0.0, old_score * max(0.05, 1.0 - float(outlier_score_penalty) * 0.65))

        obs.score = round(float(new_score), 4)
        if abs(new_score - old_score) > 1e-6:
            changed += 1
        vc["score_after"] = round(float(obs.score), 4)
        if update_raw_summary:
            try:
                obs.raw_summary.setdefault("visual_consensus", {}).update(vc)
            except Exception:
                obs.raw_summary["visual_consensus"] = vc

    meta = dict(meta)
    meta.update(
        {
            "score_update_enabled": True,
            "updated_observation_count": int(changed),
            "selected_score_boost": float(selected_score_boost),
            "outlier_score_penalty": float(outlier_score_penalty),
        }
    )
    return meta


def fuse_track_images(
    observations: Sequence[TrackObservation],
    *,
    reference_observation: TrackObservation | None = None,
    max_images: int = 9,
    max_work_side: int = 900,
    align: bool = True,
    denoise_h: float = 7.0,
    denoise_h_color: float = 7.0,
    template_window_size: int = 7,
    search_window_size: int = 21,
    consensus_enabled: bool = True,
    consensus_max_candidates: int = 32,
    consensus_feature_size: int = 64,
    visual_similarity_threshold: float = 0.66,
    evidence_similarity_threshold: float = 0.50,
    min_cluster_size: int = 2,
    ocr_similarity_enabled: bool = True,
    ocr_embedding_dim: int = 384,
    ocr_knn_k: int = 4,
    ocr_similarity_threshold: float = 0.56,
    ocr_strong_similarity_threshold: float = 0.70,
    ocr_gzip_weight: float = 0.55,
    focus_selection_enabled: bool = True,
    focus_score_weight: float = 0.18,
    focus_roi_policy: str = "price_tag",
    min_focus_norm_for_fusion: float = 0.12,
    align_mode: str = "phase_ecc",
    ecc_motion: str = "euclidean",
    ecc_min_correlation: float = 0.18,
    phase_min_response: float = 0.08,
    max_translation_ratio: float = 0.22,
    denoise_stage: str = "post_fusion",
    sr_enabled: bool = False,
    sr_scale: float = 2.0,
    sr_stage: str = "pre_nlmeans",
    sr_min_side: int = 320,
    sr_max_side: int = 1400,
    sr_method: str = "lanczos",
) -> Tuple[np.ndarray | None, dict[str, Any]]:
    """Return a fused/denoised image and metadata for a track.

    The steps are:
    1. compute compact visual features for readable frames;
    2. cluster frames by visual similarity, with a lower merge threshold when
       OCR/catalog/template evidence agrees;
    3. select the dominant consensus cluster and its central best frame;
    4. align only frames that converge to that frame;
    5. apply OpenCV fastNlMeansDenoisingColored and weighted median/mean.

    The fused image is meant for review output and QR/barcode fallback.  It is
    not used to overwrite per-frame OCR evidence.
    """
    obs = [o for o in observations if isinstance(o, TrackObservation)]
    if not obs:
        return None, {"enabled": True, "status": "empty_track"}

    selected_obs: list[TrackObservation] = []
    consensus_meta: dict[str, Any] = {"enabled": bool(consensus_enabled), "status": "disabled"}
    reference_from_consensus: TrackObservation | None = None

    if consensus_enabled:
        candidates, consensus_meta = _build_visual_consensus(
            obs,
            max_candidates=max(consensus_max_candidates, max_images),
            feature_size=consensus_feature_size,
            visual_similarity_threshold=visual_similarity_threshold,
            evidence_similarity_threshold=evidence_similarity_threshold,
            min_cluster_size=min_cluster_size,
            ocr_similarity_enabled=ocr_similarity_enabled,
            ocr_embedding_dim=ocr_embedding_dim,
            ocr_knn_k=ocr_knn_k,
            ocr_similarity_threshold=ocr_similarity_threshold,
            ocr_strong_similarity_threshold=ocr_strong_similarity_threshold,
            ocr_gzip_weight=ocr_gzip_weight,
            focus_selection_enabled=focus_selection_enabled,
            focus_score_weight=focus_score_weight,
            focus_roi_policy=focus_roi_policy,
        )
        selected_cluster_id = consensus_meta.get("selected_cluster_id")
        if candidates and selected_cluster_id is not None and int(consensus_meta.get("selected_cluster_size") or 0) >= int(min_cluster_size):
            selected = [c for c in candidates if int(c.cluster_id) == int(selected_cluster_id)]
            selected.sort(key=lambda c: float(c.fusion_score), reverse=True)
            selected_obs = [c.obs for c in selected[: max(1, int(max_images))]]
            if selected:
                reference_from_consensus = max(selected, key=lambda c: (float(c.fusion_score), float(c.original_score))).obs
        else:
            consensus_meta = dict(consensus_meta)
            consensus_meta.setdefault("warning", "consensus_cluster_not_selected")

    if not selected_obs:
        selected_obs = sorted(obs, key=lambda o: float(o.score), reverse=True)[: max(1, int(max_images))]

    if reference_observation is not None and any(o is reference_observation for o in selected_obs):
        ref = reference_observation
        reference_policy = "aggregated_best_inside_consensus"
    elif reference_from_consensus is not None:
        ref = reference_from_consensus
        reference_policy = "visual_consensus_central_best"
    elif reference_observation is not None:
        ref = reference_observation
        reference_policy = "aggregated_best_fallback"
    else:
        ref = max(selected_obs, key=lambda o: float(o.score))
        reference_policy = "score_best_fallback"

    ref_img = _read_observation_image(ref)
    if ref_img is None or ref_img.size == 0:
        return None, {"enabled": True, "status": "reference_read_failed", "reference_image": ref.image_path, "visual_consensus": consensus_meta}

    ref_img, scale = _downscale_if_needed(ref_img, max_work_side=max_work_side)
    sr_meta: dict[str, Any] = {"enabled": bool(sr_enabled), "stage": str(sr_stage or "pre_nlmeans")}
    sr_stage_l = str(sr_stage or "pre_nlmeans").lower().strip()
    sr_pre = bool(sr_enabled) and sr_stage_l in {"pre", "pre_nlmeans", "before_nlmeans", "both"}
    sr_post = bool(sr_enabled) and sr_stage_l in {"post", "post_fusion", "after_fusion", "both"}
    if sr_pre:
        ref_img, sr_ref_meta = super_resolve_crop(
            ref_img,
            enabled=True,
            scale=float(sr_scale),
            min_side=int(sr_min_side),
            max_side=int(sr_max_side),
            method=str(sr_method or "lanczos"),
        )
        sr_meta["reference"] = sr_ref_meta
    ref_h, ref_w = ref_img.shape[:2]
    candidates_obs = list(selected_obs)
    if ref not in candidates_obs:
        candidates_obs = [ref] + candidates_obs[: max(0, int(max_images) - 1)]

    sim_by_id: dict[int, float] = {}
    if consensus_enabled and isinstance(consensus_meta, Mapping):
        # Recompute a cheap reference-sim map from raw_summary if scores were
        # already updated by apply_track_consensus_scores, otherwise from the
        # local consensus candidate list.
        for o in obs:
            vc = (o.raw_summary or {}).get("visual_consensus") if isinstance(o.raw_summary, Mapping) else None
            if isinstance(vc, Mapping) and vc.get("sim_to_reference") is not None:
                sim_by_id[id(o)] = _to_float(vc.get("sim_to_reference"), 0.0)

    aligned: list[np.ndarray] = []
    weights: list[float] = []
    align_failures = 0
    source_paths: list[str] = []
    ref_gray = _alignment_gray(ref_img)
    denoise_stage_l = str(denoise_stage or "post_fusion").lower().strip()
    denoise_pre = denoise_stage_l in {"pre", "pre_frame", "pre_fusion", "both"}
    denoise_post = denoise_stage_l in {"post", "post_fusion", "after_fusion", "both"}

    for o in candidates_obs[: max(1, int(max_images))]:
        img = _read_observation_image(o)
        if img is None or img.size == 0:
            continue
        img = cv2.resize(img, (ref_w, ref_h), interpolation=cv2.INTER_AREA if max(img.shape[:2]) > max(ref_h, ref_w) else cv2.INTER_CUBIC)
        focus_norm = _obs_focus_norm(o, default=1.0 if o is ref else 0.55)
        if o is not ref and focus_selection_enabled and focus_norm < float(min_focus_norm_for_fusion) and len(candidates_obs) > 2:
            align_failures += 1
            continue
        if align and img.shape[:2] == ref_img.shape[:2] and o is not ref:
            ok, warped, align_info = _align_image_to_reference(
                img,
                ref_gray,
                mode=str(align_mode or "phase_ecc"),
                ecc_motion=str(ecc_motion or "euclidean"),
                ecc_min_correlation=float(ecc_min_correlation),
                phase_min_response=float(phase_min_response),
                max_translation_ratio=float(max_translation_ratio),
            )
            if ok:
                img = warped
            else:
                align_failures += 1
        if denoise_pre:
            img = _safe_nlmeans(
                img,
                h=float(denoise_h),
                h_color=float(denoise_h_color),
                template_window_size=int(template_window_size),
                search_window_size=int(search_window_size),
            )
        aligned.append(img)
        sim = max(0.0, min(1.0, float(sim_by_id.get(id(o), 1.0 if o is ref else 0.72))))
        weights.append(max(0.03, float(o.score)) * (0.48 + 0.32 * sim + 0.20 * max(0.0, min(1.0, focus_norm))))
        source_paths.append(str(o.image_path))

    if not aligned:
        return ref_img, {"enabled": True, "status": "fallback_reference_only", "reference_image": ref.image_path, "visual_consensus": consensus_meta}

    stack = np.stack(aligned, axis=0).astype(np.float32)
    w = np.asarray(weights, dtype=np.float32)
    w = w / max(1e-6, float(w.sum()))
    mean_img = np.tensordot(w, stack, axes=(0, 0))
    if len(aligned) >= 3:
        median_img = np.median(stack, axis=0)
        fused = 0.62 * median_img + 0.38 * mean_img
    else:
        fused = mean_img
    fused_u8 = np.clip(fused, 0, 255).astype(np.uint8)
    if denoise_post:
        fused_u8 = _safe_nlmeans(
            fused_u8,
            h=float(denoise_h),
            h_color=float(denoise_h_color),
            template_window_size=int(template_window_size),
            search_window_size=int(search_window_size),
        )
    if sr_post and not sr_pre:
        fused_u8, sr_post_meta = super_resolve_crop(
            fused_u8,
            enabled=True,
            scale=float(sr_scale),
            min_side=int(sr_min_side),
            max_side=int(sr_max_side),
            method=str(sr_method or "lanczos"),
        )
        sr_meta["post_fusion"] = sr_post_meta
    fused_u8 = _unsharp(fused_u8)

    method_name = "visual_ocr_knn_gzip_consensus_sr_nlmeans_aligned_weighted_median_mean" if sr_pre else "visual_ocr_knn_gzip_consensus_nlmeans_aligned_weighted_median_mean"
    if sr_post and not sr_pre:
        method_name = "visual_ocr_knn_gzip_consensus_nlmeans_aligned_weighted_median_mean_post_sr"

    meta = {
        "enabled": True,
        "status": "ok",
        "method": method_name,
        "frame_count": len(aligned),
        "source_count": len(candidates_obs),
        "align_enabled": bool(align),
        "align_mode": str(align_mode),
        "ecc_motion": str(ecc_motion),
        "align_failures": int(align_failures),
        "denoise_stage": str(denoise_stage),
        "focus_selection_enabled": bool(focus_selection_enabled),
        "focus_min_norm_for_fusion": float(min_focus_norm_for_fusion),
        "reference_policy": reference_policy,
        "reference_frame_index": int(ref.frame_index),
        "reference_image": str(ref.image_path),
        "reference_scale": round(float(scale), 4),
        "sr": sr_meta,
        "output_shape": list(fused_u8.shape),
        "source_images": source_paths[:12],
        "visual_consensus": consensus_meta,
    }
    return fused_u8, meta


def has_decoded_code(observations: Sequence[TrackObservation]) -> bool:
    for o in observations:
        raw = o.raw_summary or {}
        for code in raw.get("codes") or []:
            if isinstance(code, Mapping) and code.get("decoded") and str(code.get("payload") or ""):
                return True
    return False


def _build_visual_consensus(
    observations: Sequence[TrackObservation],
    *,
    max_candidates: int,
    feature_size: int,
    visual_similarity_threshold: float,
    evidence_similarity_threshold: float,
    min_cluster_size: int,
    ocr_similarity_enabled: bool = True,
    ocr_embedding_dim: int = 384,
    ocr_knn_k: int = 4,
    ocr_similarity_threshold: float = 0.56,
    ocr_strong_similarity_threshold: float = 0.70,
    ocr_gzip_weight: float = 0.55,
    focus_selection_enabled: bool = True,
    focus_score_weight: float = 0.18,
    focus_roi_policy: str = "price_tag",
) -> tuple[list[_FrameCandidate], dict[str, Any]]:
    obs_sorted = sorted([o for o in observations if isinstance(o, TrackObservation)], key=lambda o: float(o.score), reverse=True)
    obs_sorted = obs_sorted[: max(1, int(max_candidates))]

    candidates: list[_FrameCandidate] = []
    unreadable = 0
    for obs in obs_sorted:
        img = _read_observation_image(obs)
        if img is None or img.size == 0:
            unreadable += 1
            continue
        feat = _image_feature(img, feature_size=max(16, int(feature_size)))
        if feat is None:
            unreadable += 1
            continue
        ocr_text = _obs_ocr_text(obs)
        fq = compute_focus_quality(img, roi_policy=str(focus_roi_policy or "price_tag")) if focus_selection_enabled else None
        candidates.append(
            _FrameCandidate(
                obs=obs,
                image=img,
                feature=feat,
                original_score=float(obs.score),
                price=_obs_price(obs),
                product_norm=_norm_text(_obs_product(obs)),
                template_name=_obs_template(obs),
                ocr_text=ocr_text,
                ocr_feature=_ocr_embedding(ocr_text, dim=int(ocr_embedding_dim)) if ocr_similarity_enabled else None,
                focus_score=float(fq.score) if fq is not None else 0.0,
                focus_metrics=fq.to_dict() if fq is not None else None,
            )
        )

    if not candidates:
        return [], {"enabled": True, "status": "no_readable_images", "candidate_count": 0, "unreadable_count": int(unreadable)}
    _assign_focus_norms(candidates)

    if len(candidates) == 1:
        c = candidates[0]
        c.cluster_id = 0
        c.sim_to_reference = 1.0
        c.cluster_centrality = 1.0
        c.fusion_score = float(c.original_score) + float(focus_score_weight) * float(c.focus_norm)
        return candidates, {
            "enabled": True,
            "status": "single_candidate",
            "candidate_count": 1,
            "unreadable_count": int(unreadable),
            "cluster_count": 1,
            "selected_cluster_id": 0,
            "selected_cluster_size": 1,
            "clusters": [_cluster_meta(0, [c])],
        }

    n = len(candidates)
    sim = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            s = _cosine(candidates[i].feature, candidates[j].feature)
            sim[i, j] = sim[j, i] = float(s)

    ocr_sim, ocr_knn = _ocr_similarity_graph(
        candidates,
        enabled=bool(ocr_similarity_enabled),
        knn_k=int(ocr_knn_k),
        similarity_threshold=float(ocr_similarity_threshold),
        gzip_weight=float(ocr_gzip_weight),
    )

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    vis_thr = float(visual_similarity_threshold)
    ev_thr = min(float(visual_similarity_threshold), float(evidence_similarity_threshold))
    weak_thr = max(ev_thr + 0.08, vis_thr - 0.08)
    conflict_thr = max(0.84, vis_thr + 0.16)
    ocr_thr = float(ocr_similarity_threshold)
    ocr_strong_thr = float(ocr_strong_similarity_threshold)
    for i in range(n):
        for j in range(i + 1, n):
            sij = float(sim[i, j])
            tij = float(ocr_sim[i, j])
            knn_link = bool(ocr_knn[i, j])
            evidence_conflict = _candidate_evidence_conflicts(candidates[i], candidates[j])
            if evidence_conflict and sij < conflict_thr and tij < max(ocr_strong_thr + 0.08, 0.78):
                continue

            evidence_level = _candidate_evidence_level(candidates[i], candidates[j])
            if evidence_level == "strong":
                thr = ev_thr
            elif evidence_level == "weak":
                thr = weak_thr
            else:
                thr = vis_thr

            # OCR-KNN/GZip is used as an additional weak metric.  It can lower
            # the visual threshold for consecutive blurry frames with the same
            # text, but a pure OCR link still needs either strong text similarity
            # or at least some visual compatibility.
            if knn_link and tij >= ocr_thr:
                thr = min(thr, max(ev_thr, vis_thr - 0.10))
            if tij >= ocr_strong_thr:
                thr = min(thr, max(ev_thr, vis_thr - 0.16))

            if sij >= thr or (tij >= ocr_strong_thr and sij >= max(0.34, ev_thr - 0.06)):
                union(i, j)

    root_to_indices: dict[int, list[int]] = {}
    for idx in range(n):
        root_to_indices.setdefault(find(idx), []).append(idx)

    clusters_idx = list(root_to_indices.values())
    clusters_idx.sort(key=lambda idxs: (len(idxs), sum(candidates[i].original_score for i in idxs)), reverse=True)
    root_remap: dict[int, int] = {}
    for new_id, idxs in enumerate(clusters_idx):
        for idx in idxs:
            root_remap[idx] = new_id
            candidates[idx].cluster_id = new_id

    cluster_infos: list[dict[str, Any]] = []
    cluster_scores: list[tuple[int, float]] = []
    for cluster_id, idxs in enumerate(clusters_idx):
        members = [candidates[i] for i in idxs]
        if len(idxs) >= 2:
            pair_sims = [float(sim[a, b]) for pos, a in enumerate(idxs) for b in idxs[pos + 1 :]]
            centrality = float(np.mean(pair_sims)) if pair_sims else 1.0
            ocr_pair_sims = [float(ocr_sim[a, b]) for pos, a in enumerate(idxs) for b in idxs[pos + 1 :] if float(ocr_sim[a, b]) > 0.0]
            ocr_centrality = float(np.mean(ocr_pair_sims)) if ocr_pair_sims else 0.0
        else:
            centrality = 0.0
            ocr_centrality = 0.0
        price_ratio = _dominant_ratio([m.price for m in members if m.price])
        product_ratio = _dominant_ratio([m.product_norm for m in members if m.product_norm])
        template_ratio = _dominant_ratio([m.template_name for m in members if m.template_name])
        score_sum = sum(max(0.03, float(m.original_score)) for m in members)
        focus_mean = float(np.mean([max(0.0, min(1.0, m.focus_norm)) for m in members])) if members else 0.0
        best_focus = max((float(m.focus_norm) for m in members), default=0.0)
        best_raw = max(float(m.original_score) for m in members)
        agreement = max(price_ratio, product_ratio, template_ratio)
        ocr_bonus = 0.08 * min(1.0, ocr_centrality) if ocr_similarity_enabled else 0.0
        size_bonus = min(0.20, 0.045 * max(0, len(members) - 1))
        cluster_score = score_sum * (0.50 + 0.26 * centrality + 0.09 * agreement + ocr_bonus + float(focus_score_weight) * 0.45 * focus_mean) + best_raw * 0.10 + best_focus * float(focus_score_weight) * 0.22 + size_bonus
        cluster_scores.append((cluster_id, float(cluster_score)))
        cluster_infos.append(_cluster_meta(cluster_id, members, centrality=centrality, cluster_score=cluster_score, ocr_centrality=ocr_centrality))

    selected_cluster_id, selected_score = max(cluster_scores, key=lambda x: x[1])
    selected_indices = clusters_idx[selected_cluster_id]
    selected_members = [candidates[i] for i in selected_indices]

    # Find the most central strong frame inside the selected cluster.  This is
    # usually better than the globally sharpest frame when a crop belongs to a
    # neighbor or contains transient blur/glare.
    ref_idx = selected_indices[0]
    best_ref_score = -1.0
    for idx in selected_indices:
        if len(selected_indices) >= 2:
            centrality_to_cluster = float(np.mean([sim[idx, j] for j in selected_indices if j != idx]))
            ocr_centrality_to_cluster = float(np.mean([ocr_sim[idx, j] for j in selected_indices if j != idx and float(ocr_sim[idx, j]) > 0.0] or [0.0]))
        else:
            centrality_to_cluster = 1.0
            ocr_centrality_to_cluster = 1.0 if candidates[idx].ocr_text else 0.0
        c = candidates[idx]
        ev_bonus = 0.0
        if c.price:
            ev_bonus += 0.035
        if c.product_norm:
            ev_bonus += 0.025
        if c.ocr_text:
            ev_bonus += 0.025 * max(0.0, min(1.0, ocr_centrality_to_cluster))
        focus_bonus = float(focus_score_weight) * max(0.0, min(1.0, float(c.focus_norm)))
        ref_score = float(c.original_score) * (0.48 + 0.34 * centrality_to_cluster + 0.09 * max(0.0, min(1.0, ocr_centrality_to_cluster)) + 0.09 * max(0.0, min(1.0, float(c.focus_norm)))) + ev_bonus + focus_bonus
        if ref_score > best_ref_score:
            best_ref_score = ref_score
            ref_idx = idx

    for idx, c in enumerate(candidates):
        if len(clusters_idx[c.cluster_id]) >= 2:
            c.cluster_centrality = float(np.mean([sim[idx, j] for j in clusters_idx[c.cluster_id] if j != idx]))
        else:
            c.cluster_centrality = 1.0 if c.cluster_id == selected_cluster_id else 0.0
        c.sim_to_reference = float(sim[idx, ref_idx]) if 0 <= ref_idx < n else 0.0
        c.ocr_sim_to_reference = float(ocr_sim[idx, ref_idx]) if 0 <= ref_idx < n else 0.0
        if len(clusters_idx[c.cluster_id]) >= 2:
            c.ocr_cluster_centrality = float(np.mean([ocr_sim[idx, j] for j in clusters_idx[c.cluster_id] if j != idx and float(ocr_sim[idx, j]) > 0.0] or [0.0]))
            c.ocr_knn_support = float(np.mean([float(ocr_knn[idx, j]) for j in clusters_idx[c.cluster_id] if j != idx] or [0.0]))
        else:
            c.ocr_cluster_centrality = 1.0 if c.cluster_id == selected_cluster_id and c.ocr_text else 0.0
            c.ocr_knn_support = 1.0 if c.cluster_id == selected_cluster_id and c.ocr_text else 0.0
        evidence_bonus = 0.0
        if c.price and _dominant_value([m.price for m in selected_members if m.price]) == c.price:
            evidence_bonus += 0.05
        if c.product_norm and _dominant_value([m.product_norm for m in selected_members if m.product_norm]) == c.product_norm:
            evidence_bonus += 0.04
        if c.ocr_text:
            evidence_bonus += 0.035 * max(c.ocr_sim_to_reference, c.ocr_cluster_centrality)
        c.fusion_score = float(c.original_score) * (0.44 + 0.25 * c.sim_to_reference + 0.14 * c.cluster_centrality + 0.07 * max(c.ocr_sim_to_reference, c.ocr_cluster_centrality) + 0.10 * max(0.0, min(1.0, c.focus_norm))) + evidence_bonus + float(focus_score_weight) * 0.18 * max(0.0, min(1.0, c.focus_norm))

    selected_size = len(selected_indices)
    status = "ok" if selected_size >= int(min_cluster_size) else "no_cluster_above_min_size"
    return candidates, {
        "enabled": True,
        "status": status,
        "candidate_count": int(len(candidates)),
        "unreadable_count": int(unreadable),
        "cluster_count": int(len(clusters_idx)),
        "selected_cluster_id": int(selected_cluster_id),
        "selected_cluster_size": int(selected_size),
        "selected_cluster_score": round(float(selected_score), 4),
        "reference_frame_index": int(candidates[ref_idx].obs.frame_index),
        "reference_observation_id": str(candidates[ref_idx].obs.observation_id),
        "visual_similarity_threshold": float(visual_similarity_threshold),
        "evidence_similarity_threshold": float(evidence_similarity_threshold),
        "min_cluster_size": int(min_cluster_size),
        "ocr_similarity_enabled": bool(ocr_similarity_enabled),
        "ocr_embedding_dim": int(ocr_embedding_dim),
        "ocr_knn_k": int(ocr_knn_k),
        "ocr_similarity_threshold": float(ocr_similarity_threshold),
        "ocr_strong_similarity_threshold": float(ocr_strong_similarity_threshold),
        "ocr_gzip_weight": float(ocr_gzip_weight),
        "focus_selection_enabled": bool(focus_selection_enabled),
        "focus_score_weight": float(focus_score_weight),
        "focus_roi_policy": str(focus_roi_policy),
        "focus_summary": summarize_focus([c.focus_metrics or {"score": c.focus_score, "qr_score": 0.0} for c in candidates]),
        "clusters": cluster_infos[:10],
    }


def _cluster_meta(
    cluster_id: int,
    members: Sequence[_FrameCandidate],
    *,
    centrality: float | None = None,
    cluster_score: float | None = None,
    ocr_centrality: float | None = None,
) -> dict[str, Any]:
    frame_indices = [int(m.obs.frame_index) for m in members]
    prices = [m.price for m in members if m.price]
    products = [m.product_norm for m in members if m.product_norm]
    templates = [m.template_name for m in members if m.template_name]
    out = {
        "cluster_id": int(cluster_id),
        "size": int(len(members)),
        "frames": frame_indices[:20],
        "score_sum": round(float(sum(max(0.03, m.original_score) for m in members)), 4),
        "best_score": round(float(max((m.original_score for m in members), default=0.0)), 4),
        "dominant_price": _dominant_value(prices),
        "dominant_product_norm": _dominant_value(products),
        "dominant_template": _dominant_value(templates),
        "price_ratio": round(float(_dominant_ratio(prices)), 4),
        "product_ratio": round(float(_dominant_ratio(products)), 4),
        "template_ratio": round(float(_dominant_ratio(templates)), 4),
        "focus_mean": round(float(np.mean([m.focus_norm for m in members])) if members else 0.0, 4),
        "focus_best": round(float(max((m.focus_norm for m in members), default=0.0)), 4),
        "qr_focus_best": round(float(max((float((m.focus_metrics or {}).get("qr_score", 0.0)) for m in members), default=0.0)), 4),
    }
    if centrality is not None:
        out["visual_centrality"] = round(float(centrality), 4)
    if cluster_score is not None:
        out["cluster_score"] = round(float(cluster_score), 4)
    if ocr_centrality is not None:
        out["ocr_centrality"] = round(float(ocr_centrality), 4)
    return out


def _read_observation_image(obs: TrackObservation) -> np.ndarray | None:
    candidates = []
    if obs.saved_crop:
        candidates.append(Path(str(obs.saved_crop).split("#", 1)[0]))
    if obs.image_path:
        candidates.append(Path(str(obs.image_path).split("#", 1)[0]))
    for p in candidates:
        if p.exists() and p.is_file():
            img = imread_unicode(p)
            if img is not None and img.size > 0:
                return img
    return None


def _image_feature(image: np.ndarray, *, feature_size: int = 64) -> np.ndarray | None:
    try:
        if image.ndim == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        side = max(16, int(feature_size))
        gray = cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA)
        gray = cv2.equalizeHist(gray)
        gray_f = gray.astype(np.float32) / 255.0
        gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        gray_f = gray_f - float(gray_f.mean())
        mag = mag - float(mag.mean())
        feat = np.concatenate([gray_f.reshape(-1) * 0.65, mag.reshape(-1) * 0.35]).astype(np.float32)
        norm = float(np.linalg.norm(feat))
        if norm < 1e-6:
            return None
        return feat / norm
    except Exception:
        return None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    try:
        return float(np.dot(a, b) / max(1e-6, float(np.linalg.norm(a) * np.linalg.norm(b))))
    except Exception:
        return 0.0



def _ocr_similarity_graph(
    candidates: Sequence[_FrameCandidate],
    *,
    enabled: bool,
    knn_k: int,
    similarity_threshold: float,
    gzip_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(candidates)
    sim = np.eye(n, dtype=np.float32)
    knn = np.zeros((n, n), dtype=np.uint8)
    if not enabled or n <= 0:
        return sim * 0.0, knn

    for i in range(n):
        sim[i, i] = 1.0 if candidates[i].ocr_text else 0.0
        for j in range(i + 1, n):
            s = _ocr_text_similarity(candidates[i], candidates[j], gzip_weight=float(gzip_weight))
            sim[i, j] = sim[j, i] = float(s)

    k = max(1, int(knn_k))
    thr = float(similarity_threshold)
    for i in range(n):
        order = [j for j in np.argsort(-sim[i]).tolist() if j != i and float(sim[i, j]) >= thr]
        for j in order[:k]:
            knn[i, j] = 1
            knn[j, i] = 1
    return sim, knn


def _ocr_text_similarity(a: _FrameCandidate, b: _FrameCandidate, *, gzip_weight: float) -> float:
    ta = _norm_ocr_text(a.ocr_text)
    tb = _norm_ocr_text(b.ocr_text)
    if not ta or not tb:
        return 0.0
    if ta == tb:
        return 1.0
    emb_sim = 0.0
    if a.ocr_feature is not None and b.ocr_feature is not None:
        emb_sim = max(0.0, _cosine(a.ocr_feature, b.ocr_feature))
    gz_sim = _gzip_text_similarity(ta, tb)
    w = max(0.0, min(1.0, float(gzip_weight)))
    # Max term preserves exact/near-exact repeated short OCR fragments; blended
    # term makes the metric less brittle to insertion/deletion OCR noise.
    return float(max(0.72 * emb_sim + 0.28 * gz_sim, (1.0 - w) * emb_sim + w * gz_sim))


def _ocr_embedding(text: str, *, dim: int = 384) -> np.ndarray | None:
    text = _norm_ocr_text(text)
    if not text:
        return None
    dim = max(64, int(dim))
    vec = np.zeros((dim,), dtype=np.float32)
    tokens = text.split()
    for tok in tokens:
        if len(tok) >= 2:
            vec[_stable_hash("tok:" + tok) % dim] += 0.60
        padded = "^" + tok + "$"
        for n in (2, 3, 4):
            if len(padded) < n:
                continue
            weight = 1.0 if n == 3 else 0.72
            for i in range(0, len(padded) - n + 1):
                vec[_stable_hash(f"{n}:" + padded[i : i + n]) % dim] += weight
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return None
    return vec / norm


def _gzip_text_similarity(a: str, b: str) -> float:
    a = _norm_ocr_text(a)
    b = _norm_ocr_text(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ca = _gzip_len(a)
    cb = _gzip_len(b)
    cab = _gzip_len(a + "\n" + b)
    denom = max(1, max(ca, cb))
    ncd = (cab - min(ca, cb)) / denom
    # GZip NCD on very short OCR strings is noisy because of container overhead.
    # Shift/scale and clamp so it behaves as a weak similarity, not a hard rule.
    sim = 1.0 - float(ncd)
    if min(len(a), len(b)) < 12:
        sim *= 0.72
    return max(0.0, min(1.0, sim))


def _gzip_len(text: str) -> int:
    try:
        return len(gzip.compress(text.encode("utf-8"), compresslevel=6))
    except Exception:
        return len(text.encode("utf-8"))


def _stable_hash(text: str) -> int:
    try:
        return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little", signed=False)
    except Exception:
        return abs(hash(text))

def _candidate_evidence_level(a: _FrameCandidate, b: _FrameCandidate) -> str:
    """Return strong/weak/no evidence agreement for threshold selection."""
    if a.price and b.price and a.price == b.price:
        return "strong"
    if a.product_norm and b.product_norm:
        overlap = _token_overlap(a.product_norm, b.product_norm)
        if overlap >= 0.55:
            return "strong"
        if overlap >= 0.34 and a.template_name and b.template_name and a.template_name == b.template_name:
            return "weak"
    if a.template_name and b.template_name and a.template_name == b.template_name:
        return "weak"
    return ""


def _candidate_evidence_conflicts(a: _FrameCandidate, b: _FrameCandidate) -> bool:
    if a.price and b.price and a.price != b.price:
        return True
    if a.product_norm and b.product_norm and _token_overlap(a.product_norm, b.product_norm) < 0.22:
        # Product conflict alone is noisy, but with different templates it is a
        # good indication that visually similar crops are not the same label.
        if a.template_name and b.template_name and a.template_name != b.template_name:
            return True
    return False



def _obs_ocr_text(obs: TrackObservation) -> str:
    parts: list[str] = []
    raw = obs.raw_summary or {}
    for key in ("ocr_all_text_joined", "stock_status_text"):
        v = raw.get(key) if isinstance(raw, Mapping) else None
        if v:
            parts.append(str(v))
    final = obs.final or {}
    if isinstance(final, Mapping):
        for key in ("product_name", "stock_status_text", "promo_condition", "unit"):
            v = final.get(key)
            if v:
                parts.append(str(v))
    product = _obs_product(obs)
    if product:
        parts.append(product)
    price = _obs_price(obs)
    if price:
        # Price is useful for candidate support, but it must not dominate the
        # text metric.  Prefixing it makes it a separate token family.
        parts.append("price_" + price)
    text = " | ".join(parts)
    return _norm_ocr_text(text)


def _norm_ocr_text(value: Any) -> str:
    s = str(value or "").lower().replace("ё", "е")
    s = re.sub(r"[|/\\]+", " ", s)
    s = re.sub(r"[^0-9a-zа-я.%+\-]+", " ", s, flags=re.I)
    # OCR often produces repeated one-letter garbage around QR/noisy zones.
    toks = []
    for t in s.split():
        if len(t) == 1 and not t.isdigit() and t not in {"г", "к"}:
            continue
        if t in {"qr", "ean", "код", "руб", "р", "коп"}:
            continue
        toks.append(t)
    return re.sub(r"\s+", " ", " ".join(toks)).strip()


def _obs_price(obs: TrackObservation) -> str:
    for v in [
        (obs.final or {}).get("main_price") if isinstance(obs.final, Mapping) else None,
        ((obs.prices or {}).get("main") or {}).get("value") if isinstance((obs.prices or {}).get("main"), Mapping) else None,
        ((obs.raw_summary or {}).get("price_only_candidate") or {}).get("value") if isinstance((obs.raw_summary or {}).get("price_only_candidate"), Mapping) else None,
    ]:
        p = normalize_price_value(v)
        if p:
            return p
    return ""


def _obs_product(obs: TrackObservation) -> str:
    final = obs.final or {}
    if isinstance(final, Mapping):
        v = str(final.get("product_name") or "").strip()
        if v:
            return v
        pm = final.get("product_match") if isinstance(final.get("product_match"), Mapping) else {}
        v = str(pm.get("catalog_name") or "").strip()
        if v:
            return v
    cc = obs.csv_correction or {}
    if isinstance(cc.get("selected"), Mapping):
        v = str(cc["selected"].get("name") or "").strip()
        if v:
            return v
    return ""


def _obs_template(obs: TrackObservation) -> str:
    t = obs.template or {}
    return str(t.get("template_name") or t.get("name") or "").strip()


def _norm_text(value: Any) -> str:
    s = str(value or "").lower().replace("ё", "е")
    s = re.sub(r"[^0-9a-zа-я]+", " ", s, flags=re.I).strip()
    return re.sub(r"\s+", " ", s)


def _token_overlap(a: str, b: str) -> float:
    ta = set(x for x in _norm_text(a).split() if len(x) >= 2)
    tb = set(x for x in _norm_text(b).split() if len(x) >= 2)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def _dominant_value(values: Sequence[str]) -> str:
    counts: dict[str, int] = {}
    for v in values:
        if v:
            counts[v] = counts.get(v, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda x: (x[1], len(x[0])))[0]


def _dominant_ratio(values: Sequence[str]) -> float:
    vals = [v for v in values if v]
    if not vals:
        return 0.0
    dom = _dominant_value(vals)
    return sum(1 for v in vals if v == dom) / max(1, len(vals))


def _downscale_if_needed(image: np.ndarray, *, max_work_side: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    max_side = max(64, int(max_work_side))
    side = max(h, w)
    if side <= max_side:
        return image, 1.0
    scale = max_side / max(1, side)
    resized = cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    return resized, scale


def _alignment_gray(image: np.ndarray) -> np.ndarray:
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        return gray.astype(np.float32) / 255.0
    except Exception:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        return gray.astype(np.float32) / 255.0


def _align_image_to_reference(
    image: np.ndarray,
    ref_gray: np.ndarray,
    *,
    mode: str = "phase_ecc",
    ecc_motion: str = "euclidean",
    ecc_min_correlation: float = 0.18,
    phase_min_response: float = 0.08,
    max_translation_ratio: float = 0.22,
) -> tuple[bool, np.ndarray, dict[str, Any]]:
    info: dict[str, Any] = {"mode": str(mode), "ecc_motion": str(ecc_motion)}
    try:
        img_gray = _alignment_gray(image)
        h, w = ref_gray.shape[:2]
        if img_gray.shape[:2] != (h, w):
            img_gray = cv2.resize(img_gray, (w, h), interpolation=cv2.INTER_AREA)
        mode_l = str(mode or "phase_ecc").lower().strip()
        motion_l = str(ecc_motion or "euclidean").lower().strip()

        warp = _identity_warp(motion_l)
        phase_ok = False
        if mode_l in {"phase", "phase_ecc", "phase+ecc", "ecc_phase"}:
            try:
                win = cv2.createHanningWindow((w, h), cv2.CV_32F)
                (dx, dy), response = cv2.phaseCorrelate(ref_gray.astype(np.float32), img_gray.astype(np.float32), win)
                info["phase_dx"] = round(float(dx), 4)
                info["phase_dy"] = round(float(dy), 4)
                info["phase_response"] = round(float(response), 6)
                max_shift = float(max_translation_ratio) * float(max(h, w))
                if abs(float(dx)) <= max_shift and abs(float(dy)) <= max_shift and float(response) >= float(phase_min_response):
                    warp[0, 2] = float(dx)
                    warp[1, 2] = float(dy)
                    phase_ok = True
            except Exception as e:
                info["phase_error"] = type(e).__name__

        if mode_l == "phase":
            if not phase_ok:
                return False, image, info
            return True, _warp_with_matrix(image, warp, motion_l, (w, h)), info

        if mode_l in {"none", "off", "disabled"}:
            return True, image, info

        motion = _cv_motion(motion_l)
        if motion == cv2.MOTION_HOMOGRAPHY and warp.shape != (3, 3):
            w3 = np.eye(3, dtype=np.float32)
            w3[:2, :] = warp[:2, :]
            warp = w3
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
        cc, warp = cv2.findTransformECC(ref_gray, img_gray, warp.astype(np.float32), motion, criteria, None, 5)
        info["ecc_correlation"] = round(float(cc), 6)
        if float(cc) < float(ecc_min_correlation):
            if phase_ok:
                info["fallback"] = "phase_only_low_ecc"
                return True, _warp_with_matrix(image, _identity_warp("translation", tx=info.get("phase_dx", 0.0), ty=info.get("phase_dy", 0.0)), "translation", (w, h)), info
            return False, image, info
        return True, _warp_with_matrix(image, warp, motion_l, (w, h)), info
    except Exception as e:
        info["error"] = type(e).__name__
        return False, image, info


def _identity_warp(motion_l: str, *, tx: float = 0.0, ty: float = 0.0) -> np.ndarray:
    if str(motion_l).lower().strip() == "homography":
        w = np.eye(3, dtype=np.float32)
        w[0, 2] = float(tx)
        w[1, 2] = float(ty)
        return w
    w = np.eye(2, 3, dtype=np.float32)
    w[0, 2] = float(tx)
    w[1, 2] = float(ty)
    return w


def _cv_motion(motion_l: str) -> int:
    motion_l = str(motion_l or "euclidean").lower().strip()
    if motion_l in {"translation", "trans"}:
        return cv2.MOTION_TRANSLATION
    if motion_l in {"affine"}:
        return cv2.MOTION_AFFINE
    if motion_l in {"homography", "perspective"}:
        return cv2.MOTION_HOMOGRAPHY
    return cv2.MOTION_EUCLIDEAN


def _warp_with_matrix(image: np.ndarray, warp: np.ndarray, motion_l: str, size: tuple[int, int]) -> np.ndarray:
    w, h = int(size[0]), int(size[1])
    if warp.shape == (3, 3) or str(motion_l).lower().strip() == "homography":
        mat = warp if warp.shape == (3, 3) else np.vstack([warp, np.array([[0, 0, 1]], dtype=np.float32)])
        return cv2.warpPerspective(image, mat, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    return cv2.warpAffine(image, warp[:2, :], (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


def _obs_focus_norm(obs: TrackObservation, *, default: float = 0.5) -> float:
    try:
        vc = (obs.raw_summary or {}).get("visual_consensus") if isinstance(obs.raw_summary, Mapping) else None
        if isinstance(vc, Mapping) and vc.get("focus_norm") is not None:
            return max(0.0, min(1.0, float(vc.get("focus_norm"))))
    except Exception:
        pass
    return float(default)


def _assign_focus_norms(candidates: Sequence[_FrameCandidate]) -> None:
    vals = normalize_focus_scores([c.focus_metrics or {"score": c.focus_score} for c in candidates], key="score")
    for c, v in zip(candidates, vals):
        c.focus_norm = float(max(0.0, min(1.0, v)))


def _safe_nlmeans(
    image: np.ndarray,
    *,
    h: float,
    h_color: float,
    template_window_size: int,
    search_window_size: int,
) -> np.ndarray:
    try:
        tw = max(3, int(template_window_size) | 1)
        sw = max(tw + 2, int(search_window_size) | 1)
        return cv2.fastNlMeansDenoisingColored(image, None, float(h), float(h_color), tw, sw)
    except Exception:
        return image


def _unsharp(image: np.ndarray) -> np.ndarray:
    try:
        blur = cv2.GaussianBlur(image, (0, 0), 1.0)
        return cv2.addWeighted(image, 1.25, blur, -0.25, 0)
    except Exception:
        return image


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default
