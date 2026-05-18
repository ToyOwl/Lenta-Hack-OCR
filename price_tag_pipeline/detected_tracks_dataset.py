"""
Runner for already-detected price-tag tracks.

Input layout:

    root_dir/
      sequence-name/
        track_id/
          frame_0001.jpg
          frame_0002.jpg
          ...

This module treats every ``sequence-name/track_id`` directory as an already
known track.  It runs the existing single-tag OCR pipeline for every image in
that track, performs weighted track-level aggregation, selects the best frame,
and writes:

- ``debug_images/`` with the best frame per track and aggregated OCR overlay;
- ``detected_tracks_results.json`` with full track-level run results;
- ``detected_tracks_summary.tsv`` and ``detected_tracks_summary.csv`` tables;
- ``detected_tracks_frames.tsv`` with per-frame compact diagnostics.

The runner is intended for offline test sets exported from a detector/tracker,
where the detector has already cropped a candidate price tag and preserved the
track grouping in the filesystem.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import sys

import cv2
import numpy as np
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from .code_decoder import CodeDecoder
from .config import deep_merge, dump_config
from .debug_vis import _draw_text_pil, truncate_text
from .io_utils import IMG_EXTS, imread_unicode, imwrite_unicode, safe_stem
from .layout import HeuristicLayoutExtractor
from .ocr_backends import OCRBackend
from .original_coordinates import OriginalCoordinateMap, build_original_coordinate_map
from .pipeline import PriceTagPipeline
from .template_classifier import ColorNameTemplateClassifier
from .track_aggregator import (
    PriceTagTrackAggregator,
    TrackObservation,
    TrackState,
    extract_observations,
)
from .track_debug_plots import write_global_debug_plots, write_track_debug_plots
from .track_image_fusion import apply_track_consensus_scores, fuse_track_images, has_decoded_code
from .tag_qr_enhancement import build_fused_tag_variants
from .submission_output import TASK_OUTPUT_FIELDS, build_task_output_record, write_task_outputs


@dataclass(frozen=True)
class DetectionTrackFolder:
    sequence_name: str
    track_id: str
    track_dir: Path
    images: List[Path]

    @property
    def track_key(self) -> str:
        return f"{self.sequence_name}/{self.track_id}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["track_dir"] = str(self.track_dir)
        d["images"] = [str(p) for p in self.images]
        d["track_key"] = self.track_key
        return d


@dataclass
class FrameRunRecord:
    sequence_name: str
    source_track_id: str
    track_key: str
    frame_index: int
    image_path: str
    status: str
    item_stem: str
    debug_image_path: str
    score: float = 0.0
    main_price: str = ""
    product_name: str = ""
    unit: str = ""
    csv_status: str = ""
    template_name: str = ""
    template_confidence: Any = ""
    quality_status: str = ""
    glare_applied: Any = ""
    glare_method: str = ""
    needs_review: Any = ""
    bbox_source: str = ""
    x_min: Any = ""
    y_min: Any = ""
    x_max: Any = ""
    y_max: Any = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrackTableRecord:
    sequence_name: str
    source_track_id: str
    track_key: str
    source_track_dir: str
    status: str
    num_images: int
    num_observations: int
    frame_span: str
    best_image: str
    best_debug_image: str
    best_score: Any
    main_price: Any
    product_name: Any
    unit: Any
    item_id: Any
    csv_status: str
    needs_review: Any
    split_index: Any = ""
    split_count: Any = ""
    split_reason: str = ""
    price_consistency: Any = ""
    price_unique_count: Any = ""
    product_consistency: Any = ""
    validation_score: Any = ""
    glare_applied_best: Any = ""
    glare_method_best: str = ""
    debug_timeline_plot: str = ""
    debug_votes_plot: str = ""
    best_fused_image: str = ""
    fused_code_fallback_decoded: Any = ""
    fused_code_fallback_attempts: Any = ""
    stock_status: Any = ""
    catalog_gate_rejected: Any = ""
    catalog_reject_reasons: str = ""
    warnings: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def iter_detection_track_folders(
    root_dir: Path,
    *,
    recursive_images_in_track: bool = False,
    min_images_per_track: int = 1,
    skip_hidden: bool = True,
) -> List[DetectionTrackFolder]:
    """Enumerate ``root_dir/sequence-name/track_id/{images}`` folders."""
    root_dir = Path(root_dir)
    if not root_dir.exists() or not root_dir.is_dir():
        raise FileNotFoundError(f"detected-tracks root_dir does not exist or is not a directory: {root_dir}")

    tracks: List[DetectionTrackFolder] = []
    for seq_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        if skip_hidden and seq_dir.name.startswith("."):
            continue
        for track_dir in sorted([p for p in seq_dir.iterdir() if p.is_dir()]):
            if skip_hidden and track_dir.name.startswith("."):
                continue
            images = _iter_track_images(track_dir, recursive=recursive_images_in_track)
            if len(images) < int(min_images_per_track):
                continue
            tracks.append(
                DetectionTrackFolder(
                    sequence_name=seq_dir.name,
                    track_id=track_dir.name,
                    track_dir=track_dir,
                    images=images,
                )
            )
    return tracks


def process_detected_tracks_dataset(
    cfg: Mapping[str, Any],
    *,
    template_classifier: ColorNameTemplateClassifier,
    layout_detector: HeuristicLayoutExtractor,
    ocr_backend: OCRBackend,
    code_decoder: CodeDecoder,
) -> Dict[str, Any]:
    """Run OCR aggregation for a detected-track dataset."""
    dt_cfg = cfg.get("detected_tracks_dataset", {}) if isinstance(cfg, Mapping) else {}
    root_dir = Path(str(dt_cfg.get("root_dir") or cfg.get("io", {}).get("input_dir") or "")).expanduser()
    if not str(root_dir):
        raise ValueError("detected_tracks_dataset.root_dir or io.input_dir must be set")

    out_dir = Path(str(dt_cfg.get("out_dir") or cfg.get("io", {}).get("out_dir") or "runs/detected_tracks_dataset")).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    recursive_images = bool(dt_cfg.get("recursive_images_in_track", False))
    min_images_per_track = int(dt_cfg.get("min_images_per_track", 1))
    frame_index_cfg = _frame_index_cfg(dt_cfg)
    original_coord_map = build_original_coordinate_map(dt_cfg, root_dir=root_dir)
    copy_best_debug = bool(dt_cfg.get("copy_best_debug", True))
    keep_items = bool(dt_cfg.get("keep_items", False))
    keep_tracks_dir = bool(dt_cfg.get("keep_tracks_dir", False))
    write_frame_table = bool(dt_cfg.get("write_frame_table", True))
    write_debug_plots = bool(dt_cfg.get("write_debug_plots", True))
    task_output_cfg = _task_output_cfg(cfg, dt_cfg)
    track_fusion_cfg = dt_cfg.get("track_fusion", {}) if isinstance(dt_cfg.get("track_fusion"), Mapping) else {}
    track_fusion_enabled = bool(track_fusion_cfg.get("enabled", True))
    debug_dir_name = str(dt_cfg.get("debug_images_dir_name", "debug_images") or "debug_images")
    debug_images_dir = out_dir / debug_dir_name
    debug_plots_dir = out_dir / str(dt_cfg.get("debug_plots_dir_name", "debug_plots") or "debug_plots")
    if copy_best_debug or track_fusion_enabled:
        debug_images_dir.mkdir(parents=True, exist_ok=True)
    if write_debug_plots:
        debug_plots_dir.mkdir(parents=True, exist_ok=True)

    runtime_cfg = _runtime_cfg_for_detected_tracks(cfg, dt_cfg)
    if not keep_tracks_dir:
        runtime_cfg.setdefault("track_aggregation", {})["copy_best_frames"] = False

    tracks = iter_detection_track_folders(
        root_dir,
        recursive_images_in_track=recursive_images,
        min_images_per_track=min_images_per_track,
        skip_hidden=bool(dt_cfg.get("skip_hidden", True)),
    )
    if not tracks:
        raise ValueError(f"no detection-track folders found under {root_dir}; expected root_dir/sequence-name/track_id/{{images}}")

    with open(out_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        f.write(dump_config(runtime_cfg))

    pipeline = PriceTagPipeline(
        template_classifier=template_classifier,
        layout_detector=layout_detector,
        ocr_backend=ocr_backend,
        code_decoder=code_decoder,
        config=runtime_cfg,
    )
    sr_recovery_cfg = _sr_recovery_cfg(runtime_cfg, dt_cfg)
    recovery_pipeline: Optional[PriceTagPipeline] = None
    if bool(sr_recovery_cfg.get("enabled", False)):
        recovery_runtime_cfg = deep_merge(runtime_cfg, dict(sr_recovery_cfg.get("config_overrides") or {}))
        recovery_pipeline = PriceTagPipeline(
            template_classifier=template_classifier,
            layout_detector=layout_detector,
            ocr_backend=ocr_backend,
            code_decoder=code_decoder,
            config=recovery_runtime_cfg,
        )
    aggregator = PriceTagTrackAggregator.from_config({**dict(runtime_cfg), "track_aggregation": _force_track_aggregation_enabled(runtime_cfg)})

    results: List[Dict[str, Any]] = []
    table_rows: List[TrackTableRecord] = []
    frame_rows: List[FrameRunRecord] = []
    errors: List[Dict[str, Any]] = []

    iterator: Iterable[DetectionTrackFolder] = tracks
    if tqdm is not None:
        iterator = tqdm(tracks, desc="detected-tracks")

    for numeric_track_id, track_folder in enumerate(iterator):
        track_state = TrackState(track_id=numeric_track_id)
        frame_records_for_track: List[FrameRunRecord] = []
        indexed_images = _indexed_track_images(track_folder.images, frame_index_cfg)
        for frame_index, image_path in indexed_images:
            item_stem = _make_item_stem(track_folder.sequence_name, track_folder.track_id, frame_index, image_path)
            try:
                result, obs_list, frame_record = _process_track_image(
                    pipeline=pipeline,
                    cfg=runtime_cfg,
                    image_path=image_path,
                    out_dir=out_dir,
                    item_stem=item_stem,
                    track_folder=track_folder,
                    frame_index=frame_index,
                    best_blur_reference=aggregator.best_blur_reference,
                    original_coord_map=original_coord_map,
                )
                for obs in obs_list:
                    track_state.add(obs)
                frame_rows.append(frame_record)
                frame_records_for_track.append(frame_record)
            except Exception as e:
                err = {
                    "sequence_name": track_folder.sequence_name,
                    "source_track_id": track_folder.track_id,
                    "track_key": track_folder.track_key,
                    "frame_index": frame_index,
                    "image_path": str(image_path),
                    "error": repr(e),
                }
                errors.append(err)
                frame_record = FrameRunRecord(
                    sequence_name=track_folder.sequence_name,
                    source_track_id=track_folder.track_id,
                    track_key=track_folder.track_key,
                    frame_index=frame_index,
                    image_path=str(image_path),
                    status="error",
                    item_stem=item_stem,
                    debug_image_path="",
                    error=repr(e),
                )
                frame_rows.append(frame_record)
                frame_records_for_track.append(frame_record)

        if track_state.observations and recovery_pipeline is not None and bool(sr_recovery_cfg.get("enabled", False)):
            recovery_info = _maybe_run_track_sr_recovery(
                recovery_pipeline=recovery_pipeline,
                cfg=deep_merge(runtime_cfg, dict(sr_recovery_cfg.get("config_overrides") or {})),
                sr_recovery_cfg=sr_recovery_cfg,
                track_state=track_state,
                track_folder=track_folder,
                indexed_images=indexed_images,
                out_dir=out_dir,
                best_blur_reference=aggregator.best_blur_reference,
                frame_records_for_track=frame_records_for_track,
                global_frame_rows=frame_rows,
                original_coord_map=original_coord_map,
            )
            if recovery_info.get("attempted"):
                track_state.warnings.append("sr_recovery_attempted")
                if recovery_info.get("added_observations", 0):
                    track_state.warnings.append("sr_recovery_added_observations")

        if track_state.observations:
            split_states = aggregator.split_track_state(track_state)
            for split_index, (sub_state, split_meta) in enumerate(split_states):
                split_count = len(split_states)
                split_track_id = track_folder.track_id if split_count == 1 else f"{track_folder.track_id}__split_{split_index:02d}"
                split_folder = DetectionTrackFolder(
                    sequence_name=track_folder.sequence_name,
                    track_id=split_track_id,
                    track_dir=track_folder.track_dir,
                    images=track_folder.images,
                )
                visual_consensus_info: Dict[str, Any] = {}
                if bool(track_fusion_cfg.get("score_consensus_enabled", track_fusion_enabled)):
                    visual_consensus_info = apply_track_consensus_scores(
                        sub_state.observations,
                        enabled=True,
                        max_candidates=int(track_fusion_cfg.get("consensus_max_candidates", 32)),
                        feature_size=int(track_fusion_cfg.get("consensus_feature_size", 64)),
                        visual_similarity_threshold=float(track_fusion_cfg.get("visual_similarity_threshold", 0.66)),
                        evidence_similarity_threshold=float(track_fusion_cfg.get("evidence_similarity_threshold", 0.50)),
                        min_cluster_size=int(track_fusion_cfg.get("min_cluster_size", 2)),
                        ocr_similarity_enabled=bool(track_fusion_cfg.get("ocr_similarity_enabled", True)),
                        ocr_embedding_dim=int(track_fusion_cfg.get("ocr_embedding_dim", 384)),
                        ocr_knn_k=int(track_fusion_cfg.get("ocr_knn_k", 4)),
                        ocr_similarity_threshold=float(track_fusion_cfg.get("ocr_similarity_threshold", 0.56)),
                        ocr_strong_similarity_threshold=float(track_fusion_cfg.get("ocr_strong_similarity_threshold", 0.70)),
                        ocr_gzip_weight=float(track_fusion_cfg.get("ocr_gzip_weight", 0.55)),
                        focus_selection_enabled=bool(track_fusion_cfg.get("focus_selection_enabled", True)),
                        focus_score_weight=float(track_fusion_cfg.get("focus_score_weight", 0.18)),
                        focus_roi_policy=str(track_fusion_cfg.get("focus_roi_policy", "price_tag")),
                        selected_score_boost=float(track_fusion_cfg.get("selected_score_boost", 0.22)),
                        outlier_score_penalty=float(track_fusion_cfg.get("outlier_score_penalty", 0.32)),
                    )

                aggregated = aggregator._aggregate_track(sub_state, out_dir=out_dir)  # noqa: SLF001 - public runner, same package
                if visual_consensus_info:
                    aggregated["visual_consensus"] = visual_consensus_info
                if getattr(sub_state, "warnings", None):
                    aggregated.setdefault("warnings", [])
                    aggregated["warnings"] = list(dict.fromkeys(list(aggregated.get("warnings") or []) + list(sub_state.warnings)))
                aggregated["sequence_name"] = track_folder.sequence_name
                aggregated["source_track_id"] = split_track_id
                aggregated["source_track_id_original"] = track_folder.track_id
                aggregated["track_key"] = split_folder.track_key
                aggregated["source_track_dir"] = str(track_folder.track_dir)
                aggregated["num_images"] = len(track_folder.images)
                aggregated["split"] = {**dict(split_meta), "split_index": split_index, "split_count": split_count}
                if split_count > 1:
                    aggregated.setdefault("warnings", [])
                    aggregated["warnings"] = list(dict.fromkeys(list(aggregated.get("warnings") or []) + ["source_track_was_split_by_aggregation"]))

                best_debug = ""
                best_obs = _find_observation_for_aggregated_best(sub_state.observations, aggregated)
                if track_fusion_enabled:
                    fused_info = _write_track_fusion_image(
                        observations=sub_state.observations,
                        reference_observation=best_obs,
                        aggregated_track=aggregated,
                        debug_images_dir=debug_images_dir,
                        track_folder=split_folder,
                        code_decoder=code_decoder,
                        cfg=track_fusion_cfg,
                    )
                    if fused_info:
                        aggregated["fused_image"] = fused_info
                if copy_best_debug:
                    best_debug = _write_best_debug_image(
                        obs=best_obs,
                        aggregated_track=aggregated,
                        debug_images_dir=debug_images_dir,
                        track_folder=split_folder,
                        max_image_side=int(dt_cfg.get("debug_max_image_side", 520)),
                        upscale_small=bool(dt_cfg.get("debug_upscale_small", True)),
                    )
                    aggregated["best_debug_image"] = best_debug

                if write_debug_plots:
                    plot_prefix = f"{_safe_name(track_folder.sequence_name)}__{_safe_name(split_track_id)}"
                    plot_paths = write_track_debug_plots(aggregated, debug_plots_dir, name_prefix=plot_prefix)
                    aggregated["debug_plots"] = plot_paths
                else:
                    plot_paths = {}

                frame_idx_set = {int(o.frame_index) for o in sub_state.observations}
                aggregated["frame_records"] = [r.to_dict() for r in frame_records_for_track if int(r.frame_index) in frame_idx_set]
                table_row = _track_table_row(split_folder, aggregated, best_debug)
                results.append(aggregated)
                table_rows.append(table_row)
        else:
            aggregated = {
                "sequence_name": track_folder.sequence_name,
                "source_track_id": track_folder.track_id,
                "source_track_id_original": track_folder.track_id,
                "track_key": track_folder.track_key,
                "source_track_dir": str(track_folder.track_dir),
                "status": "error",
                "num_images": len(track_folder.images),
                "num_observations": 0,
                "warnings": ["no_successful_observations"],
                "observations": [],
                "frame_records": [r.to_dict() for r in frame_records_for_track],
            }
            table_row = _track_table_row(track_folder, aggregated, "")
            results.append(aggregated)
            table_rows.append(table_row)

    output: Dict[str, Any] = {
        "status": "ok" if not errors else "ok_with_errors",
        "root_dir": str(root_dir),
        "out_dir": str(out_dir),
        "debug_images_dir": str(debug_images_dir) if (copy_best_debug or track_fusion_enabled) else "",
        "debug_plots_dir": str(debug_plots_dir) if write_debug_plots else "",
        "keep_items": keep_items,
        "keep_tracks_dir": keep_tracks_dir,
        "sequence_count": len({t.sequence_name for t in tracks}),
        "track_folder_count": len(tracks),
        "processed_track_count": len(results),
        "processed_frame_count": len(frame_rows),
        "error_count": len(errors),
        "tracks": results,
        "errors": errors,
        "original_coordinates": original_coord_map.describe(),
    }

    if write_debug_plots:
        output["global_debug_plots"] = write_global_debug_plots(results, debug_plots_dir)

    _write_outputs(
        output,
        table_rows,
        frame_rows if write_frame_table else [],
        out_dir,
        write_frame_table=write_frame_table,
        task_output_cfg=task_output_cfg,
    )
    _cleanup_optional_output_dirs(out_dir, keep_items=keep_items, keep_tracks_dir=keep_tracks_dir)
    return output


def _process_track_image(
    *,
    pipeline: PriceTagPipeline,
    cfg: Mapping[str, Any],
    image_path: Path,
    out_dir: Path,
    item_stem: str,
    track_folder: DetectionTrackFolder,
    frame_index: int,
    best_blur_reference: float,
    original_coord_map: Optional[OriginalCoordinateMap] = None,
) -> Tuple[Dict[str, Any], List[TrackObservation], FrameRunRecord]:
    image = imread_unicode(image_path)
    if image is None:
        raise RuntimeError("failed_to_read_image")

    crop_h, crop_w = image.shape[:2]
    fallback_bbox = [0, 0, max(1, int(crop_w)), max(1, int(crop_h))]
    coord_result = (
        original_coord_map.lookup(
            frame_idx=frame_index,
            tr_id=track_folder.track_id,
            sequence_name=track_folder.sequence_name,
            fallback_bbox=fallback_bbox,
        )
        if original_coord_map is not None
        else None
    )
    original_bbox = list(coord_result.bbox) if coord_result is not None else list(fallback_bbox)
    original_bbox_source = str(coord_result.source) if coord_result is not None else "crop_bbox_no_map"

    image_out_dir = out_dir / "items" / item_stem
    image_out_dir.mkdir(parents=True, exist_ok=True)

    if bool(cfg.get("tilt_corrector", {}).get("enabled", False)):
        image_proc, tilt_meta = pipeline._apply_tilt_if_needed(image, image_out_dir=image_out_dir, stem=item_stem)  # noqa: SLF001
        tilt_meta["stage"] = "detected_track_single_tag"
    else:
        image_proc = image
        tilt_meta = {"enabled": False, "applied": False, "angle": 0.0, "stage": "detected_track_single_tag"}

    extra_meta = {
        "tilt": tilt_meta,
        "detected_track": {
            "sequence_name": track_folder.sequence_name,
            "track_id": track_folder.track_id,
            "track_key": track_folder.track_key,
            "track_dir": str(track_folder.track_dir),
            "frame_index": frame_index,
            "source_image_path": str(image_path),
            "crop_bbox": fallback_bbox,
            "original_bbox": original_bbox,
            "original_bbox_source": original_bbox_source,
        },
        "input_structure": {"mode": "single_tag", "reason": "detected_track_crop"},
    }
    result = pipeline._process_single_tag_array(  # noqa: SLF001
        image=image_proc,
        image_path=image_path,
        out_dir=out_dir,
        stem=item_stem,
        extra_meta=extra_meta,
        virtual_image_path=str(image_path),
    )
    result["_item_stem"] = item_stem
    debug_path = out_dir / "items" / item_stem / f"{item_stem}_debug.jpg"
    result["_debug_image_path"] = str(debug_path) if debug_path.exists() else ""

    observations = extract_observations(result, frame_index=frame_index, best_blur_reference=best_blur_reference)
    _apply_original_bbox_to_observations(
        observations,
        original_bbox=original_bbox,
        original_bbox_source=original_bbox_source,
        crop_bbox=fallback_bbox,
        cfg=cfg,
    )
    for i, obs in enumerate(observations):
        obs.observation_id = f"{safe_stem(Path(track_folder.sequence_name))}_{safe_stem(Path(track_folder.track_id))}_frame{frame_index:06d}_{i:02d}"
        obs.track_hint = f"sequence={track_folder.sequence_name} track={track_folder.track_id} frame={frame_index}"
        obs.raw_summary.update(
            {
                "sequence_name": track_folder.sequence_name,
                "source_track_id": track_folder.track_id,
                "track_key": track_folder.track_key,
                "track_dir": str(track_folder.track_dir),
                "item_stem": item_stem,
                "debug_image_path": result["_debug_image_path"],
                "crop_bbox": fallback_bbox,
                "original_bbox": original_bbox,
                "original_bbox_source": original_bbox_source,
            }
        )

    best_obs = observations[0] if observations else None
    final = best_obs.final if best_obs is not None else (result.get("final") if isinstance(result.get("final"), Mapping) else {})
    template = result.get("template") if isinstance(result.get("template"), Mapping) else {}
    quality = result.get("quality") if isinstance(result.get("quality"), Mapping) else {}
    csv_correction = result.get("csv_correction") if isinstance(result.get("csv_correction"), Mapping) else {}
    glare_meta = result.get("glare_suppression") if isinstance(result.get("glare_suppression"), Mapping) else {}
    frame_record = FrameRunRecord(
        sequence_name=track_folder.sequence_name,
        source_track_id=track_folder.track_id,
        track_key=track_folder.track_key,
        frame_index=frame_index,
        image_path=str(image_path),
        status=str(result.get("status", "")),
        item_stem=item_stem,
        debug_image_path=result["_debug_image_path"],
        score=float(best_obs.score) if best_obs is not None else 0.0,
        main_price=str(final.get("main_price") or ""),
        product_name=str(final.get("product_name") or ""),
        unit=str(final.get("unit") or ""),
        csv_status=str(csv_correction.get("status") or ""),
        template_name=str(template.get("template_name") or ""),
        template_confidence=template.get("confidence", ""),
        quality_status=str(quality.get("status") or ""),
        glare_applied=glare_meta.get("applied", ""),
        glare_method=str(glare_meta.get("method") or ""),
        needs_review=final.get("needs_review", ""),
        bbox_source=original_bbox_source,
        x_min=original_bbox[0],
        y_min=original_bbox[1],
        x_max=original_bbox[2],
        y_max=original_bbox[3],
        error=str(result.get("error") or ""),
    )
    return result, observations, frame_record




def _apply_original_bbox_to_observations(
    observations: Sequence[TrackObservation],
    *,
    original_bbox: Sequence[int],
    original_bbox_source: str,
    crop_bbox: Sequence[int],
    cfg: Mapping[str, Any],
) -> None:
    """Attach full-frame bbox to observations.

    For detected-track crop datasets one track directory usually represents one
    tag bbox in the original frame.  Therefore the default behavior is to replace
    every observation bbox with that full-frame tag bbox.  For future rail/cell
    modes ``coordinate_child_mode=translate_local`` can translate local child
    bboxes by the full-frame crop origin.
    """
    if not observations:
        return
    dt_cfg = cfg.get("detected_tracks_dataset", {}) if isinstance(cfg.get("detected_tracks_dataset"), Mapping) else {}
    ocfg = dt_cfg.get("original_coordinates", {}) if isinstance(dt_cfg.get("original_coordinates"), Mapping) else {}
    child_mode = str(ocfg.get("coordinate_child_mode", "track_bbox") or "track_bbox").lower()
    full = [int(round(float(x))) for x in list(original_bbox)[:4]]
    crop = [int(round(float(x))) for x in list(crop_bbox)[:4]]
    ox, oy = int(full[0]), int(full[1])
    for obs in observations:
        local_bbox = list(obs.bbox)
        obs.raw_summary.setdefault("local_crop_bbox", local_bbox)
        obs.raw_summary["crop_bbox"] = crop
        obs.raw_summary["original_bbox"] = full
        obs.raw_summary["original_bbox_source"] = original_bbox_source
        if child_mode in {"translate", "translate_local", "child"} and obs.mode != "single_tag":
            x1, y1, x2, y2 = [int(round(float(x))) for x in local_bbox[:4]]
            obs.bbox = [ox + x1, oy + y1, ox + x2, oy + y2]
            obs.raw_summary["original_bbox_mode"] = "translated_child_bbox"
        else:
            obs.bbox = list(full)
            obs.raw_summary["original_bbox_mode"] = "track_bbox"


def _sr_recovery_cfg(runtime_cfg: Mapping[str, Any], dt_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve track-level SR recovery config.

    The ordinary super_resolution trigger works before OCR and can only use zone
    class and crop geometry.  This recovery stage is deliberately track-level:
    it runs after the light pass and enables PaddleOCR SR only when the track has
    too few frames with a credible product-name / catalog match.
    """
    local = dt_cfg.get("sr_recovery", {}) if isinstance(dt_cfg.get("sr_recovery"), Mapping) else {}
    global_cfg = runtime_cfg.get("sr_recovery", {}) if isinstance(runtime_cfg.get("sr_recovery"), Mapping) else {}
    return deep_merge(dict(global_cfg), dict(local))


def _maybe_run_track_sr_recovery(
    *,
    recovery_pipeline: PriceTagPipeline,
    cfg: Mapping[str, Any],
    sr_recovery_cfg: Mapping[str, Any],
    track_state: TrackState,
    track_folder: DetectionTrackFolder,
    indexed_images: Sequence[Tuple[int, Path]],
    out_dir: Path,
    best_blur_reference: float,
    frame_records_for_track: List[FrameRunRecord],
    global_frame_rows: List[FrameRunRecord],
    original_coord_map: Optional[OriginalCoordinateMap] = None,
) -> Dict[str, Any]:
    reason = _track_needs_sr_recovery(track_state.observations, sr_recovery_cfg)
    if not reason.get("triggered", False):
        return {"enabled": True, "attempted": False, "reason": reason}

    max_frames = max(1, int(sr_recovery_cfg.get("max_recovery_frames", 3)))
    score_multiplier = float(sr_recovery_cfg.get("score_multiplier", 0.92))
    suffix = str(sr_recovery_cfg.get("item_suffix", "__srrec") or "__srrec")
    selected = _select_sr_recovery_frames(indexed_images, track_state.observations, sr_recovery_cfg, max_frames=max_frames)

    added_obs = 0
    frame_errors: List[Dict[str, Any]] = []
    frame_indices: List[int] = []
    for frame_index, image_path in selected:
        item_stem = _make_item_stem(track_folder.sequence_name, track_folder.track_id, frame_index, image_path) + suffix
        try:
            result, obs_list, frame_record = _process_track_image(
                pipeline=recovery_pipeline,
                cfg=cfg,
                image_path=image_path,
                out_dir=out_dir,
                item_stem=item_stem,
                track_folder=track_folder,
                frame_index=frame_index,
                best_blur_reference=best_blur_reference,
                original_coord_map=original_coord_map,
            )
            frame_record.item_stem = item_stem
            frame_record.track_key = f"{track_folder.track_key}#sr_recovery"
            global_frame_rows.append(frame_record)
            frame_records_for_track.append(frame_record)
            for i, obs in enumerate(obs_list):
                obs.observation_id = f"{obs.observation_id}_srrec_{i:02d}"
                obs.track_hint = f"{obs.track_hint} sr_recovery=1"
                obs.score = float(obs.score) * score_multiplier
                obs.raw_summary.update(
                    {
                        "sr_recovery": True,
                        "sr_recovery_reason": reason,
                        "sr_recovery_item_stem": item_stem,
                    }
                )
                track_state.add(obs)
                added_obs += 1
            frame_indices.append(int(frame_index))
        except Exception as e:
            frame_errors.append({"frame_index": int(frame_index), "image_path": str(image_path), "error": repr(e)})

    return {
        "enabled": True,
        "attempted": True,
        "reason": reason,
        "selected_frame_indices": frame_indices,
        "added_observations": added_obs,
        "errors": frame_errors,
    }


def _track_needs_sr_recovery(observations: Sequence[TrackObservation], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    min_good_product_frames = int(cfg.get("min_good_product_frames", 2))
    min_good_price_frames = int(cfg.get("min_good_price_frames", 0))
    trigger_if_product_missing = bool(cfg.get("trigger_if_product_missing", True))
    trigger_if_price_only = bool(cfg.get("trigger_if_price_only", True))
    min_obs = int(cfg.get("min_observations_before_recovery", 1))

    good_product_frames = sorted({int(o.frame_index) for o in observations if _is_good_product_observation(o, cfg)})
    good_price_frames = sorted({int(o.frame_index) for o in observations if _is_good_price_observation(o)})
    has_any_price = bool(good_price_frames)
    has_any_product = bool(good_product_frames)

    reasons: List[str] = []
    if len(observations) < min_obs:
        return {
            "triggered": False,
            "reason": "too_few_observations_for_recovery_decision",
            "num_observations": len(observations),
            "min_observations_before_recovery": min_obs,
        }
    if len(good_product_frames) < min_good_product_frames:
        reasons.append("few_good_product_frames")
    if min_good_price_frames > 0 and len(good_price_frames) < min_good_price_frames:
        reasons.append("few_good_price_frames")
    if trigger_if_product_missing and not has_any_product:
        reasons.append("product_missing")
    if trigger_if_price_only and has_any_price and not has_any_product:
        reasons.append("price_only_track")

    return {
        "triggered": bool(reasons),
        "reasons": list(dict.fromkeys(reasons)),
        "num_observations": len(observations),
        "good_product_frames": good_product_frames,
        "good_product_frame_count": len(good_product_frames),
        "min_good_product_frames": min_good_product_frames,
        "good_price_frames": good_price_frames,
        "good_price_frame_count": len(good_price_frames),
        "min_good_price_frames": min_good_price_frames,
    }


def _is_good_product_observation(obs: TrackObservation, cfg: Mapping[str, Any]) -> bool:
    min_chars = int(cfg.get("min_product_chars", 8))
    min_alpha = int(cfg.get("min_product_alpha_chars", 6))
    text = str((obs.final or {}).get("product_name") or "").strip()
    alpha_count = sum(1 for ch in text if ch.isalpha())
    if len(text) >= min_chars and alpha_count >= min_alpha:
        return True

    pm = (obs.final or {}).get("product_match") if isinstance((obs.final or {}).get("product_match"), Mapping) else {}
    accepted_statuses = set(str(x) for x in (cfg.get("accepted_catalog_statuses") or ["accepted", "strong_accept", "soft_accept", "price_text_soft_accept"]))
    status = str(pm.get("status") or pm.get("source") or "")
    score = _safe_float(pm.get("score", pm.get("text_score", 0.0)), 0.0)
    if status in accepted_statuses and score >= float(cfg.get("min_catalog_score", 0.46)):
        return True
    return False


def _is_good_price_observation(obs: TrackObservation) -> bool:
    value = str((obs.final or {}).get("main_price") or "").strip()
    if not value:
        return False
    try:
        v = float(str(value).replace(",", "."))
        return v > 0.0
    except Exception:
        return False


def _select_sr_recovery_frames(
    indexed_images: Sequence[Tuple[int, Path]],
    observations: Sequence[TrackObservation],
    cfg: Mapping[str, Any],
    *,
    max_frames: int,
) -> List[Tuple[int, Path]]:
    obs_by_frame: Dict[int, List[TrackObservation]] = {}
    for obs in observations:
        obs_by_frame.setdefault(int(obs.frame_index), []).append(obs)

    prefer_without_product = bool(cfg.get("prefer_frames_without_product", True))
    candidates: List[Tuple[float, int, Path]] = []
    for frame_idx, path in indexed_images:
        frame_obs = obs_by_frame.get(int(frame_idx), [])
        best_score = max([float(o.score) for o in frame_obs] or [0.0])
        good_product = any(_is_good_product_observation(o, cfg) for o in frame_obs)
        good_price = any(_is_good_price_observation(o) for o in frame_obs)
        score = best_score
        if prefer_without_product and not good_product:
            score += 0.35
        if good_price and not good_product:
            score += 0.18
        candidates.append((score, int(frame_idx), path))
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return [(idx, path) for _, idx, path in candidates[:max_frames]]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)

def _write_track_fusion_image(
    *,
    observations: Sequence[TrackObservation],
    reference_observation: TrackObservation,
    aggregated_track: Mapping[str, Any],
    debug_images_dir: Path,
    track_folder: DetectionTrackFolder,
    code_decoder: CodeDecoder,
    cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    fused, meta = fuse_track_images(
        observations,
        reference_observation=reference_observation,
        max_images=int(cfg.get("max_images", 9)),
        max_work_side=int(cfg.get("max_work_side", 900)),
        align=bool(cfg.get("align", True)),
        denoise_h=float(cfg.get("denoise_h", 7.0)),
        denoise_h_color=float(cfg.get("denoise_h_color", 7.0)),
        template_window_size=int(cfg.get("template_window_size", 7)),
        search_window_size=int(cfg.get("search_window_size", 21)),
        consensus_enabled=bool(cfg.get("fusion_consensus_enabled", True)),
        consensus_max_candidates=int(cfg.get("consensus_max_candidates", 32)),
        consensus_feature_size=int(cfg.get("consensus_feature_size", 64)),
        visual_similarity_threshold=float(cfg.get("visual_similarity_threshold", 0.66)),
        evidence_similarity_threshold=float(cfg.get("evidence_similarity_threshold", 0.50)),
        min_cluster_size=int(cfg.get("min_cluster_size", 2)),
        ocr_similarity_enabled=bool(cfg.get("ocr_similarity_enabled", True)),
        ocr_embedding_dim=int(cfg.get("ocr_embedding_dim", 384)),
        ocr_knn_k=int(cfg.get("ocr_knn_k", 4)),
        ocr_similarity_threshold=float(cfg.get("ocr_similarity_threshold", 0.56)),
        ocr_strong_similarity_threshold=float(cfg.get("ocr_strong_similarity_threshold", 0.70)),
        ocr_gzip_weight=float(cfg.get("ocr_gzip_weight", 0.55)),
        focus_selection_enabled=bool(cfg.get("focus_selection_enabled", True)),
        focus_score_weight=float(cfg.get("focus_score_weight", 0.18)),
        focus_roi_policy=str(cfg.get("focus_roi_policy", "price_tag")),
        min_focus_norm_for_fusion=float(cfg.get("min_focus_norm_for_fusion", 0.12)),
        align_mode=str(cfg.get("align_mode", "phase_ecc")),
        ecc_motion=str(cfg.get("ecc_motion", "euclidean")),
        ecc_min_correlation=float(cfg.get("ecc_min_correlation", 0.18)),
        phase_min_response=float(cfg.get("phase_min_response", 0.08)),
        max_translation_ratio=float(cfg.get("max_translation_ratio", 0.22)),
        denoise_stage=str(cfg.get("denoise_stage", "post_fusion")),
        sr_enabled=bool(cfg.get("sr_enabled", False)),
        sr_scale=float(cfg.get("sr_scale", 2.0)),
        sr_stage=str(cfg.get("sr_stage", "pre_nlmeans")),
        sr_min_side=int(cfg.get("sr_min_side", 320)),
        sr_max_side=int(cfg.get("sr_max_side", 1400)),
        sr_method=str(cfg.get("sr_method", "lanczos")),
    )
    out: Dict[str, Any] = dict(meta or {})
    if fused is None or fused.size == 0:
        return out

    debug_images_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{_safe_name(track_folder.sequence_name)}__{_safe_name(track_folder.track_id)}__best_denoised.jpg"
    out_path = debug_images_dir / out_name
    imwrite_unicode(out_path, fused)
    out["path"] = str(out_path)

    image_variants: List[Tuple[str, np.ndarray]] = [("fused", fused)]
    correction_meta: Dict[str, Any] = {}
    if bool(cfg.get("tag_correction_enabled", True)):
        for variant_name, variant_img, variant_meta in build_fused_tag_variants(fused, cfg):
            if variant_img is None or variant_img.size == 0:
                continue
            image_variants.append((variant_name, variant_img))
            correction_meta[variant_name] = variant_meta
            if bool(cfg.get("save_tag_corrected", True)):
                corrected_name = f"{_safe_name(track_folder.sequence_name)}__{_safe_name(track_folder.track_id)}__{variant_name}.jpg"
                corrected_path = debug_images_dir / corrected_name
                imwrite_unicode(corrected_path, variant_img)
                out.setdefault("corrected_images", {})[variant_name] = str(corrected_path)
    if correction_meta:
        out["tag_correction"] = correction_meta

    decode_on_fused = bool(cfg.get("decode_codes_on_fused", True))
    only_when_missing = bool(cfg.get("decode_only_when_missing", True))
    decode_corrected = bool(cfg.get("decode_codes_on_corrected", True))
    should_decode = decode_on_fused and ((not only_when_missing) or (not has_decoded_code(observations)))
    if should_decode:
        decode_variants = image_variants if decode_corrected else [("fused", fused)]
        all_codes: List[Dict[str, Any]] = []
        debug_by_variant: Dict[str, Any] = {}
        for variant_name, variant_img in decode_variants:
            codes = code_decoder.decode(variant_img)
            code_debug = code_decoder.get_debug_info() if hasattr(code_decoder, "get_debug_info") else {}
            debug_by_variant[variant_name] = {
                "attempt_count": len(code_debug.get("attempts") or []),
                "roi_count": code_debug.get("roi_count", 0),
                "detected_count": code_debug.get("detected_count", 0),
                "decoded_count": code_debug.get("decoded_count", 0),
            }
            for c in codes:
                d = c.to_dict()
                d["source_image_variant"] = variant_name
                if d.get("decoder"):
                    d["decoder"] = f"{variant_name}|{d['decoder']}"
                all_codes.append(d)
            if any(d.get("decoded") and str(d.get("payload") or "") for d in all_codes):
                # A later corrected variant can still be useful, but do not burn
                # CPU on all variants if a payload is already recovered.
                if bool(cfg.get("stop_code_decode_after_first_payload", True)):
                    break
        codes_dict = _dedupe_code_dicts(all_codes)
        decoded = [c for c in codes_dict if c.get("decoded") and str(c.get("payload") or "")]
        out["code_fallback"] = {
            "enabled": True,
            "reason": "no_frame_code_decoded" if not has_decoded_code(observations) else "forced",
            "decoded_count": len(decoded),
            "attempt_count": sum(int(v.get("attempt_count") or 0) for v in debug_by_variant.values()),
            "roi_count": sum(int(v.get("roi_count") or 0) for v in debug_by_variant.values()),
            "detected_count": len(codes_dict),
            "variants": debug_by_variant,
            "codes": codes_dict,
        }
        if decoded:
            final = aggregated_track.get("aggregated_final") if isinstance(aggregated_track.get("aggregated_final"), dict) else None
            if final is not None:
                final["decoded_code_fallback"] = decoded[0]
                final.setdefault("decoded_codes", decoded)
    else:
        out["code_fallback"] = {"enabled": False, "reason": "frame_code_already_present" if only_when_missing else "disabled"}

    return out


def _dedupe_code_dicts(codes: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    decoded_first = sorted(
        [dict(c) for c in codes],
        key=lambda d: (not bool(d.get("decoded") and str(d.get("payload") or "")), -float(d.get("conf") or 0.0)),
    )
    for d in decoded_first:
        if d.get("decoded") and str(d.get("payload") or ""):
            key = (str(d.get("kind") or ""), str(d.get("fmt") or ""), str(d.get("payload") or ""))
        else:
            bbox = d.get("bbox") or []
            key = (str(d.get("kind") or ""), tuple(round(float(x) / 12) for x in bbox) if bbox else str(d.get("decoder") or ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out[:16]


def _write_best_debug_image(
    *,
    obs: TrackObservation,
    aggregated_track: Mapping[str, Any],
    debug_images_dir: Path,
    track_folder: DetectionTrackFolder,
    max_image_side: int = 520,
    upscale_small: bool = True,
) -> str:
    """Write a structured review card for one aggregated price tag.

    Layout requested for manual audit:
      1. price-tag image at the top;
      2. flat task-output fields below;
      3. extra diagnostic information last.
    """
    fused_meta = aggregated_track.get("fused_image") if isinstance(aggregated_track.get("fused_image"), Mapping) else {}
    fused_path = str(fused_meta.get("path") or "")
    src_image = Path(fused_path) if fused_path else Path(str(obs.image_path).split("#", 1)[0])
    image = imread_unicode(src_image)
    if image is None:
        return ""

    task_row = build_task_output_record(aggregated_track, {
        "fallback_frame_index_as_timestamp": True,
        "main_price_target": "price_card",
    })
    final = aggregated_track.get("aggregated_final") if isinstance(aggregated_track.get("aggregated_final"), Mapping) else {}
    votes = aggregated_track.get("votes") if isinstance(aggregated_track.get("votes"), Mapping) else {}
    product_match = final.get("product_match") if isinstance(final.get("product_match"), Mapping) else {}
    diagnostics = aggregated_track.get("diagnostics") if isinstance(aggregated_track.get("diagnostics"), Mapping) else {}
    validation = aggregated_track.get("validation") if isinstance(aggregated_track.get("validation"), Mapping) else {}
    warnings = [str(w) for w in (aggregated_track.get("warnings") or [])]

    price_vote = votes.get("main_price") if isinstance(votes.get("main_price"), Mapping) else {}
    product_vote = votes.get("product_name") if isinstance(votes.get("product_name"), Mapping) else {}
    price_diag = diagnostics.get("price") if isinstance(diagnostics.get("price"), Mapping) else {}
    product_diag = diagnostics.get("product") if isinstance(diagnostics.get("product"), Mapping) else {}
    code_fb = fused_meta.get("code_fallback") if isinstance(fused_meta.get("code_fallback"), Mapping) else {}

    extra_rows = [
        ("track", f"{track_folder.sequence_name}/{track_folder.track_id}"),
        ("status", str(aggregated_track.get("status") or "")),
        ("observations", str(aggregated_track.get("num_observations") or 0)),
        ("best_score", f"{float(obs.score):.3f}"),
        ("price_vote", f"value={price_vote.get('value', '')} weight={price_vote.get('weight', '')} ambiguous={price_vote.get('ambiguous', False)}"),
        ("product_vote", f"value={truncate_text(product_vote.get('value', ''), 72)} weight={product_vote.get('weight', '')} ambiguous={product_vote.get('ambiguous', False)}"),
        ("price_consistency", f"winner_ratio={price_diag.get('winner_ratio', '')} unique={price_diag.get('unique_count', '')}"),
        ("product_consistency", f"winner_ratio={product_diag.get('winner_ratio', '')} unique={product_diag.get('unique_count', '')}"),
        ("catalog", f"{product_match.get('status') or '-'} item_id={product_match.get('item_id') or '-'} score={product_match.get('score') or product_match.get('text_score') or '-'}"),
        ("catalog_name", truncate_text(product_match.get("catalog_name") or "", 110)),
        ("validation", f"score={validation.get('score', '')} needs_review={validation.get('needs_review', '')}"),
        ("fused", f"path={Path(fused_path).name if fused_path else '-'} frames={fused_meta.get('frame_count', '-')} code_fb={code_fb.get('decoded_count', 0) if code_fb else 0}"),
        ("warnings", truncate_text(", ".join(warnings), 130)),
    ]

    rendered = _render_structured_track_debug(
        image,
        title=f"{track_folder.sequence_name}/{track_folder.track_id}",
        task_row=task_row,
        extra_rows=extra_rows,
        max_image_side=max_image_side,
        upscale_small=upscale_small,
    )
    out_name = f"{_safe_name(track_folder.sequence_name)}__{_safe_name(track_folder.track_id)}__best.jpg"
    out_path = debug_images_dir / out_name
    imwrite_unicode(out_path, rendered)
    return str(out_path)



def _write_outputs(
    output: Mapping[str, Any],
    track_rows: Sequence[TrackTableRecord],
    frame_rows: Sequence[FrameRunRecord],
    out_dir: Path,
    *,
    write_frame_table: bool = True,
    task_output_cfg: Mapping[str, Any] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_dict = dict(output)
    tracks_for_task = output_dict.get("tracks") if isinstance(output_dict.get("tracks"), list) else []
    output_dict["task_output"] = write_task_outputs(tracks_for_task, out_dir, task_output_cfg or {})
    with open(out_dir / "detected_tracks_results.json", "w", encoding="utf-8") as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=2)

    track_dicts = [r.to_dict() for r in track_rows]
    frame_dicts = [r.to_dict() for r in frame_rows]
    _write_tsv(out_dir / "detected_tracks_summary.tsv", track_dicts)
    _write_csv(out_dir / "detected_tracks_summary.csv", track_dicts)
    if write_frame_table:
        _write_tsv(out_dir / "detected_tracks_frames.tsv", frame_dicts)


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: _cell(row.get(c, "")) for c in cols})


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: _cell(row.get(c, "")) for c in cols})


def _track_table_row(track_folder: DetectionTrackFolder, aggregated: Mapping[str, Any], best_debug: str) -> TrackTableRecord:
    best = aggregated.get("best_observation") if isinstance(aggregated.get("best_observation"), Mapping) else {}
    final = aggregated.get("aggregated_final") if isinstance(aggregated.get("aggregated_final"), Mapping) else {}
    product_match = final.get("product_match") if isinstance(final.get("product_match"), Mapping) else {}
    csv_status = ""
    if isinstance(product_match, Mapping):
        csv_status = str(product_match.get("status") or product_match.get("source") or "")
    diag = aggregated.get("diagnostics") if isinstance(aggregated.get("diagnostics"), Mapping) else {}
    price_diag = diag.get("price") if isinstance(diag.get("price"), Mapping) else {}
    product_diag = diag.get("product") if isinstance(diag.get("product"), Mapping) else {}
    validation = aggregated.get("validation") if isinstance(aggregated.get("validation"), Mapping) else {}
    split = aggregated.get("split") if isinstance(aggregated.get("split"), Mapping) else {}
    debug_plots = aggregated.get("debug_plots") if isinstance(aggregated.get("debug_plots"), Mapping) else {}
    fused_image = aggregated.get("fused_image") if isinstance(aggregated.get("fused_image"), Mapping) else {}
    code_fb = fused_image.get("code_fallback") if isinstance(fused_image.get("code_fallback"), Mapping) else {}
    catalog_gate = aggregated.get("catalog_gate") if isinstance(aggregated.get("catalog_gate"), Mapping) else {}
    return TrackTableRecord(
        sequence_name=track_folder.sequence_name,
        source_track_id=track_folder.track_id,
        track_key=track_folder.track_key,
        source_track_dir=str(track_folder.track_dir),
        status=str(aggregated.get("status") or ""),
        num_images=len(track_folder.images),
        num_observations=int(aggregated.get("num_observations") or 0),
        frame_span="-".join(str(x) for x in (aggregated.get("frame_span") or [])),
        best_image=str(best.get("image_path") or best.get("saved_crop") or ""),
        best_debug_image=best_debug or str(aggregated.get("best_debug_image") or ""),
        best_score=best.get("score", ""),
        main_price=final.get("main_price", ""),
        product_name=final.get("product_name", ""),
        unit=final.get("unit", ""),
        item_id=product_match.get("item_id", "") if isinstance(product_match, Mapping) else "",
        csv_status=csv_status,
        needs_review=final.get("needs_review", ""),
        split_index=split.get("split_index", ""),
        split_count=split.get("split_count", ""),
        split_reason=split.get("reason", ""),
        price_consistency=price_diag.get("winner_ratio", ""),
        price_unique_count=price_diag.get("unique_count", ""),
        product_consistency=product_diag.get("winner_ratio", ""),
        validation_score=validation.get("score", ""),
        glare_applied_best=best.get("glare_applied", ""),
        glare_method_best=best.get("glare_method", ""),
        debug_timeline_plot=debug_plots.get("timeline", ""),
        debug_votes_plot=debug_plots.get("votes", ""),
        best_fused_image=fused_image.get("path", ""),
        fused_code_fallback_decoded=code_fb.get("decoded_count", ""),
        fused_code_fallback_attempts=code_fb.get("attempt_count", ""),
        stock_status=final.get("stock_status", ""),
        catalog_gate_rejected=catalog_gate.get("rejected", ""),
        catalog_reject_reasons=",".join(str(x) for x in (catalog_gate.get("reject_reasons") or [])),
        warnings=",".join(str(w) for w in (aggregated.get("warnings") or [])),
    )





def _task_output_cfg(cfg: Mapping[str, Any], dt_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve flat task-output config from global and dataset sections."""
    global_cfg = cfg.get("task_output", {}) if isinstance(cfg.get("task_output"), Mapping) else {}
    local_cfg = dt_cfg.get("task_output", {}) if isinstance(dt_cfg.get("task_output"), Mapping) else {}
    return deep_merge(dict(global_cfg), dict(local_cfg))


def _runtime_cfg_for_detected_tracks(cfg: Mapping[str, Any], dt_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a runtime config with detected-track-specific output policy."""
    import copy

    out = copy.deepcopy(dict(cfg))
    output = out.setdefault("output", {})
    # The dataset runner should produce compact track-level reports by default.
    # Heavy per-frame artifacts are opt-in.
    output["save_debug"] = bool(dt_cfg.get("save_frame_debug", dt_cfg.get("keep_items", False) and output.get("save_debug", False)))
    output["save_crops"] = bool(dt_cfg.get("save_frame_crops", dt_cfg.get("keep_items", False) and output.get("save_crops", False)))
    output["write_item_json"] = bool(dt_cfg.get("write_frame_json", dt_cfg.get("keep_items", False) and output.get("write_item_json", False)))

    tilt = out.setdefault("tilt_corrector", {})
    if not output["save_debug"]:
        tilt["save_tilt_debug"] = False

    input_structure = out.setdefault("input_structure", {})
    if not output["save_debug"]:
        input_structure["save_debug"] = False

    rail = out.setdefault("rail_segmentation", {})
    if not output["save_debug"]:
        rail["save_rail_debug"] = False
        rail["save_cell_crops"] = False

    return out


def _cleanup_optional_output_dirs(out_dir: Path, *, keep_items: bool, keep_tracks_dir: bool) -> None:
    if not keep_items:
        shutil.rmtree(out_dir / "items", ignore_errors=True)
        shutil.rmtree(out_dir / "_work_items", ignore_errors=True)
    if not keep_tracks_dir:
        shutil.rmtree(out_dir / "tracks", ignore_errors=True)




def _render_structured_track_debug(
    image: np.ndarray,
    *,
    title: str,
    task_row: Mapping[str, Any],
    extra_rows: Sequence[Tuple[str, str]],
    max_image_side: int = 620,
    upscale_small: bool = True,
) -> np.ndarray:
    """Render a human-readable card: tag image -> task fields -> diagnostics."""
    if image is None or image.size == 0:
        image = np.full((90, 220, 3), 245, dtype=np.uint8)
    h, w = image.shape[:2]
    max_side = max(180, int(max_image_side))
    scale = min(1.0, max_side / max(1, max(h, w)))
    if upscale_small and max(h, w) < 280:
        scale = min(max_side / max(1, max(h, w)), max(2.4, 430.0 / max(1, max(h, w))))
    if abs(scale - 1.0) > 1e-3:
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        image = cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=interp)
    ih, iw = image.shape[:2]

    panel_w = max(1040, iw + 40)
    margin = 20
    title_h = 46
    image_h = ih + 2 * margin

    task_fields = [
        ("filename", "filename"),
        ("product_name", "product_name"),
        ("price_default", "price_default"),
        ("price_card", "price_card"),
        ("price_discount", "price_discount"),
        ("barcode", "barcode"),
        ("discount_amount", "discount_amount"),
        ("id_sku", "id_sku"),
        ("print_datetime", "print_datetime"),
        ("code", "code"),
        ("additional_info", "additional_info"),
        ("color", "color"),
        ("special_symbols", "special_symbols"),
        ("frame_timestamp", "frame_timestamp"),
        ("bbox", "bbox"),
    ]
    qr_fields = [
        ("qr_code_barcode", "qr_code_barcode"),
        ("price1_qr", "price1_qr"),
        ("price2_qr", "price2_qr"),
        ("price3_qr", "price3_qr"),
        ("price4_qr", "price4_qr"),
        ("wholesale_level_1_count", "wholesale_level_1_count"),
        ("wholesale_level_1_price", "wholesale_level_1_price"),
        ("wholesale_level_2_count", "wholesale_level_2_count"),
        ("wholesale_level_2_price", "wholesale_level_2_price"),
        ("action_price_qr", "action_price_qr"),
        ("action_code_qr", "action_code_qr"),
    ]

    task_rows = []
    for key, label in task_fields:
        if key == "bbox":
            val = f"{task_row.get('x_min','')},{task_row.get('y_min','')},{task_row.get('x_max','')},{task_row.get('y_max','')}"
        else:
            val = task_row.get(key, "")
        task_rows.append((label, val))
    qr_rows = [(label, task_row.get(key, "")) for key, label in qr_fields]

    section_h_task = _section_height(len(task_rows), columns=2)
    section_h_qr = _section_height(len(qr_rows), columns=2)
    section_h_extra = _section_height(len(extra_rows), columns=1, value_lines=True)
    canvas_h = title_h + image_h + section_h_task + section_h_qr + section_h_extra + margin
    canvas = np.full((canvas_h, panel_w, 3), 255, dtype=np.uint8)

    # Header.
    cv2.rectangle(canvas, (0, 0), (panel_w - 1, title_h - 1), (245, 247, 250), -1)
    cv2.rectangle(canvas, (0, 0), (panel_w - 1, title_h - 1), (220, 225, 232), 1)
    canvas = _draw_text_pil(canvas, f"Ценник: {title}", (margin, 11), font_size=22, color_bgr=(20, 20, 20), bold=True)

    # Top image.
    y = title_h
    cv2.rectangle(canvas, (0, y), (panel_w - 1, y + image_h - 1), (252, 252, 252), -1)
    x_img = max(margin, (panel_w - iw) // 2)
    canvas[y + margin:y + margin + ih, x_img:x_img + iw] = image
    cv2.rectangle(canvas, (x_img, y + margin), (x_img + iw - 1, y + margin + ih - 1), (165, 165, 165), 1)
    y += image_h

    canvas, y = _draw_kv_section(canvas, y, "CSV/JSON поля", task_rows, columns=2)
    canvas, y = _draw_kv_section(canvas, y, "Данные QR-кода", qr_rows, columns=2)
    canvas, y = _draw_kv_section(canvas, y, "Дополнительная диагностика", extra_rows, columns=1)
    return canvas[: min(canvas.shape[0], y + margin), :, :]


def _section_height(n_rows: int, *, columns: int = 2, value_lines: bool = False) -> int:
    rows_per_col = (max(1, n_rows) + max(1, columns) - 1) // max(1, columns)
    row_h = 25 if not value_lines else 28
    return 42 + rows_per_col * row_h + 14


def _draw_kv_section(
    canvas: np.ndarray,
    y: int,
    title: str,
    rows: Sequence[Tuple[str, Any]],
    *,
    columns: int = 2,
) -> Tuple[np.ndarray, int]:
    h, w = canvas.shape[:2]
    margin = 20
    section_top = y
    rows = list(rows)
    rows_per_col = (max(1, len(rows)) + max(1, columns) - 1) // max(1, columns)
    row_h = 25
    section_h = 42 + rows_per_col * row_h + 14
    cv2.rectangle(canvas, (margin, section_top + 8), (w - margin - 1, section_top + section_h - 1), (248, 250, 252), -1)
    cv2.rectangle(canvas, (margin, section_top + 8), (w - margin - 1, section_top + section_h - 1), (218, 225, 235), 1)
    canvas = _draw_text_pil(canvas, title, (margin + 12, section_top + 18), font_size=18, color_bgr=(38, 70, 105), bold=True)

    col_w = (w - 2 * margin - 24) // max(1, columns)
    y0 = section_top + 46
    for idx, (key, value) in enumerate(rows):
        col = idx // rows_per_col
        row = idx % rows_per_col
        x = margin + 12 + col * col_w
        yy = y0 + row * row_h
        key_text = truncate_text(str(key), 30 if columns == 1 else 27)
        value_text = truncate_text(str(value or ""), 78 if columns == 1 else 32)
        key_font = 14 if columns == 1 else 13
        value_font = 14 if columns == 1 else 13
        value_x = x + (205 if columns == 1 else 245)
        canvas = _draw_text_pil(canvas, key_text, (x, yy), font_size=key_font, color_bgr=(80, 80, 80), bold=True)
        canvas = _draw_text_pil(canvas, value_text, (value_x, yy), font_size=value_font, color_bgr=(15, 15, 15), bold=False)
    return canvas, section_top + section_h


def _render_compact_track_debug(
    image: np.ndarray,
    lines: Sequence[str],
    *,
    max_image_side: int = 520,
    upscale_small: bool = True,
) -> np.ndarray:
    if image is None or image.size == 0:
        image = np.full((80, 160, 3), 240, dtype=np.uint8)
    h, w = image.shape[:2]
    max_side = max(120, int(max_image_side))
    scale = min(1.0, max_side / max(1, max(h, w)))
    if upscale_small and max(h, w) < 220:
        # Small track crops need to be visually inspectable; upscale them close
        # to the compact canvas width instead of leaving a tiny image on white
        # background.
        scale = min(max_side / max(1, max(h, w)), max(2.0, 320.0 / max(1, max(h, w))))
    if abs(scale - 1.0) > 1e-3:
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        image = cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=interp)
    ih, iw = image.shape[:2]

    line_h = 23
    panel_w = max(iw, 420)
    panel_h = 12 + line_h * max(1, len(lines))
    canvas_w = panel_w
    canvas_h = panel_h + ih
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    canvas[:panel_h, :, :] = 248
    canvas[panel_h:panel_h + ih, :iw, :] = image
    cv2.rectangle(canvas, (0, panel_h), (iw - 1, panel_h + ih - 1), (205, 205, 205), 1)

    y = 7
    for idx, line in enumerate(lines):
        canvas = _draw_text_pil(
            canvas,
            truncate_text(str(line), 120),
            (8, y),
            font_size=16 if idx == 0 else 15,
            color_bgr=(0, 0, 0),
            bold=(idx == 0),
        )
        y += line_h
    return canvas




def _find_observation_for_aggregated_best(observations: Sequence[TrackObservation], aggregated: Mapping[str, Any]) -> TrackObservation:
    if not observations:
        raise ValueError("no observations")
    best = aggregated.get("best_observation") if isinstance(aggregated.get("best_observation"), Mapping) else {}
    obs_id = str(best.get("observation_id") or "")
    if obs_id:
        for o in observations:
            if str(o.observation_id) == obs_id:
                return o
    frame = best.get("frame_index")
    if frame is not None:
        try:
            fi = int(frame)
            for o in observations:
                if int(o.frame_index) == fi:
                    return o
        except Exception:
            pass
    return max(observations, key=lambda o: float(o.score))


def _frame_index_cfg(dt_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Return frame-index extraction configuration.

    The detected-track dataset is usually made of already-cropped frames where
    file stems are the original video frame ids, e.g. ``6085.jpg``.  In that
    case frame_index must be 6085, not the local 0-based position inside the
    track folder.  This value is then used by plots, aggregation diagnostics and
    task-output frame_timestamp conversion.
    """
    raw = dt_cfg.get("frame_index", {}) if isinstance(dt_cfg.get("frame_index"), Mapping) else {}
    cfg = {
        "source": "auto",              # auto | filename_stem | first_number | regex | enumerate
        "regex": "",                   # optional regex with one capturing group
        "sort_by": "frame_index",      # frame_index | natural_name | name | path
        "strict": False,
        "fallback": "enumerate",       # enumerate | error
    }
    cfg.update(dict(raw))
    # Backward-compatible flat aliases.
    for old_key, new_key in (
        ("frame_index_source", "source"),
        ("frame_index_regex", "regex"),
        ("frame_index_sort_by", "sort_by"),
        ("sort_images_by", "sort_by"),
    ):
        if old_key in dt_cfg:
            cfg[new_key] = dt_cfg.get(old_key)
    return cfg


def _indexed_track_images(images: Sequence[Path], cfg: Mapping[str, Any]) -> List[Tuple[int, Path]]:
    indexed: List[Tuple[int, int, Path]] = []
    used: set[int] = set()
    for ordinal, path in enumerate(images):
        frame_idx = _extract_frame_index(path, ordinal, cfg)
        if frame_idx in used:
            # Duplicate numeric names are pathological for our track model. Keep
            # deterministic uniqueness while preserving the original value scale.
            base = int(frame_idx) * 100000 + ordinal
            while base in used:
                base += 1
            frame_idx = base
        used.add(int(frame_idx))
        indexed.append((int(frame_idx), ordinal, path))

    sort_by = str(cfg.get("sort_by") or "frame_index").lower()
    if sort_by in {"frame_index", "frame", "numeric", "number"}:
        indexed.sort(key=lambda x: (x[0], x[1], str(x[2]).lower()))
    elif sort_by in {"natural", "natural_name"}:
        indexed.sort(key=lambda x: (_natural_key(x[2].name), x[1]))
    elif sort_by == "name":
        indexed.sort(key=lambda x: (x[2].name.lower(), x[1]))
    elif sort_by == "path":
        indexed.sort(key=lambda x: (str(x[2]).lower(), x[1]))
    # else: keep original order from _iter_track_images.
    return [(frame_idx, path) for frame_idx, _, path in indexed]


def _extract_frame_index(path: Path, ordinal: int, cfg: Mapping[str, Any]) -> int:
    source = str(cfg.get("source") or "auto").lower()
    strict = bool(cfg.get("strict", False))
    fallback = str(cfg.get("fallback") or "enumerate").lower()
    stem = path.stem

    def fail_or_fallback(reason: str) -> int:
        if strict or fallback == "error":
            raise ValueError(f"cannot extract frame_index from {path}: {reason}")
        return int(ordinal)

    if source in {"enumerate", "ordinal", "local"}:
        return int(ordinal)

    regex = str(cfg.get("regex") or "").strip()
    if source == "regex" or regex:
        if not regex:
            return fail_or_fallback("frame_index.source=regex but regex is empty")
        m = re.search(regex, stem) or re.search(regex, path.name)
        if m:
            value = m.group(1) if m.groups() else m.group(0)
            try:
                return int(value)
            except Exception:
                return fail_or_fallback(f"regex value is not int: {value!r}")
        if source == "regex":
            return fail_or_fallback(f"regex did not match: {regex}")

    if source in {"filename_stem", "stem", "auto"}:
        if re.fullmatch(r"\d+", stem):
            return int(stem)
        if source in {"filename_stem", "stem"}:
            return fail_or_fallback(f"stem is not an integer: {stem!r}")

    if source in {"first_number", "first_int", "auto"}:
        m = re.search(r"\d+", stem) or re.search(r"\d+", path.name)
        if m:
            return int(m.group(0))
        if source in {"first_number", "first_int"}:
            return fail_or_fallback("no integer token found")

    if source == "auto":
        return int(ordinal)
    return fail_or_fallback(f"unknown source={source!r}")


def _natural_key(text: str) -> Tuple[Any, ...]:
    parts = re.split(r"(\d+)", str(text).lower())
    return tuple(int(p) if p.isdigit() else p for p in parts)

def _iter_track_images(track_dir: Path, *, recursive: bool) -> List[Path]:
    globber = track_dir.rglob if recursive else track_dir.glob
    return sorted([p for p in globber("*") if p.is_file() and p.suffix.lower() in IMG_EXTS], key=lambda p: _natural_key(p.name))


def _make_item_stem(sequence_name: str, track_id: str, frame_index: int, image_path: Path) -> str:
    seq = _safe_name(sequence_name)
    tid = _safe_name(track_id)
    stem = safe_stem(image_path)
    return f"{seq}__{tid}__f{frame_index:06d}__{stem}"


def _force_track_aggregation_enabled(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    base = dict(cfg.get("track_aggregation", {}) if isinstance(cfg.get("track_aggregation"), Mapping) else {})
    base["enabled"] = True
    return base


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def print_detected_tracks_summary(result: Mapping[str, Any], *, stream: Any = None) -> None:
    stream = stream or sys.stdout
    print(
        f"[OK] tracks={result.get('processed_track_count')} frames={result.get('processed_frame_count')} "
        f"errors={result.get('error_count')} out_dir={result.get('out_dir')}",
        file=stream,
    )


def _safe_name(value: Any) -> str:
    import re

    s = str(value or "").strip()
    s = re.sub(r"[^\w\-.а-яА-ЯёЁ]+", "_", s, flags=re.UNICODE).strip("._-")
    return s or "unnamed"


def _pad_image_width(image: np.ndarray, *, min_width: int = 920) -> np.ndarray:
    if image is None or image.size == 0:
        return image
    h, w = image.shape[:2]
    if w >= min_width:
        return image
    out = np.full((h, min_width, 3), 255, dtype=image.dtype)
    out[:, :w, :] = image
    # Draw a subtle separator so the real crop is visible against the white pad.
    cv2.line(out, (w, 0), (w, max(0, h - 1)), (220, 220, 220), 1)
    return out
