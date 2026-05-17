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
from .config import dump_config
from .debug_vis import _draw_text_pil, truncate_text
from .io_utils import IMG_EXTS, imread_unicode, imwrite_unicode, safe_stem
from .layout import HeuristicLayoutExtractor
from .ocr_backends import OCRBackend
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
    copy_best_debug = bool(dt_cfg.get("copy_best_debug", True))
    keep_items = bool(dt_cfg.get("keep_items", False))
    keep_tracks_dir = bool(dt_cfg.get("keep_tracks_dir", False))
    write_frame_table = bool(dt_cfg.get("write_frame_table", True))
    write_debug_plots = bool(dt_cfg.get("write_debug_plots", True))
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
        for frame_index, image_path in enumerate(track_folder.images):
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
                        selected_score_boost=float(track_fusion_cfg.get("selected_score_boost", 0.22)),
                        outlier_score_penalty=float(track_fusion_cfg.get("outlier_score_penalty", 0.32)),
                    )

                aggregated = aggregator._aggregate_track(sub_state, out_dir=out_dir)  # noqa: SLF001 - public runner, same package
                if visual_consensus_info:
                    aggregated["visual_consensus"] = visual_consensus_info
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
    }

    if write_debug_plots:
        output["global_debug_plots"] = write_global_debug_plots(results, debug_plots_dir)

    _write_outputs(output, table_rows, frame_rows if write_frame_table else [], out_dir, write_frame_table=write_frame_table)
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
) -> Tuple[Dict[str, Any], List[TrackObservation], FrameRunRecord]:
    image = imread_unicode(image_path)
    if image is None:
        raise RuntimeError("failed_to_read_image")

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
        error=str(result.get("error") or ""),
    )
    return result, observations, frame_record


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
    )
    out: Dict[str, Any] = dict(meta or {})
    if fused is None or fused.size == 0:
        return out

    debug_images_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{_safe_name(track_folder.sequence_name)}__{_safe_name(track_folder.track_id)}__best_denoised.jpg"
    out_path = debug_images_dir / out_name
    imwrite_unicode(out_path, fused)
    out["path"] = str(out_path)

    decode_on_fused = bool(cfg.get("decode_codes_on_fused", True))
    only_when_missing = bool(cfg.get("decode_only_when_missing", True))
    should_decode = decode_on_fused and ((not only_when_missing) or (not has_decoded_code(observations)))
    if should_decode:
        codes = code_decoder.decode(fused)
        code_debug = code_decoder.get_debug_info() if hasattr(code_decoder, "get_debug_info") else {}
        codes_dict = [c.to_dict() for c in codes]
        decoded = [c for c in codes_dict if c.get("decoded") and str(c.get("payload") or "")]
        out["code_fallback"] = {
            "enabled": True,
            "reason": "no_frame_code_decoded" if not has_decoded_code(observations) else "forced",
            "decoded_count": len(decoded),
            "attempt_count": len(code_debug.get("attempts") or []),
            "roi_count": code_debug.get("roi_count", 0),
            "detected_count": code_debug.get("detected_count", len(codes_dict)),
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


def _write_best_debug_image(
    *,
    obs: TrackObservation,
    aggregated_track: Mapping[str, Any],
    debug_images_dir: Path,
    track_folder: DetectionTrackFolder,
    max_image_side: int = 520,
    upscale_small: bool = True,
) -> str:
    # Best-track debug must be compact and readable.  Do not reuse per-frame
    # debug images here: they contain layout boxes, OCR panels and large white
    # padding, which makes the track review folder noisy for small crops.
    fused_meta = aggregated_track.get("fused_image") if isinstance(aggregated_track.get("fused_image"), Mapping) else {}
    fused_path = str(fused_meta.get("path") or "")
    src_image = Path(fused_path) if fused_path else Path(str(obs.image_path).split("#", 1)[0])
    image = imread_unicode(src_image)
    if image is None:
        return ""

    final = aggregated_track.get("aggregated_final") if isinstance(aggregated_track.get("aggregated_final"), Mapping) else {}
    votes = aggregated_track.get("votes") if isinstance(aggregated_track.get("votes"), Mapping) else {}
    product_match = final.get("product_match") if isinstance(final.get("product_match"), Mapping) else {}
    warnings = aggregated_track.get("warnings") or []

    price_vote = votes.get("main_price") if isinstance(votes.get("main_price"), Mapping) else {}
    product_vote = votes.get("product_name") if isinstance(votes.get("product_name"), Mapping) else {}
    price = str(final.get("main_price") or "")
    product = str(final.get("product_name") or "")
    status = str(aggregated_track.get("status") or "")

    lines = [
        f"{track_folder.sequence_name}/{track_folder.track_id}  obs={aggregated_track.get('num_observations', 0)}  score={obs.score:.3f}  status={status}",
        f"Цена: {price or '-'}    скидка: {final.get('discount_percent_raw') or '-'}    unit: {final.get('unit') or '-'}    item_id: {product_match.get('item_id') or '-'}",
        f"Товар: {truncate_text(product, 86) if product else '-'}",
        f"DB: {product_match.get('status') or '-'} {truncate_text(product_match.get('catalog_name') or '', 76)}",
        f"Голоса: price_w={price_vote.get('weight', 0)} amb={price_vote.get('ambiguous', False)}; "
        f"product_w={product_vote.get('weight', 0)} amb={product_vote.get('ambiguous', False)}",
    ]
    if fused_path:
        code_fb = fused_meta.get("code_fallback") if isinstance(fused_meta.get("code_fallback"), Mapping) else {}
        lines.append(
            f"best=fused nlmeans frames={fused_meta.get('frame_count', '-')} "
            f"code_fb={code_fb.get('decoded_count', 0) if code_fb else 0} "
            f"attempts={code_fb.get('attempt_count', 0) if code_fb else 0}"
        )
    if final.get("stock_status") == "out_of_stock":
        lines.append("Статус товара: товар закончился")
    if warnings:
        lines.append("warn: " + truncate_text(",".join(str(w) for w in warnings), 110))

    rendered = _render_compact_track_debug(image, lines, max_image_side=max_image_side, upscale_small=upscale_small)
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
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "detected_tracks_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

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

def _iter_track_images(track_dir: Path, *, recursive: bool) -> List[Path]:
    globber = track_dir.rglob if recursive else track_dir.glob
    return sorted([p for p in globber("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])


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
