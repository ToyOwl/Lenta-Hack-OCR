"""YAML configuration loader for price tag crop pipeline."""

from __future__ import annotations

import copy
import json
import yaml

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

DEFAULT_CONFIG: Dict[str, Any] = {
  "io": {"image": "", "input_dir": "", "out_dir": "runs/price_tag_crop_pipeline", "recursive": True, },

  "output": {"save_debug": True, "save_crops": True, "write_item_json": True, "write_summary_json": True, "write_summary_tsv": True,},

  "quality": {"min_width": 120, "min_height": 80, "blur_warn": 60.0,},

  "color_names": {"backend": "lab_soft", "lut_path": "", "temperature": 950.0, "max_pixels": 80000, "w2c_index_order": "rgb_fast",},

  "template_classifier": { "enabled": True,},

  "tilt_corrector": {"enabled": False, "coarse_step": 1.0, "fine_step": 0.2, "angle_range": 35.0, "min_text_height": 12, "min_abs_angle": 1.2, "max_work_side": 900,
    "save_tilt_debug": True, "apply_before_structure": True, "apply_before_rail": False, "apply_to_rail_cells": True, "debug": False,},

  "glare_suppression": {"enabled": False, "method": "hybrid", "blend_alpha": 0.72, "clahe_clip_limit": 2.0, "clahe_tile_grid_size": 8,
        "glare_v_threshold": 210, "glare_s_threshold": 72, "glare_min_area_ratio": 0.002, "glare_dilate_px": 2, "inpaint_radius": 3.0,
        "dcp_patch_size": 9, "dcp_omega": 0.78, "dcp_t0": 0.22, "dcp_top_percent": 0.001, "dcp_guided_blur": 9,
        "max_work_side": 720, "save_debug": False, "debug": False,},

  "input_structure": {"mode": "auto", "save_debug": True, "min_cells_for_rail": 2, "decision_threshold": 0.58, "min_rail_width_ratio": 0.42,
        "min_rail_area_ratio": 0.045, "max_single_like_aspect": 2.35, "wide_crop_aspect": 2.65, "require_separator_or_wide": True, "min_separator_count": 1, "debug": False,},

  "rail_segmentation": {"enabled": False, "process_cells": True, "fallback_to_full_tag": True, "save_rail_debug": True, "save_cell_crops": True, "search_y_min_ratio": 0.35,
        "search_y_max_ratio": 0.98, "min_rail_width_ratio": 0.45, "min_rail_height_ratio": 0.055, "max_rail_height_ratio": 0.72, "rail_expand_y_ratio": 0.020,
        "min_cell_width": 80, "max_cell_width": 520, "min_cells": 2, "max_rails": 3, "cell_expand_px": 4, "split_strategy": "beam", "beam_size": 12, "beam_boundary_step_px": 48,
        "beam_min_cell_width_ratio": 0.11, "beam_max_cell_width_ratio": 0.42, "beam_min_path_score": 0.45, "same_template_beam": True, "debug": False, },

  "layout": {"debug": False,},

  "code_decoder": {"enabled": True, "use_pyzbar": True, "try_opencv_qr": True, "qr_roi_scan": True, "preprocessing_variants": True, "keep_undecoded_qr": True,
        "qr_sr_enabled": True, "qr_sr_scale": 2.0, "qr_sr_min_side": 420, "qr_sr_max_side": 1400, "qr_sr_method": "lanczos",
        "qr_perspective_warp": True, "qr_morphology": True, "max_rois": 6, "max_variants_per_roi": 10, "qr_contour_max_rois": 4,},

  "ocr": {"backend": "paddle", "gpu": False, "full_tag_ocr": True, "debug": False,
          "paddle": {"lang": "ru", "ocr_version": "PP-OCRv5", "engine": "","use_angle_cls": True, "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": "eslav_PP-OCRv5_mobile_rec", "input_mode": "ndarray",  "mkldnn": False, "pir_api": False, "offline": False,
            "paddlex_model_source": "", "paddlex_disable_model_source_check": True,
            "rec_model_dir": "", "det_model_dir": "", "cls_model_dir": "", },},

  "super_resolution": {
        "enabled": False,
        "profile": "light",
        "strategy": "replace_if_small",
        "save_crops": False,
        "include_raw_when_appending": True,
        "max_variants_per_crop": 3,
        "apply_to": ["main_price*", "*price*", "product_name", "scale_number", "promo_header", "footer_note", "card_price_small", "no_card_price_small"],
        "skip_for": ["full_tag", "qr_code", "linear_barcode", "barcode", "datamatrix"],
        "trigger": {"run_if_min_side_lt": 180, "run_if_height_lt": 52, "always_for_classes": ["main_price*", "*kopeeks*", "*old_price*", "*discount*"]},
        "profiles": {
            "light": {
                "backends": [
                    {"name": "opencv_lanczos_x2", "backend": "opencv_resize", "method": "lanczos", "scale": 2.0, "max_side": 1200, "sharpen": True}
                ]
            },
            "heavy": {
                "backends": [
                    {"name": "opencv_lanczos_x2", "backend": "opencv_resize", "method": "lanczos", "scale": 2.0, "max_side": 1200, "sharpen": True},
                    {"name": "paddle_telescope_tbsrn_dynamic", "backend": "paddle_sr_dynamic", "repo_dir": "third_party/PaddleOCR", "config": "third_party/PaddleOCR/configs/sr/sr_telescope.yml", "checkpoint_prefix": "models/paddle_sr/_work/extracted/telescope/sr_telescope_train/best_accuracy", "image_shape": "3,32,128", "use_gpu": False, "network_downscale": 2, "norm": "minus_one_one", "output_scale_hint": 2.0}
                ]
            }
        },
    },

  "llm_corrector": {"backend": "none",  "model_path": "", "endpoint_url": "", "api_key": "", "model_name": "", "n_ctx": 4096,
        "paddlenlp_device": "auto",   "paddlenlp_dtype": "auto",  "paddlenlp_local_files_only": False, "paddlenlp_trust_remote_code": False,
        "n_batch": 512, "temperature": 0.0, "top_p": 0.95, "max_tokens": 384, "timeout_s": 20.0, "force_run": False,
        "run_on_templates": ["shelf_red_promo", "hanging_yellow_promo_large", "progressive", "progressive_yellow"], "run_on_warning_quality": True, "min_ocr_confidence": 0.65,
        "fallback_when_unavailable": True, "catalog_top_k": 8, "min_catalog_match_score": 0.56, "debug": False, "verbose": False,
        "catalog": { "path": "", "name_column": "name", "item_id_column": "item_id", "brand_column": "brand", "manufacturer_column": "manufacturer",
            "country_column": "country", "price_columns": ["price", "price_regular", "cost", "cost_regular"], "top_k": 8,
            "min_text_score": 0.30, "min_accept_score": 0.56, "max_rows": 200000,  },
    },


  "csv_corrector": {"enabled": False, "path": "", "top_k": 8, "min_text_score": 0.26, "min_accept_score": 0.58,
        "allow_price_correction": True, "allow_price_only_match": True, "allow_price_only_autofill": False, "force_catalog_price_when_matched": False,
        "max_price_conflict_ratio": 0.12, "max_price_only_candidates": 5, "include_raw_row": False, "allow_close_family_match": True, "family_match_min_name_overlap": 0.72, "family_match_min_price_score": 0.92, "min_text_score_with_price": 0.42, "min_price_score_for_soft_accept": 0.90, "debug": False,
        "catalog": {"path": "", "name_column": "name", "item_id_column": "item_id", "category_id_column": "category_id",
            "brand_column": "brand", "category_column": "category_name", "unit_column": "unit", "quantity_column": "quantity",
            "price_columns": ["price", "price_regular", "cost", "cost_regular"], "top_k": 8, "min_text_score": 0.26,
            "min_accept_score": 0.58, "max_rows": 250000, "fuzzy_token_match": True, "min_fuzzy_token_score": 0.74, "max_candidate_rows": 20000,},
    },

  "track_aggregation": {"enabled": False, "min_iou": 0.04, "max_center_jump_ratio": 0.45, "assignment_threshold": 0.38,
        "max_frame_gap": 12, "prevent_two_observations_same_frame": True, "best_blur_reference": 160.0,
        "price_vote_boost": 1.30, "product_vote_boost": 1.10, "ambiguity_ratio": 0.84, "copy_best_frames": True,
        "min_product_chars": 4, "price_only_ok": True,
        "prefer_larger_decimal_shift": True, "decimal_shift_alias_enabled": True, "decimal_shift_min_larger_price": 5.0,
        "digit_flip_alias_enabled": True,
        "split_mixed_tracks": True, "split_min_segment_observations": 2, "split_min_price_support": 2,
        "split_min_reliable_score": 0.72, "split_price_gap_ratio": 0.16, "split_product_token_overlap": 0.34,
        "false_tag_filter_enabled": True, "false_tag_reject_score": 0.68, "false_tag_review_score": 0.45,
        "false_tag_max_tall_aspect": 0.62, "false_tag_large_area": 120000, "false_tag_min_reasonable_price": 2.0,
        "unstable_price_consistency_review": 0.58, "unstable_price_unique_reject": 4, "debug": False,},

  "task_output": {"enabled": True, "csv_name": "detected_tracks_task_output.csv", "json_name": "detected_tracks_task_output.json",
        "absent_value": "нет", "empty_value": "", "main_price_target": "price_card",
        "video_fps": 0.0, "fallback_frame_index_as_timestamp": False,},

  "detected_tracks_dataset": {"root_dir": "", "out_dir": "runs/detected_tracks_dataset", "recursive_images_in_track": False,
        "min_images_per_track": 1, "skip_hidden": True,
        "frame_index": {"source": "auto", "regex": "", "sort_by": "frame_index", "strict": False, "fallback": "enumerate"},
        "original_coordinates": {"enabled": False, "csv_path": "", "frame_col": "frame_idx", "track_col": "tr_id", "xyxy_col": "xyxy",
            "delimiter": "auto", "encoding": "utf-8-sig", "strict": False, "fallback_to_crop_bbox": True,
            "coordinate_child_mode": "track_bbox", "auto_find": False,
            "auto_find_names": ["original_coordinates.csv", "detections.csv", "tracks.csv", "bboxes.csv"]},
        "copy_best_debug": True, "debug_images_dir_name": "debug_images",
        "debug_plots_dir_name": "debug_plots", "debug_max_image_side": 520, "debug_upscale_small": True,
        "write_debug_plots": True, "keep_items": False, "keep_tracks_dir": False,
        "save_frame_debug": False, "save_frame_crops": False, "write_frame_json": False, "write_frame_table": True,
        "track_fusion": {"enabled": True, "max_images": 9, "max_work_side": 900, "align": True,
            "denoise_h": 7.0, "denoise_h_color": 7.0, "template_window_size": 7, "search_window_size": 21,
            "score_consensus_enabled": True, "fusion_consensus_enabled": True, "consensus_max_candidates": 32,
            "consensus_feature_size": 64, "visual_similarity_threshold": 0.66, "evidence_similarity_threshold": 0.50,
            "min_cluster_size": 2,
            "ocr_similarity_enabled": True, "ocr_embedding_dim": 384, "ocr_knn_k": 4,
            "ocr_similarity_threshold": 0.56, "ocr_strong_similarity_threshold": 0.70, "ocr_gzip_weight": 0.55,
            "focus_selection_enabled": True, "focus_score_weight": 0.18, "focus_roi_policy": "price_tag", "min_focus_norm_for_fusion": 0.12,
            "selected_score_boost": 0.22, "outlier_score_penalty": 0.32,
            "align_mode": "phase_ecc", "ecc_motion": "euclidean", "ecc_min_correlation": 0.18, "phase_min_response": 0.08, "max_translation_ratio": 0.22,
            "denoise_stage": "post_fusion",
            "sr_enabled": True, "sr_scale": 2.0, "sr_stage": "pre_nlmeans", "sr_min_side": 360, "sr_max_side": 1400, "sr_method": "lanczos",
            "tag_correction_enabled": True, "save_tag_corrected": True, "tag_correction_profile": "safe", "tag_correction_sr_enabled": False, "tag_correction_clahe_clip": 1.85,
            "tag_correction_text_gain": 0.22, "tag_correction_red_zone_gain": 0.07, "tag_correction_glare_suppression": True,
            "tag_correction_gray_world_strength": 0.35, "tag_correction_glare_max_area_ratio": 0.045,
            "tag_correction_blackhat_kernel_ratio": 0.020, "tag_correction_final_unsharp_amount": 0.10,
            "decode_codes_on_fused": True, "decode_codes_on_corrected": True, "decode_only_when_missing": True, "stop_code_decode_after_first_payload": True,},
        "sr_recovery": {"enabled": False, "min_good_product_frames": 2, "min_good_price_frames": 0, "trigger_if_product_missing": True,
            "trigger_if_price_only": True, "min_observations_before_recovery": 2, "max_recovery_frames": 3,
            "score_multiplier": 0.92, "item_suffix": "__srrec", "prefer_frames_without_product": True,
            "min_product_chars": 8, "min_product_alpha_chars": 6, "min_catalog_score": 0.46,
            "accepted_catalog_statuses": ["accepted", "strong_accept", "soft_accept", "price_text_soft_accept"],
            "config_overrides": {},},},

  "price_parser": {"allow_compact_in_main_price_zones": True, "allow_compact_in_full_tag": False,},

  "zones": {"code_classes": ["qr_code", "linear_barcode", "barcode", "datamatrix"], "ocr_exclude_classes": ["qr_code", "linear_barcode", "barcode", "datamatrix"],
            "price_compact_classes_prefixes": ["main_price"], "crop_expand_px": 3, },
}

def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(dict(base))

    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = copy.deepcopy(v)
    return out

def load_yaml(path: str | Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is not installed. Run: pip install pyyaml")
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {p}")
    return data

def load_config(path: str | Path | None = None, overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg_meta: Dict[str, Any] = {
        "cwd": str(Path.cwd().resolve()),
    }
    if path:
        config_path = Path(path).expanduser().resolve()
        cfg_meta["config_path"] = str(config_path)
        cfg_meta["config_dir"] = str(config_path.parent)
        cfg = deep_merge(cfg, load_yaml(config_path))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    cfg["_meta"] = deep_merge(cfg.get("_meta", {}) if isinstance(cfg.get("_meta"), Mapping) else {}, cfg_meta)
    return cfg


def get_by_path(cfg: Mapping[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur

def set_by_path(cfg: MutableMapping[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur: MutableMapping[str, Any] = cfg
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, MutableMapping):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value

def apply_cli_overrides(cfg: Dict[str, Any], pairs: Iterable[tuple[str, Any]]) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    for path, value in pairs:
        if value is not None:
            set_by_path(out, path, value)
    return out

def dump_config(cfg: Mapping[str, Any]) -> str:
    if yaml is not None:
        return yaml.safe_dump(dict(cfg), allow_unicode=True, sort_keys=False)
    return json.dumps(cfg, ensure_ascii=False, indent=2)
