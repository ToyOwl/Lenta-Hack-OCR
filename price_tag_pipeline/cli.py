"""YAML-first CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from .code_decoder import CodeDecoder
from .config import apply_cli_overrides, dump_config, load_config
from .io_utils import iter_images
from .layout import HeuristicLayoutExtractor
from .ocr_backends import NullOCR, OCRBackend, PaddleOCRBackend
from .pipeline import PriceTagPipeline
from .template_classifier import ColorNameTemplateClassifier
from .track_aggregator import PriceTagTrackAggregator, write_track_outputs


def build_code_decoder(cfg: Dict[str, Any]) -> CodeDecoder:
    c = cfg.get("code_decoder", {})
    return CodeDecoder(
        use_pyzbar=bool(c.get("use_pyzbar", True)),
        try_opencv_qr=bool(c.get("try_opencv_qr", True)),
        qr_roi_scan=bool(c.get("qr_roi_scan", True)),
        preprocessing_variants=bool(c.get("preprocessing_variants", True)),
        keep_undecoded_qr=bool(c.get("keep_undecoded_qr", True)),
    )


def _bool_from_legacy_positive(pcfg: Dict[str, Any], positive_key: str, legacy_enable_key: str, legacy_disable_key: str, default: bool) -> bool:
    """Resolve positive runtime flags while accepting old YAMLs for migration."""
    if positive_key in pcfg:
        return bool(pcfg.get(positive_key))
    if legacy_enable_key in pcfg:
        return bool(pcfg.get(legacy_enable_key))
    if legacy_disable_key in pcfg:
        return not bool(pcfg.get(legacy_disable_key))
    return bool(default)

def build_ocr_backend(cfg: Dict[str, Any]) -> OCRBackend:
    ocr_cfg = cfg.get("ocr", {})
    backend = str(ocr_cfg.get("backend", "none")).lower().strip()
    if backend == "none":
        return NullOCR()
    if backend == "paddle":
        pcfg = ocr_cfg.get("paddle", {})
        if not isinstance(pcfg, dict):
            pcfg = {}
        return PaddleOCRBackend(
            lang=str(pcfg.get("lang", "ru")),
            use_angle_cls=bool(pcfg.get("use_angle_cls", True)),
            use_gpu=bool(ocr_cfg.get("gpu", False)),
            rec_model_dir=str(pcfg.get("rec_model_dir", "") or ""),
            det_model_dir=str(pcfg.get("det_model_dir", "") or ""),
            cls_model_dir=str(pcfg.get("cls_model_dir", "") or ""),
            ocr_version=str(pcfg.get("ocr_version", "PP-OCRv5") or ""),
            engine=str(pcfg.get("engine", "") or ""),
            text_detection_model_name=str(pcfg.get("text_detection_model_name", "") or ""),
            text_recognition_model_name=str(pcfg.get("text_recognition_model_name", "") or ""),
            debug=bool(ocr_cfg.get("debug", False)),
            input_mode=str(pcfg.get("input_mode", "ndarray") or "ndarray"),
            mkldnn=_bool_from_legacy_positive(pcfg, "mkldnn", "enable_mkldnn", "disable_mkldnn", False),
            pir_api=_bool_from_legacy_positive(pcfg, "pir_api", "enable_pir_api", "disable_pir_api", False),
        )
    raise ValueError(f"Unknown OCR backend: {backend}. Supported values: paddle, none")

def build_layout_extractor(cfg: Dict[str, Any], code_decoder: CodeDecoder) -> HeuristicLayoutExtractor:
    lcfg = cfg.get("layout", {})
    return HeuristicLayoutExtractor(code_decoder=code_decoder, debug=bool(lcfg.get("debug", False)))

def build_template_classifier(cfg: Dict[str, Any]) -> ColorNameTemplateClassifier:
    ccfg = cfg.get("color_names", {})
    return ColorNameTemplateClassifier(
        cn_backend=str(ccfg.get("backend", "lab_soft")),
        cn_lut_path=str(ccfg.get("lut_path", "") or "") or None,
        cn_temperature=float(ccfg.get("temperature", 950.0)),
        cn_max_pixels=int(ccfg.get("max_pixels", 80000)),
        cn_w2c_index_order=str(ccfg.get("w2c_index_order", "rgb_fast") or "rgb_fast"),
    )

def resolve_images(cfg: Dict[str, Any]) -> List[Path]:
    io = cfg.get("io", {})
    image = str(io.get("image", "") or "")
    input_dir = str(io.get("input_dir", "") or "")
    recursive = bool(io.get("recursive", True))
    if image and input_dir:
        raise ValueError("Specify only one of io.image or io.input_dir")
    if image:
        return [Path(image)]
    if input_dir:
        return iter_images(Path(input_dir), recursive=recursive)
    raise ValueError("Specify io.image or io.input_dir in YAML or CLI")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YAML-first OCR/code pipeline for already-cropped Lenta-like price tags")
    p.add_argument("--config", type=str, default="", help="YAML config path")

    p.add_argument("--image", type=str, default=None, help="Override io.image")
    p.add_argument("--input_dir", type=str, default=None, help="Override io.input_dir")
    p.add_argument("--out_dir", type=str, default=None, help="Override io.out_dir")
    p.add_argument("--recursive", action="store_true", default=None, help="Override io.recursive=true")
    p.add_argument("--non_recursive", action="store_true", default=None, help="Override io.recursive=false")
    p.add_argument("--save_debug", action="store_true", default=None, help="Override output.save_debug=true")
    p.add_argument("--no_save_debug", action="store_true", default=None, help="Override output.save_debug=false")
    p.add_argument("--save_crops", action="store_true", default=None, help="Override output.save_crops=true")
    p.add_argument("--no_save_crops", action="store_true", default=None, help="Override output.save_crops=false")
    p.add_argument("--enable_tilt", action="store_true", default=None, help="Override tilt_corrector.enabled=true")
    p.add_argument("--disable_tilt", action="store_true", default=None, help="Override tilt_corrector.enabled=false")
    p.add_argument("--enable_glare_suppression", action="store_true", default=None, help="Enable glare/haze suppression preprocessing")
    p.add_argument("--disable_glare_suppression", action="store_true", default=None, help="Disable glare/haze suppression preprocessing")
    p.add_argument("--glare_method", type=str, default=None, choices=["hybrid", "auto", "clahe", "inpaint_glare", "dark_channel", "none"], help="Glare suppression method")
    p.add_argument("--save_glare_debug", action="store_true", default=None, help="Save *_glare_suppressed.jpg under items/")
    p.add_argument("--enable_rail_split", action="store_true", default=None, help="Override rail_segmentation.enabled=true")
    p.add_argument("--disable_rail_split", action="store_true", default=None, help="Override rail_segmentation.enabled=false")
    p.add_argument("--tilt_before_structure", action="store_true", default=None, help="Override tilt_corrector.apply_before_structure=true")
    p.add_argument("--no_tilt_before_structure", action="store_true", default=None, help="Override tilt_corrector.apply_before_structure=false")
    p.add_argument("--input_mode", type=str, default=None, choices=["auto", "single_tag", "price_rail"], help="Route input before OCR: auto | single_tag | price_rail")
    p.add_argument("--rail_split_strategy", type=str, default=None, choices=["beam", "projection"], help="Override rail_segmentation.split_strategy")
    p.add_argument("--no_process_rail_cells", action="store_true", default=None, help="Only split/save rail cells; do not OCR each cell")

    p.add_argument("--ocr_backend", type=str, default=None, choices=["paddle", "none"], help="Override ocr.backend")
    p.add_argument("--ocr_debug", action="store_true", default=None, help="Write OCR raw diagnostics to item JSON")
    p.add_argument("--paddle_input_mode", type=str, default=None, choices=["ndarray", "path"], help="Override ocr.paddle.input_mode")
    p.add_argument("--enable_llm_correction", action="store_true", default=None, help="Set llm_corrector.backend to paddlenlp unless --llm_backend is also supplied")
    p.add_argument("--disable_llm_correction", action="store_true", default=None, help="Set llm_corrector.backend=none")
    p.add_argument("--llm_model_path", type=str, default=None, help="Override llm_corrector.model_path. For PaddleNLP this is a local model directory.")
    p.add_argument("--llm_model_name", type=str, default=None, help="Override llm_corrector.model_name, e.g. Qwen/Qwen2-0.5B")
    p.add_argument("--llm_backend", type=str, default=None, choices=["none", "paddlenlp", "openai_compatible", "http"], help="Override llm_corrector.backend")
    p.add_argument("--llm_endpoint_url", type=str, default=None, help="Override llm_corrector.endpoint_url for OpenAI-compatible local server")
    p.add_argument("--product_catalog", type=str, default=None, help="Override llm_corrector.catalog.path CSV")
    p.add_argument("--llm_force_run", action="store_true", default=None, help="Run LLM corrector for every processed single tag")
    p.add_argument("--enable_csv_correction", action="store_true", default=None, help="Enable deterministic OCR correction from structured CSV")
    p.add_argument("--disable_csv_correction", action="store_true", default=None, help="Disable deterministic OCR correction from structured CSV")
    p.add_argument("--csv_catalog", type=str, default=None, help="Override csv_corrector.catalog.path")
    p.add_argument("--enable_track_aggregation", action="store_true", default=None, help="Aggregate OCR over consecutive frames/tracks and select best frame")
    p.add_argument("--disable_track_aggregation", action="store_true", default=None, help="Disable track-level aggregation")
    p.add_argument("--track_no_copy_best_frames", action="store_true", default=None, help="Do not copy best frame/crop per track")
    p.add_argument("--disable_digit_flip_alias", action="store_true", default=None, help="Do not merge 6/9 OCR flip variants such as 66.00 and 99.00 during track aggregation")
    p.add_argument("--print_config", action="store_true", help="Print resolved YAML config and exit")
    return p.parse_args()

def cli_overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    cfg_over: Dict[str, Any] = {}
    pairs = []

    if args.image is not None:
        pairs.append(("io.image", args.image))
        pairs.append(("io.input_dir", ""))

    if args.input_dir is not None:
        pairs.append(("io.input_dir", args.input_dir))
        pairs.append(("io.image", ""))
    pairs.append(("io.out_dir", args.out_dir))

    if args.recursive:
        pairs.append(("io.recursive", True))

    if args.non_recursive:
        pairs.append(("io.recursive", False))

    if args.save_debug:
        pairs.append(("output.save_debug", True))

    if args.no_save_debug:
        pairs.append(("output.save_debug", False))

    if args.save_crops:
        pairs.append(("output.save_crops", True))

    if args.no_save_crops:
        pairs.append(("output.save_crops", False))

    if args.enable_tilt:
        pairs.append(("tilt_corrector.enabled", True))

    if args.disable_tilt:
        pairs.append(("tilt_corrector.enabled", False))

    if args.enable_glare_suppression:
        pairs.append(("glare_suppression.enabled", True))

    if args.disable_glare_suppression:
        pairs.append(("glare_suppression.enabled", False))

    pairs.append(("glare_suppression.method", args.glare_method))

    if args.save_glare_debug:
        pairs.append(("glare_suppression.save_debug", True))

    if args.enable_rail_split:
        pairs.append(("rail_segmentation.enabled", True))

    if args.disable_rail_split:
        pairs.append(("rail_segmentation.enabled", False))

    if args.tilt_before_structure:
        pairs.append(("tilt_corrector.apply_before_structure", True))

    if args.no_tilt_before_structure:
        pairs.append(("tilt_corrector.apply_before_structure", False))
    pairs.append(("input_structure.mode", args.input_mode))
    pairs.append(("rail_segmentation.split_strategy", args.rail_split_strategy))

    if args.no_process_rail_cells:
        pairs.append(("rail_segmentation.process_cells", False))
    pairs.append(("ocr.backend", args.ocr_backend))

    if args.ocr_debug:
        pairs.append(("ocr.debug", True))
    pairs.append(("ocr.paddle.input_mode", args.paddle_input_mode))

    if args.enable_llm_correction and args.llm_backend is None:
        pairs.append(("llm_corrector.backend", "paddlenlp"))

    if args.disable_llm_correction:
        pairs.append(("llm_corrector.backend", "none"))

    pairs.append(("llm_corrector.model_path", args.llm_model_path))
    pairs.append(("llm_corrector.model_name", args.llm_model_name))
    pairs.append(("llm_corrector.backend", args.llm_backend))
    pairs.append(("llm_corrector.endpoint_url", args.llm_endpoint_url))
    pairs.append(("llm_corrector.catalog.path", args.product_catalog))

    if args.llm_force_run:
        pairs.append(("llm_corrector.force_run", True))

    if args.enable_csv_correction:
        pairs.append(("csv_corrector.enabled", True))

    if args.disable_csv_correction:
        pairs.append(("csv_corrector.enabled", False))

    pairs.append(("csv_corrector.catalog.path", args.csv_catalog))

    if args.enable_track_aggregation:
        pairs.append(("track_aggregation.enabled", True))

    if args.disable_track_aggregation:
        pairs.append(("track_aggregation.enabled", False))

    if args.track_no_copy_best_frames:
        pairs.append(("track_aggregation.copy_best_frames", False))
    if args.disable_digit_flip_alias:
        pairs.append(("track_aggregation.digit_flip_alias_enabled", False))
    return apply_cli_overrides(cfg_over, pairs)


def main() -> int:
    args = parse_args()
    overrides = cli_overrides_from_args(args)
    cfg = load_config(args.config or None, overrides=overrides)

    if args.print_config:
        print(dump_config(cfg))
        return 0

    out_dir = Path(cfg.get("io", {}).get("out_dir", "runs/price_tag_crop_pipeline"))
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        images = resolve_images(cfg)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    if not images:
        print("[ERROR] no images found", file=sys.stderr)
        return 2

    with open(out_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        f.write(dump_config(cfg))

    code_decoder = build_code_decoder(cfg)
    layout = build_layout_extractor(cfg, code_decoder=code_decoder)
    ocr = build_ocr_backend(cfg)
    template_classifier = build_template_classifier(cfg)
    pipeline = PriceTagPipeline(
        template_classifier=template_classifier,
        layout_detector=layout,
        ocr_backend=ocr,
        code_decoder=code_decoder,
        config=cfg,
    )

    track_aggregator = PriceTagTrackAggregator.from_config(cfg)

    summary: List[Dict[str, Any]] = []
    iterator = tqdm(images, desc="price-tags") if tqdm is not None else images
    for frame_index, img_path in enumerate(iterator):
        try:
            res = pipeline.process_image(img_path, out_dir=out_dir)
        except Exception as e:
            res = {"image_path": str(img_path), "status": "error", "error": repr(e)}
        if track_aggregator.enabled:
            try:
                track_aggregator.add_result(res, frame_index=frame_index)
            except Exception as e:
                res.setdefault("warnings", []).append(f"track_aggregation_failed:{type(e).__name__}:{e}")

        rail_seg = res.get("rail_segmentation") or {}
        input_struct = res.get("input_structure") or {}
        summary.append({
            "image_path": res.get("image_path"),
            "status": res.get("status"),
            "mode": res.get("mode", "single_tag"),
            "input_structure_reason": input_struct.get("reason"),
            "input_structure_confidence": input_struct.get("confidence"),
            "template_name": (res.get("template") or {}).get("template_name"),
            "template_confidence": (res.get("template") or {}).get("confidence"),
            "main_price": ((res.get("prices") or {}).get("main") or {}).get("value"),
            "final_main_price": (res.get("final") or {}).get("main_price"),
            "final_product_name": (res.get("final") or {}).get("product_name"),
            "llm_status": (res.get("llm_correction") or {}).get("status"),
            "csv_status": (res.get("csv_correction") or {}).get("status"),
            "needs_review": (res.get("final") or {}).get("needs_review"),
            "codes_decoded": sum(1 for c in res.get("codes", []) if c.get("decoded")),
            "quality_status": (res.get("quality") or {}).get("status"),
            "rail_count": len(rail_seg.get("rails", [])),
            "rail_cell_count": len(rail_seg.get("cells", [])),
            "tilt_angle": (res.get("tilt") or {}).get("angle"),
            "glare_applied": (res.get("glare_suppression") or {}).get("applied"),
            "glare_method": (res.get("glare_suppression") or {}).get("method"),
            "error": res.get("error"),
        })

    output_cfg = cfg.get("output", {})
    if bool(output_cfg.get("write_summary_json", True)):
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    if bool(output_cfg.get("write_summary_tsv", True)):
        with open(out_dir / "summary.tsv", "w", encoding="utf-8") as f:
            cols = ["image_path", "status", "mode", "input_structure_reason", "input_structure_confidence", "template_name", "template_confidence", "main_price", "final_main_price", "final_product_name", "llm_status", "csv_status", "needs_review", "codes_decoded", "quality_status", "rail_count", "rail_cell_count", "tilt_angle", "glare_applied", "glare_method", "error"]
            f.write("\t".join(cols) + "\n")
            for r in summary:
                f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    if track_aggregator.enabled:
        track_result = track_aggregator.aggregate(out_dir=out_dir)
        write_track_outputs(track_result, out_dir)

    print(f"[OK] processed={len(summary)} out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
