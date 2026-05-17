# -*- coding: utf-8 -*-
"""CLI for root_dir/sequence-name/track_id/{images} detected-track datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

from price_tag_pipeline.cli import (
    build_code_decoder,
    build_layout_extractor,
    build_ocr_backend,
    build_template_classifier,
)
from price_tag_pipeline.config import apply_cli_overrides, dump_config, load_config
from price_tag_pipeline.detected_tracks_dataset import process_detected_tracks_dataset, print_detected_tracks_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run OCR aggregation over root_dir/sequence-name/track_id/{images} detected-track dataset"
    )
    p.add_argument("--config", type=str, default="", help="YAML config path")
    p.add_argument("--root_dir", type=str, required=False, default=None, help="Dataset root: root/sequence-name/track_id/{images}")
    p.add_argument("--out_dir", type=str, required=False, default=None, help="Output directory")
    p.add_argument("--recursive_images_in_track", action="store_true", default=None, help="Search images inside track_id recursively")
    p.add_argument("--min_images_per_track", type=int, default=None, help="Skip track folders with fewer images")

    p.add_argument("--ocr_backend", type=str, default=None, choices=["paddle", "none"], help="Override ocr.backend")
    p.add_argument("--enable_tilt", action="store_true", default=None, help="Override tilt_corrector.enabled=true")
    p.add_argument("--disable_tilt", action="store_true", default=None, help="Override tilt_corrector.enabled=false")
    p.add_argument("--enable_glare_suppression", action="store_true", default=None, help="Enable glare/haze suppression preprocessing")
    p.add_argument("--disable_glare_suppression", action="store_true", default=None, help="Disable glare/haze suppression preprocessing")
    p.add_argument("--glare_method", type=str, default=None, choices=["hybrid", "auto", "clahe", "inpaint_glare", "dark_channel", "none"], help="Glare suppression method")
    p.add_argument("--save_glare_debug", action="store_true", default=None, help="Save *_glare_suppressed.jpg under items/; implies --keep_items")
    p.add_argument("--enable_csv_correction", action="store_true", default=None, help="Enable deterministic OCR correction from structured CSV")
    p.add_argument("--disable_csv_correction", action="store_true", default=None, help="Disable deterministic OCR correction from structured CSV")
    p.add_argument("--csv_catalog", type=str, default=None, help="Override csv_corrector.catalog.path")
    p.add_argument("--disable_llm_correction", action="store_true", default=None, help="Set llm_corrector.backend=none")
    p.add_argument("--save_debug", action="store_true", default=None, help="Override output.save_debug=true")
    p.add_argument("--no_save_debug", action="store_true", default=None, help="Override output.save_debug=false")
    p.add_argument("--keep_items", action="store_true", default=None, help="Keep per-frame items/ artifacts")
    p.add_argument("--keep_tracks_dir", action="store_true", default=None, help="Keep copied best-frame tracks/ directory")
    p.add_argument("--no_best_debug", action="store_true", default=None, help="Do not write compact debug_images best-frame overlays")
    p.add_argument("--save_frame_debug", action="store_true", default=None, help="Save heavy per-frame debug images under items/")
    p.add_argument("--write_frame_json", action="store_true", default=None, help="Save per-frame item JSON files under items/")
    p.add_argument("--write_debug_plots", action="store_true", default=None, help="Write per-track and global aggregation debug plots")
    p.add_argument("--no_debug_plots", action="store_true", default=None, help="Disable aggregation debug plots")
    p.add_argument("--no_split_tracks", action="store_true", default=None, help="Disable splitting of mixed source track folders")
    p.add_argument("--disable_digit_flip_alias", action="store_true", default=None, help="Do not merge 6/9 OCR flip variants such as 66.00 and 99.00 during aggregation/splitting")
    p.add_argument("--disable_false_tag_filter", action="store_true", default=None, help="Disable poster/banner false-candidate rejection")
    p.add_argument("--print_config", action="store_true", help="Print resolved config and exit")
    return p.parse_args()


def cli_overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    pairs = []
    pairs.append(("detected_tracks_dataset.root_dir", args.root_dir))
    pairs.append(("detected_tracks_dataset.out_dir", args.out_dir))
    pairs.append(("io.input_dir", args.root_dir))
    pairs.append(("io.out_dir", args.out_dir))
    pairs.append(("detected_tracks_dataset.recursive_images_in_track", args.recursive_images_in_track))
    pairs.append(("detected_tracks_dataset.min_images_per_track", args.min_images_per_track))
    pairs.append(("ocr.backend", args.ocr_backend))

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
        pairs.append(("detected_tracks_dataset.keep_items", True))
    if args.enable_csv_correction:
        pairs.append(("csv_corrector.enabled", True))
    if args.disable_csv_correction:
        pairs.append(("csv_corrector.enabled", False))
    pairs.append(("csv_corrector.catalog.path", args.csv_catalog))
    if args.disable_llm_correction:
        pairs.append(("llm_corrector.backend", "none"))
    if args.save_debug:
        pairs.append(("output.save_debug", True))
    if args.no_save_debug:
        pairs.append(("output.save_debug", False))
    if args.keep_items:
        pairs.append(("detected_tracks_dataset.keep_items", True))
    if args.keep_tracks_dir:
        pairs.append(("detected_tracks_dataset.keep_tracks_dir", True))
    if args.no_best_debug:
        pairs.append(("detected_tracks_dataset.copy_best_debug", False))
    if args.save_frame_debug:
        pairs.append(("detected_tracks_dataset.save_frame_debug", True))
        pairs.append(("detected_tracks_dataset.keep_items", True))
    if args.write_frame_json:
        pairs.append(("detected_tracks_dataset.write_frame_json", True))
        pairs.append(("detected_tracks_dataset.keep_items", True))
    if args.write_debug_plots:
        pairs.append(("detected_tracks_dataset.write_debug_plots", True))
    if args.no_debug_plots:
        pairs.append(("detected_tracks_dataset.write_debug_plots", False))
    if args.no_split_tracks:
        pairs.append(("track_aggregation.split_mixed_tracks", False))
    if args.disable_digit_flip_alias:
        pairs.append(("track_aggregation.digit_flip_alias_enabled", False))
    if args.disable_false_tag_filter:
        pairs.append(("track_aggregation.false_tag_filter_enabled", False))

    return apply_cli_overrides({}, pairs)


def main() -> int:
    args = parse_args()
    overrides = cli_overrides_from_args(args)
    cfg = load_config(args.config or None, overrides=overrides)

    if args.print_config:
        print(dump_config(cfg))
        return 0

    try:
        code_decoder = build_code_decoder(cfg)
        layout = build_layout_extractor(cfg, code_decoder=code_decoder)
        ocr = build_ocr_backend(cfg)
        template_classifier = build_template_classifier(cfg)
        result = process_detected_tracks_dataset(
            cfg,
            template_classifier=template_classifier,
            layout_detector=layout,
            ocr_backend=ocr,
            code_decoder=code_decoder,
        )
        print_detected_tracks_summary(result)
        return 0
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
