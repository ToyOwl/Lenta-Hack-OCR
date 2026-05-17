"""
End-to-end pipeline for Lenta-like price tags and shelf price rails.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from .code_decoder import CodeDecoder
from .csv_corrector import StructuredCSVOCRCorrector
from .config import DEFAULT_CONFIG, deep_merge
from .debug_vis import draw_debug_with_ocr
from .image_ops import crop_box
from .input_structure import InputStructureAnalyzer, InputStructureDecision, draw_structure_decision_debug
from .io_utils import imread_unicode, imwrite_unicode, safe_stem
from .layout import HeuristicLayoutExtractor
from .llm_corrector import LLMPriceTagCorrector
from .ocr_backends import OCRBackend
from .pipeline_types import OCRItem
from .price_parser import choose_main_price, parse_prices_from_texts
from .product_card import baseline_final_from_result, is_plausible_product_name_text
from .preprocess_glare import apply_glare_suppression_from_config
from .price_rail_splitter import PriceRailSplitter, RailCell
from .quality import compute_quality
from .spatial_parser import parse_full_tag_spatial
from .template_classifier import ColorNameTemplateClassifier
from .tilt_corrector import TiltCorrector


def _norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _best_text(items: List[OCRItem]) -> str:
    texts = [_norm_spaces(it.text) for it in items if _norm_spaces(it.text)]
    return " | ".join(texts)


def _digits_norm(s: str) -> str:
    repl = {
        "О": "0", "O": "0", "o": "0", "о": "0",
        "I": "1", "l": "1", "|": "1", "!": "1",
        "S": "5", "s": "5", ",": ".",
    }
    out = s or ""
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def _extract_price_string(s: str) -> str:
    s = _digits_norm(s)
    s_compact = re.sub(r"\s+", "", s)
    m = re.search(r"(\d{1,5})[\.\-](\d{2})", s_compact)
    if m:
        return f"{int(m.group(1))}.{m.group(2)}"
    for c in re.findall(r"\d{3,7}", s_compact):
        if len(c) >= 3:
            rub, kop = c[:-2], c[-2:]
            if rub:
                return f"{int(rub)}.{kop}"
    return ""


def _parse_fields(template_name: str, zone_ocr: Dict[str, List[OCRItem]]) -> Dict[str, Any]:
    if template_name == "hanging_yellow_promo_large":
        product_name = _best_text(zone_ocr.get("product_name", []))
        scale_raw = _best_text(zone_ocr.get("scale_number", []))
        old_raw = _best_text(zone_ocr.get("old_price", []))
        main_raw = _best_text(zone_ocr.get("main_price", []))
        rub_raw = _best_text(zone_ocr.get("main_price_rubles", []))
        kop_raw = _best_text(zone_ocr.get("main_price_kopeeks", []))
        scale_number = ""
        m = re.search(r"\d{2,4}", _digits_norm(scale_raw))
        if m:
            scale_number = m.group(0)
        main_price = ""
        rubs = re.findall(r"\d{1,5}", _digits_norm(rub_raw))
        kops = re.findall(r"\d{2}", _digits_norm(kop_raw))
        if rubs and kops:
            main_price = f"{int(rubs[0])}.{kops[0][:2]}"
        else:
            main_price = _extract_price_string(main_raw)
        return {
            "promo_header": _best_text(zone_ocr.get("promo_header", [])),
            "product_name": product_name,
            "scale_number": scale_number,
            "old_price_raw": old_raw,
            "old_price": _extract_price_string(old_raw),
            "main_price_raw": main_raw,
            "main_price_rubles_raw": rub_raw,
            "main_price_kopeeks_raw": kop_raw,
            "main_price": main_price,
            "footer_note": _best_text(zone_ocr.get("footer_note", [])),
        }

    if template_name in {"shelf_red_promo", "progressive", "progressive_yellow"}:
        main_raw = _best_text(zone_ocr.get("main_price", []))
        rub_raw = _best_text(zone_ocr.get("main_price_rubles", []))
        kop_raw = _best_text(zone_ocr.get("main_price_kopeeks", []))
        rubs = re.findall(r"\d{1,5}", _digits_norm(rub_raw))
        kops = re.findall(r"\d{2}", _digits_norm(kop_raw))
        if rubs and kops:
            main_price = f"{int(rubs[0])}.{kops[0][:2]}"
        else:
            main_price = _extract_price_string(main_raw)
        return {
            "product_name": _best_text(zone_ocr.get("product_name", [])),
            "card_price_small": _extract_price_string(_best_text(zone_ocr.get("card_price_small", []))),
            "no_card_price_small": _extract_price_string(_best_text(zone_ocr.get("no_card_price_small", []))),
            "main_price": main_price,
        }

    return {k: _best_text(v) for k, v in zone_ocr.items()}


def _result_summary(res: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "image_path": res.get("image_path"),
        "status": res.get("status"),
        "mode": res.get("mode", "single_tag"),
        "template_name": (res.get("template") or {}).get("template_name"),
        "template_confidence": (res.get("template") or {}).get("confidence"),
        "main_price": ((res.get("prices") or {}).get("main") or {}).get("value"),
        "final_main_price": (res.get("final") or {}).get("main_price"),
        "final_product_name": (res.get("final") or {}).get("product_name"),
        "llm_status": (res.get("llm_correction") or {}).get("status"),
        "needs_review": (res.get("final") or {}).get("needs_review"),
        "quality_status": (res.get("quality") or {}).get("status"),
        "error": res.get("error"),
    }


class PriceTagPipeline:
    def __init__(
        self,
        template_classifier: ColorNameTemplateClassifier,
        layout_detector: HeuristicLayoutExtractor,
        ocr_backend: OCRBackend,
        code_decoder: CodeDecoder,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.template_classifier = template_classifier
        self.layout_extractor = layout_detector
        self.ocr_backend = ocr_backend
        self.code_decoder = code_decoder
        self.config: Dict[str, Any] = deep_merge(DEFAULT_CONFIG, dict(config or {}))
        self.llm_corrector = LLMPriceTagCorrector.from_config(self.config)
        self.csv_corrector = StructuredCSVOCRCorrector.from_config(self.config)

    def process_image(self, image_path: Path, out_dir: Path) -> Dict[str, Any]:
        image = imread_unicode(image_path)
        if image is None:
            return {"image_path": str(image_path), "status": "error", "error": "failed_to_read_image"}

        stem = safe_stem(image_path)
        image_out_dir = out_dir / "items" / stem
        image_out_dir.mkdir(parents=True, exist_ok=True)

        rail_cfg = self.config.get("rail_segmentation", {})
        tilt_cfg = self.config.get("tilt_corrector", {})
        input_cfg = self.config.get("input_structure", {})

        input_mode = str(input_cfg.get("mode", "auto") or "auto").lower().strip()
        rail_enabled = bool(rail_cfg.get("enabled", False)) or input_mode in {"rail", "price_rail", "shelf_rail"}
        if input_mode in {"single", "single_tag", "tag"}:
            rail_enabled = False

        # Stage 0: optional TILT before structure routing.  We do not know yet
        # whether the input is a single tag or a shelf rail, but the splitter
        # must see approximately horizontal text/rail geometry.  The metadata is
        # carried forward so the image is not rotated twice.
        pre_tilt_applied = False
        if bool(tilt_cfg.get("enabled", False)) and bool(tilt_cfg.get("apply_before_structure", False)):
            image_struct, structure_tilt_meta = self._apply_tilt_if_needed(image, image_out_dir=image_out_dir, stem=f"{stem}_pre_structure")
            pre_tilt_applied = bool(structure_tilt_meta.get("applied", False))
            structure_tilt_meta["stage"] = "pre_structure"
        else:
            image_struct = image.copy()
            structure_tilt_meta = {
                "enabled": bool(tilt_cfg.get("enabled", False)),
                "applied": False,
                "angle": 0.0,
                "reason": "pre_structure_tilt_disabled",
                "input_shape": list(image.shape),
                "output_shape": list(image.shape),
                "stage": "pre_structure",
            }

        pre_split_result: Optional[Dict[str, Any]] = None
        input_decision: Optional[InputStructureDecision] = None
        if rail_enabled:
            splitter = PriceRailSplitter.from_config(dict(rail_cfg))
            pre_split_result = splitter.split(image_struct)
            analyzer = InputStructureAnalyzer.from_config(input_cfg)
            input_decision = analyzer.decide(image_struct, pre_split_result)

            if bool(input_cfg.get("save_debug", True)):
                try:
                    dbg = splitter.draw_debug(image_struct, pre_split_result)
                    dbg = draw_structure_decision_debug(dbg, input_decision)
                    imwrite_unicode(image_out_dir / f"{stem}_input_structure_debug.jpg", dbg)
                except Exception:
                    pass

            if input_decision.mode == "price_rail":
                if (not pre_tilt_applied) and bool(tilt_cfg.get("enabled", False)) and bool(tilt_cfg.get("apply_before_rail", False)):
                    image_proc, tilt_meta = self._apply_tilt_if_needed(image_struct, image_out_dir=image_out_dir, stem=stem)
                    tilt_meta["stage"] = "pre_rail"
                    pre_split_result = splitter.split(image_proc)
                else:
                    image_proc = image_struct.copy()
                    tilt_meta = dict(structure_tilt_meta)
                    if not tilt_meta.get("applied", False):
                        tilt_meta.setdefault("reason", "tilt_not_applied_before_rail")

                rail_result = self._process_price_rail(
                    image=image_proc,
                    image_path=image_path,
                    out_dir=out_dir,
                    parent_stem=stem,
                    image_out_dir=image_out_dir,
                    tilt_meta=tilt_meta,
                    split_result=pre_split_result,
                    input_decision=input_decision,
                )
                if rail_result is not None:
                    return rail_result

                # Conservative fallback: the router selected rail, but the rail
                # processor rejected the split.  If pre-structure TILT already
                # happened, process the same deskewed image as a single tag;
                # otherwise apply normal TILT now.
                if pre_tilt_applied:
                    image_proc = image_struct
                    tilt_meta = dict(structure_tilt_meta)
                else:
                    image_proc, tilt_meta = self._apply_tilt_if_needed(image, image_out_dir=image_out_dir, stem=stem)
                return self._process_single_tag_array(
                    image=image_proc,
                    image_path=image_path,
                    out_dir=out_dir,
                    stem=stem,
                    extra_meta={"tilt": tilt_meta, "input_structure": input_decision.to_dict()},
                )

        # Single-tag route.  Reuse pre-structure TILT if it already rotated the
        # image; otherwise run the ordinary single-tag TILT here.
        if pre_tilt_applied:
            image_proc = image_struct
            tilt_meta = dict(structure_tilt_meta)
        else:
            image_proc, tilt_meta = self._apply_tilt_if_needed(image, image_out_dir=image_out_dir, stem=stem)
        return self._process_single_tag_array(
            image=image_proc,
            image_path=image_path,
            out_dir=out_dir,
            stem=stem,
            extra_meta={
                "tilt": tilt_meta,
                "input_structure": input_decision.to_dict() if input_decision is not None else {"mode": "single_tag", "reason": "rail_router_disabled"},
            },
        )

    def _apply_glare_suppression_if_needed(self, image: np.ndarray, image_out_dir: Path, stem: str) -> tuple[np.ndarray, Dict[str, Any]]:
        gcfg = self.config.get("glare_suppression", {})
        if not bool(gcfg.get("enabled", False)):
            return image.copy(), {"enabled": False, "applied": False, "method": "none", "input_shape": list(image.shape), "output_shape": list(image.shape)}
        fixed, meta = apply_glare_suppression_from_config(
            image,
            self.config,
            debug_dir=image_out_dir,
            stem=stem,
        )
        return fixed, meta

    def _apply_tilt_if_needed(self, image: np.ndarray, image_out_dir: Path, stem: str) -> tuple[np.ndarray, Dict[str, Any]]:
        tcfg = self.config.get("tilt_corrector", {})
        if not bool(tcfg.get("enabled", False)):
            return image.copy(), {"enabled": False, "applied": False, "angle": 0.0}

        corrector = TiltCorrector(
            coarse_step=float(tcfg.get("coarse_step", 1.0)),
            fine_step=float(tcfg.get("fine_step", 0.2)),
            angle_range=float(tcfg.get("angle_range", 35.0)),
            min_text_height=int(tcfg.get("min_text_height", 12)),
            min_abs_angle=float(tcfg.get("min_abs_angle", 1.2)),
            max_work_side=int(tcfg.get("max_work_side", 900)),
            debug=bool(tcfg.get("debug", False)),
        )
        fixed, angle = corrector.correct(image)
        meta = corrector.get_debug_info()
        meta.update({"enabled": True, "angle": float(angle), "input_shape": list(image.shape), "output_shape": list(fixed.shape)})
        if bool(tcfg.get("save_tilt_debug", True)) and abs(angle) > 0.0:
            imwrite_unicode(image_out_dir / f"{stem}_tilt_fixed.jpg", fixed)
        return fixed, meta

    def _process_price_rail(
        self,
        image: np.ndarray,
        image_path: Path,
        out_dir: Path,
        parent_stem: str,
        image_out_dir: Path,
        tilt_meta: Dict[str, Any],
        split_result: Optional[Dict[str, Any]] = None,
        input_decision: Optional[InputStructureDecision] = None,
    ) -> Optional[Dict[str, Any]]:
        rail_cfg = self.config.get("rail_segmentation", {})
        splitter = PriceRailSplitter.from_config(dict(rail_cfg))
        if split_result is None:
            split_result = splitter.split(image)
        cell_objects: List[RailCell] = split_result.get("_cell_objects", []) or []

        if bool(rail_cfg.get("save_rail_debug", True)):
            try:
                dbg = splitter.draw_debug(image, split_result)
                imwrite_unicode(image_out_dir / f"{parent_stem}_rail_debug.jpg", dbg)
            except Exception:
                pass

        if len(cell_objects) < int(rail_cfg.get("min_cells", 2)):
            if bool(rail_cfg.get("fallback_to_full_tag", True)):
                return None
            return {
                "image_path": str(image_path),
                "status": "error",
                "mode": "price_rail",
                "error": "rail_segmentation_found_too_few_cells",
                "tilt": tilt_meta,
                "input_structure": input_decision.to_dict() if input_decision is not None else None,
                "rail_segmentation": self._public_split_result(split_result),
            }

        rail_cells_dir = image_out_dir / "rail_cells"
        if bool(rail_cfg.get("save_cell_crops", True)):
            rail_cells_dir.mkdir(parents=True, exist_ok=True)

        child_results: List[Dict[str, Any]] = []
        process_cells = bool(rail_cfg.get("process_cells", True))
        for cell in cell_objects:
            cell_img = crop_box(image, cell.bbox)
            child_stem = f"{parent_stem}_rail{cell.rail_index:02d}_cell{cell.cell_index:03d}_{cell.tag_type}"
            crop_rel = ""
            if bool(rail_cfg.get("save_cell_crops", True)):
                crop_path = rail_cells_dir / f"{child_stem}.jpg"
                imwrite_unicode(crop_path, cell_img)
                crop_rel = str(crop_path)

            if process_cells:
                child_out_dir = out_dir / "items" / child_stem
                child_out_dir.mkdir(parents=True, exist_ok=True)
                child_tilt_meta: Dict[str, Any]
                if bool(self.config.get("tilt_corrector", {}).get("enabled", False)) and bool(self.config.get("tilt_corrector", {}).get("apply_to_rail_cells", True)):
                    cell_img_proc, child_tilt_meta = self._apply_tilt_if_needed(cell_img, image_out_dir=child_out_dir, stem=child_stem)
                    child_tilt_meta["parent_tilt"] = tilt_meta
                else:
                    cell_img_proc = cell_img
                    child_tilt_meta = {"enabled": False, "applied": False, "angle": 0.0, "parent_tilt": tilt_meta}
                child_meta = {
                    "tilt": child_tilt_meta,
                    "rail_cell": cell.to_dict(),
                    "source_image_path": str(image_path),
                    "saved_cell_crop": crop_rel,
                }
                child_res = self._process_single_tag_array(
                    image=cell_img_proc,
                    image_path=image_path,
                    out_dir=out_dir,
                    stem=child_stem,
                    extra_meta=child_meta,
                    virtual_image_path=f"{image_path}#rail{cell.rail_index:02d}_cell{cell.cell_index:03d}",
                )
                child_summary = _result_summary(child_res) | {"rail_cell": cell.to_dict(), "saved_cell_crop": crop_rel}
                # Keep enough child data for track-level aggregation and best-frame
                # selection without forcing a second read of per-cell JSON files.
                for key in ("quality", "template", "final", "prices", "parsed", "csv_correction", "llm_correction"):
                    if key in child_res:
                        child_summary[key] = child_res.get(key)
                child_results.append(child_summary)
            else:
                child_results.append({"status": "not_processed", "rail_cell": cell.to_dict(), "saved_cell_crop": crop_rel})

        parent_result: Dict[str, Any] = {
            "image_path": str(image_path),
            "status": "ok",
            "mode": "price_rail",
            "tilt": tilt_meta,
            "input_structure": input_decision.to_dict() if input_decision is not None else None,
            "rail_segmentation": self._public_split_result(split_result),
            "cell_results": child_results,
        }

        if bool(self.config.get("output", {}).get("write_item_json", True)):
            with open(image_out_dir / f"{parent_stem}.json", "w", encoding="utf-8") as f:
                json.dump(parent_result, f, ensure_ascii=False, indent=2)
        return parent_result

    @staticmethod
    def _public_split_result(split_result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "rails": split_result.get("rails", []),
            "cells": split_result.get("cells", []),
            "debug_boxes": split_result.get("debug_boxes", []),
        }

    def _process_single_tag_array(
        self,
        image: np.ndarray,
        image_path: Path,
        out_dir: Path,
        stem: str,
        extra_meta: Optional[Dict[str, Any]] = None,
        virtual_image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        if image is None or image.size == 0:
            return {"image_path": virtual_image_path or str(image_path), "status": "error", "error": "empty_image"}

        h, w = image.shape[:2]
        image_out_dir = out_dir / "items" / stem
        image_out_dir.mkdir(parents=True, exist_ok=True)

        image, glare_meta = self._apply_glare_suppression_if_needed(image, image_out_dir=image_out_dir, stem=stem)

        qcfg = self.config.get("quality", {})
        quality = compute_quality(
            image,
            min_w=int(qcfg.get("min_width", 120)),
            min_h=int(qcfg.get("min_height", 80)),
            blur_warn=float(qcfg.get("blur_warn", 60.0)),
        )
        template = self.template_classifier.classify(image, quality=quality)
        boxes = self.layout_extractor.extract(image, template.to_dict())

        zones_cfg = self.config.get("zones", {})
        output_cfg = self.config.get("output", {})
        ocr_cfg = self.config.get("ocr", {})
        price_cfg = self.config.get("price_parser", {})

        code_classes = set(zones_cfg.get("code_classes", ["qr_code", "linear_barcode", "barcode", "datamatrix"]))
        ocr_exclude = set(zones_cfg.get("ocr_exclude_classes", list(code_classes)))
        compact_prefixes = tuple(zones_cfg.get("price_compact_classes_prefixes", ["main_price"]))
        crop_expand_px = int(zones_cfg.get("crop_expand_px", 3))

        code_hint_boxes = [b for b in boxes if b.cls in code_classes]
        if bool(self.config.get("code_decoder", {}).get("enabled", True)):
            codes = self.code_decoder.decode(image, hint_boxes=code_hint_boxes)
        else:
            codes = []

        ocr_zones = [b for b in boxes if b.cls not in ocr_exclude]
        zone_prices: Dict[str, List[Dict[str, Any]]] = {}
        zone_ocr: Dict[str, List[OCRItem]] = {}
        crop_records: List[Dict[str, Any]] = []

        save_crops = bool(output_cfg.get("save_crops", True))
        ocr_jobs: List[Dict[str, Any]] = []
        for idx, b in enumerate(ocr_zones):
            crop = crop_box(image, b.expand(w, h, px=crop_expand_px))
            if crop.size == 0:
                continue
            if save_crops:
                imwrite_unicode(image_out_dir / "crops" / f"{idx:02d}_{b.cls}.jpg", crop)
            allow_compact = bool(price_cfg.get("allow_compact_in_main_price_zones", True)) and b.cls.startswith(compact_prefixes)
            ocr_jobs.append({
                "kind": "zone",
                "idx": idx,
                "class": b.cls,
                "bbox": b.to_xyxy(),
                "conf": b.conf,
                "source": b.source,
                "image": crop,
                "allow_compact": allow_compact,
            })

        full_items: List[OCRItem] = []
        full_debug: Dict[str, Any] = {}
        if bool(ocr_cfg.get("full_tag_ocr", True)):
            ocr_jobs.append({
                "kind": "full_tag",
                "class": "full_tag",
                "image": image,
                "allow_compact": bool(price_cfg.get("allow_compact_in_full_tag", False)),
            })

        batch_enabled = bool(ocr_cfg.get("batch_inference", True))
        min_batch_jobs = max(2, int(ocr_cfg.get("min_batch_jobs", 2)))
        batch_size = max(1, int(ocr_cfg.get("batch_size", 16)))
        if batch_enabled and len(ocr_jobs) >= min_batch_jobs and hasattr(self.ocr_backend, "recognize_batch"):
            batch_items = self.ocr_backend.recognize_batch(
                [job["image"] for job in ocr_jobs],
                zones=[job.get("class") for job in ocr_jobs],
                batch_size=batch_size,
            )
            batch_debug = self.ocr_backend.get_debug_info() if bool(ocr_cfg.get("debug", False)) else {}
        else:
            batch_items = []
            for job in ocr_jobs:
                batch_items.append(self.ocr_backend.recognize(job["image"], zone=job.get("class")))
            batch_debug = self.ocr_backend.get_debug_info() if bool(ocr_cfg.get("debug", False)) else {}

        for job_index, (job, items) in enumerate(zip(ocr_jobs, batch_items)):
            items = list(items or [])
            prices = parse_prices_from_texts([it.text for it in items], allow_compact=bool(job.get("allow_compact", False)))
            if job.get("kind") == "full_tag":
                full_items = items
                full_debug = {"batch_job_index": job_index, **batch_debug} if bool(ocr_cfg.get("debug", False)) else {}
                if prices:
                    zone_prices["full_tag"] = prices
                continue

            cls_name = str(job.get("class") or "")
            zone_ocr.setdefault(cls_name, []).extend(items)
            if prices:
                zone_prices.setdefault(cls_name, []).extend(prices)
            crop_records.append({
                "class": cls_name,
                "bbox": job.get("bbox"),
                "conf": job.get("conf"),
                "source": job.get("source"),
                "ocr": [it.to_dict() for it in items],
                "ocr_debug": {"batch_job_index": job_index, **batch_debug} if bool(ocr_cfg.get("debug", False)) else {},
                "price_candidates": prices,
            })

        main_price = choose_main_price(zone_prices, template.template_name)
        parsed_fields = _parse_fields(template.template_name, zone_ocr)

        spatial_cfg = self.config.get("spatial_parser", {})
        spatial_result: Dict[str, Any] = {"fields": {}, "semantic_boxes": [], "price_candidates": [], "geometry": {}, "ocr_lines": []}
        semantic_boxes = []
        if bool(spatial_cfg.get("enabled", True)) and full_items:
            spatial_result = parse_full_tag_spatial(image, template.template_name, full_items)
            semantic_boxes = spatial_result.get("semantic_boxes", []) or []
            spatial_fields = spatial_result.get("fields", {}) or {}
            product_as_prior_only = bool(spatial_cfg.get("product_as_ocr_prior", True))
            for k, v in spatial_fields.items():
                if v in (None, ""):
                    continue
                if str(k) == "product_name" and product_as_prior_only:
                    # Spatial parser is a geometric/OCR prior.  It must not
                    # blindly overwrite product_name because false service text
                    # and dense OCR garbage then leak into DB matching.
                    if is_plausible_product_name_text(v):
                        parsed_fields.setdefault("spatial_product_name_prior", v)
                    else:
                        parsed_fields.setdefault("spatial_product_name_rejected", v)
                    continue
                parsed_fields[k] = v

        if parsed_fields.get("main_price"):
            try:
                rub, kop = str(parsed_fields["main_price"]).split(".")
                main_price = {
                    "value": float(parsed_fields["main_price"]),
                    "rubles": int(rub),
                    "kopeeks": int(kop),
                    "raw_match": str(parsed_fields["main_price"]),
                    "parser": "ocr_spatial_parser" if semantic_boxes else "template_field_parser",
                    "confidence": 0.88 if semantic_boxes else 0.70,
                    "zone": "full_tag_ocr_spatial" if semantic_boxes else "template_fields",
                    "type": "with_card_or_promo_main" if template.template_name in {"shelf_red_promo", "progressive", "progressive_yellow"} else "main_or_unknown",
                }
            except Exception:
                pass

        all_text: List[str] = []
        for rec in crop_records:
            all_text.extend([it.get("text", "") for it in rec.get("ocr", [])])
        all_text.extend([it.text for it in full_items])

        result: Dict[str, Any] = {
            "image_path": virtual_image_path or str(image_path),
            "status": "ok",
            "mode": "single_tag",
            "quality": quality,
            "template": template.to_dict(),
            "layout": [b.to_dict() for b in boxes],
            "codes": [c.to_dict() for c in codes],
            "ocr": {
                "backend": getattr(self.ocr_backend, "init_api", type(self.ocr_backend).__name__),
                "zones": crop_records,
                "zone_texts": {k: _best_text(v) for k, v in zone_ocr.items()},
                "full_tag": [it.to_dict() for it in full_items],
                "full_tag_debug": full_debug,
                "all_text_joined": " | ".join([t for t in all_text if t]),
            },
            "parsed": parsed_fields,
            "spatial": {
                "fields": spatial_result.get("fields", {}),
                "semantic_boxes": [b.to_dict() for b in semantic_boxes],
                "price_candidates": spatial_result.get("price_candidates", []),
                "geometry": spatial_result.get("geometry", {}),
                "ocr_lines": spatial_result.get("ocr_lines", []),
            },
            "prices": {
                "main": main_price,
                "by_zone": zone_prices,
            },
            "glare_suppression": glare_meta,
        }
        if extra_meta:
            result.update(extra_meta)

        # Optional LLM/catalog post-correction.  It is placed after baseline OCR,
        # spatial parser and rail metadata have all been collected, but before
        # JSON/debug writing so the overlay can display final fields.
        llm_result = self.llm_corrector.correct_result(result)
        if llm_result is not None:
            result["llm_correction"] = llm_result.to_dict()
            if llm_result.final:
                result["final"] = llm_result.final
        if "final" not in result:
            # Keep the product-card schema stable even when llm_corrector.backend=none.
            result["final"] = baseline_final_from_result(result)

        # Optional deterministic correction from structured CSV.  This stage is
        # deliberately separate from the LLM corrector and can be used with
        # llm_corrector.backend=none for fully offline/product-catalog correction.
        csv_result = self.csv_corrector.correct_result(result)
        if csv_result is not None:
            result["csv_correction"] = csv_result
            if csv_result.get("final"):
                result["final"] = csv_result["final"]

        if bool(output_cfg.get("write_item_json", True)):
            with open(image_out_dir / f"{stem}.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        if bool(output_cfg.get("save_debug", True)):
            combined = template.color_features.get("combined", {})
            title_lines = [
                f"template={template.template_name} conf={template.confidence:.2f} quality={quality['status']}",
                f"main_price={main_price.get('value') if main_price else None} codes_ok={sum(1 for c in codes if c.decoded)}",
                f"CN: bottom_warm={combined.get('bottom_red_orange_brown_pink', 0):.2f} bottom_cool={combined.get('bottom_cool_blue_purple', 0):.2f} top_yellow={combined.get('top_yellow_orange_brown', 0):.2f}",
            ]
            debug_boxes = semantic_boxes if bool(self.config.get("spatial_parser", {}).get("draw_semantic_boxes", True)) and semantic_boxes else boxes
            dbg = draw_debug_with_ocr(
                image,
                debug_boxes,
                codes,
                title_lines,
                ocr_by_zone=zone_ocr,
                parsed=result.get("final") or parsed_fields,
                full_ocr_items=full_items if bool(self.config.get("spatial_parser", {}).get("draw_full_ocr", True)) else [],
            )
            imwrite_unicode(image_out_dir / f"{stem}_debug.jpg", dbg)

        return result
