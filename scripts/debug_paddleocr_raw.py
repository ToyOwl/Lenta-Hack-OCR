"""Dump raw PaddleOCR 3.x/2.x output for a single image.

Useful when PaddleOCR creates models successfully but pipeline JSON contains empty OCR.

Examples:
  python scripts/debug_paddleocr_raw.py --image images/00003.png --lang ru --ocr_version PP-OCRv5
  python scripts/debug_paddleocr_raw.py --image images/00003.png --input_mode path --out runs/raw_ocr_debug
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from price_tag_pipeline.io_utils import imread_unicode, imwrite_unicode  # noqa: E402
from price_tag_pipeline.image_ops import normalize_for_ocr  # noqa: E402
from price_tag_pipeline.ocr_backends import parse_paddle_result, summarize_raw_result, safe_repr  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--out", default="runs/paddleocr_raw_debug")
    p.add_argument("--lang", default="ru")
    p.add_argument("--ocr_version", default="PP-OCRv5")
    p.add_argument("--device", default="cpu")
    p.add_argument("--input_mode", choices=["ndarray", "path"], default="ndarray")
    p.add_argument("--no_preprocess", action="store_true")
    p.add_argument("--use_textline_orientation", action="store_true")
    p.add_argument("--mkldnn", action="store_true", help="Enable MKLDNN/oneDNN. Default: false due to PaddlePaddle 3.3.x CPU instability on some hosts.")
    p.add_argument("--pir_api", action="store_true", help="Enable Paddle PIR API. Default: false for legacy OCR compatibility.")
    return p.parse_args()


def json_safe(obj):
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "preview": obj.reshape(-1)[:20].tolist() if obj.size <= 10000 else [],
        }
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return safe_repr(obj, 1000)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.pir_api:
        os.environ.setdefault("FLAGS_enable_pir_api", "0")
    if not args.mkldnn:
        os.environ.setdefault("FLAGS_use_onednn", "0")
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        os.environ.setdefault("MKLDNN_DISABLE_WORKSPACE", "1")

    from paddleocr import PaddleOCR

    img = imread_unicode(Path(args.image))
    if img is None:
        raise SystemExit(f"failed to read image: {args.image}")

    if not args.no_preprocess:
        img = normalize_for_ocr(img, clahe=True, sharpen=True)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    input_obj = img
    input_desc = "ndarray"
    if args.input_mode == "path":
        tmp_path = out_dir / "input_for_paddle.png"
        imwrite_unicode(tmp_path, img)
        input_obj = str(tmp_path)
        input_desc = str(tmp_path)

    kwargs = dict(
        lang=args.lang,
        ocr_version=args.ocr_version,
        device=args.device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=bool(args.use_textline_orientation),
        enable_mkldnn=bool(args.mkldnn),
    )
    while True:
        try:
            ocr = PaddleOCR(**kwargs)
            break
        except (TypeError, ValueError) as e:
            msg = str(e)
            if "enable_mkldnn" in kwargs and ("enable_mkldnn" in msg or "Unknown argument" in msg):
                kwargs.pop("enable_mkldnn", None)
                continue
            raise

    result = ocr.predict(input_obj)
    items = parse_paddle_result(result, zone="debug")

    summary = {
        "image": str(args.image),
        "input_desc": input_desc,
        "input_shape": list(img.shape),
        "raw_summary": summarize_raw_result(result, max_items=10, max_repr=1500),
        "parsed_items": [x.to_dict() for x in items],
    }

    with open(out_dir / "paddleocr_raw_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=json_safe)

    with open(out_dir / "paddleocr_raw_repr.txt", "w", encoding="utf-8") as f:
        f.write(safe_repr(result, 20000))

    print(json.dumps(summary["parsed_items"], ensure_ascii=False, indent=2))
    print(f"[OK] out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
