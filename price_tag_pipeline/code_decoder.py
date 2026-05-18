# -*- coding: utf-8 -*-
"""QR/barcode decoding for price tag crops.

The decoder is deliberately conservative for automatic payload acceptance, but
aggressive in *attempting* QR/barcode recovery.  Real shelf-label crops are small,
blurred and often contain a QR in the upper-right area that layout extraction does
not always mark as a separate box.  Therefore ``decode`` tries:

- explicit layout hint boxes;
- full crop;
- several likely QR ROIs, primarily top/right quadrants;
- multiple preprocessing variants: upscaled, CLAHE, sharpened, Otsu/adaptive
  threshold and inverted threshold.

Decoded payloads are deduplicated.  Undecoded QR detections are kept only as weak
visual evidence, so debug overlays can show that QR detection was attempted.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .image_ops import crop_box, normalize_for_ocr, upscale_if_small
from .pipeline_types import Box, DecodedCode
from .tag_qr_enhancement import make_qr_preprocess_variants, warp_qr_from_points


class CodeDecoder:
    def __init__(
        self,
        use_pyzbar: bool = True,
        try_opencv_qr: bool = True,
        *,
        qr_roi_scan: bool = True,
        preprocessing_variants: bool = True,
        keep_undecoded_qr: bool = True,
        qr_sr_enabled: bool = True,
        qr_sr_scale: float = 2.0,
        qr_sr_min_side: int = 420,
        qr_sr_max_side: int = 1400,
        qr_sr_method: str = "lanczos",
        qr_perspective_warp: bool = True,
        qr_morphology: bool = True,
        max_rois: int = 6,
        max_variants_per_roi: int = 10,
        qr_contour_max_rois: int = 4,
    ) -> None:
        self.try_opencv_qr = bool(try_opencv_qr)
        self.qr_roi_scan = bool(qr_roi_scan)
        self.preprocessing_variants = bool(preprocessing_variants)
        self.keep_undecoded_qr = bool(keep_undecoded_qr)
        self.qr_sr_enabled = bool(qr_sr_enabled)
        self.qr_sr_scale = float(qr_sr_scale)
        self.qr_sr_min_side = int(qr_sr_min_side)
        self.qr_sr_max_side = int(qr_sr_max_side)
        self.qr_sr_method = str(qr_sr_method or "lanczos")
        self.qr_perspective_warp = bool(qr_perspective_warp)
        self.qr_morphology = bool(qr_morphology)
        self.max_rois = max(1, int(max_rois))
        self.max_variants_per_roi = max(1, int(max_variants_per_roi))
        self.qr_contour_max_rois = max(0, int(qr_contour_max_rois))
        self.pyzbar_decode = None
        if use_pyzbar:
            try:
                from pyzbar.pyzbar import decode as pyzbar_decode  # type: ignore
                self.pyzbar_decode = pyzbar_decode
            except Exception:
                self.pyzbar_decode = None
        self.qr_detector = cv2.QRCodeDetector()
        self.last_debug: Dict[str, Any] = {}

    def decode(self, image: np.ndarray, hint_boxes: Optional[List[Box]] = None) -> List[DecodedCode]:
        self.last_debug = {"attempts": [], "roi_count": 0, "decoded_count": 0, "detected_count": 0}
        results: List[DecodedCode] = []
        if image is None or image.size == 0:
            return results
        h, w = image.shape[:2]

        seen_rois: List[Tuple[int, int, int, int, str]] = []
        if hint_boxes:
            for b in hint_boxes:
                bb = b.expand(w, h, px=8).clip(w, h)
                if bb.area <= 0:
                    continue
                seen_rois.append((bb.x1, bb.y1, bb.x2, bb.y2, str(b.cls or "hint")))

        # Always try the whole tag; OpenCV QR can sometimes recover warped QR
        # better from context than from a tight crop.
        seen_rois.append((0, 0, w, h, "full_tag"))

        if self.qr_roi_scan:
            for b in self._candidate_qr_rois(image):
                seen_rois.append((b.x1, b.y1, b.x2, b.y2, b.cls))

        # Deduplicate ROIs by approximate coordinates.
        uniq: List[Tuple[int, int, int, int, str]] = []
        keys = set()
        for x1, y1, x2, y2, kind in seen_rois:
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            key = (round(x1 / 12), round(y1 / 12), round(x2 / 12), round(y2 / 12), kind)
            if key in keys:
                continue
            keys.add(key)
            uniq.append((x1, y1, x2, y2, kind))

        uniq.sort(key=lambda r: self._roi_priority(r, w=w, h=h))
        if len(uniq) > self.max_rois:
            uniq = uniq[: self.max_rois]
        self.last_debug["roi_count"] = len(uniq)
        for x1, y1, x2, y2, kind in uniq:
            crop = image[y1:y2, x1:x2]
            results.extend(self._decode_crop(crop, offset=(x1, y1), kind_hint=kind))

        deduped = self._dedupe(results)
        self.last_debug["decoded_count"] = sum(1 for r in deduped if r.decoded)
        self.last_debug["detected_count"] = len(deduped)
        return deduped

    def detect_qr_boxes(self, image: np.ndarray) -> List[Box]:
        if image is None or image.size == 0:
            return []
        h, w = image.shape[:2]
        boxes: List[Box] = []
        for variant_name, im, scale in self._preprocess_variants(image):
            try:
                ok, pts = self.qr_detector.detectMulti(im)
                if ok and pts is not None:
                    for p in pts:
                        boxes.append(self._points_to_box(p, scale=scale, offset=(0, 0), conf=0.65, source=f"opencv_qr_detect:{variant_name}"))
                else:
                    ok2, pts2 = self.qr_detector.detect(im)
                    if ok2 and pts2 is not None:
                        boxes.append(self._points_to_box(pts2, scale=scale, offset=(0, 0), conf=0.60, source=f"opencv_qr_detect:{variant_name}"))
            except Exception:
                continue
        return self._dedupe_boxes([b.clip(w, h) for b in boxes if b.area > 0])

    def get_debug_info(self) -> Dict[str, Any]:
        return dict(self.last_debug or {})

    def _candidate_qr_rois(self, image: np.ndarray) -> List[Box]:
        """Return likely QR boxes even if layout did not find them.

        Russian shelf labels usually place QR/DataMatrix in the upper-right area;
        barcodes are usually lower-right/lower-center.  These boxes are not final
        detections, only decode attempts.
        """
        h, w = image.shape[:2]
        boxes: List[Box] = []
        if w < 40 or h < 40:
            return boxes
        boxes.extend([
            Box("qr_roi_top_right", int(w * 0.52), 0, w, int(h * 0.72), 0.25, "qr_roi_scan"),
            Box("qr_roi_upper_right_square", int(w * 0.58), 0, w, min(h, int(w * 0.48)), 0.25, "qr_roi_scan"),
            Box("qr_roi_right_mid", int(w * 0.52), int(h * 0.15), w, int(h * 0.82), 0.20, "qr_roi_scan"),
            Box("barcode_roi_bottom", int(w * 0.20), int(h * 0.55), w, h, 0.18, "qr_roi_scan"),
        ])
        # Also try detected QR-like square components from binarized image.
        boxes.extend(self._contour_qr_like_rois(image))
        return [b.clip(w, h) for b in boxes if b.area > 0]

    def _roi_priority(self, roi: Tuple[int, int, int, int, str], *, w: int, h: int) -> Tuple[int, float]:
        x1, y1, x2, y2, kind = roi
        cx = (x1 + x2) * 0.5 / max(1, w)
        cy = (y1 + y2) * 0.5 / max(1, h)
        area = max(1, (x2 - x1) * (y2 - y1)) / max(1, w * h)
        order = {
            "full_tag": 0,
            "qr_roi_top_right": 1,
            "qr_roi_upper_right_square": 2,
            "qr_roi_right_mid": 3,
            "barcode_roi_bottom": 4,
            "qr_roi_contour": 5,
        }
        # Prefer upper-right square-ish contours, but keep full_tag first.
        right_top_bonus = abs(1.0 - cx) + cy
        size_penalty = abs(area - 0.16)
        return (order.get(kind, 9), float(right_top_bonus + 0.35 * size_penalty))

    def _contour_qr_like_rois(self, image: np.ndarray) -> List[Box]:
        h, w = image.shape[:2]
        out: List[Box] = []
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7)
            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                x, y, ww, hh = cv2.boundingRect(c)
                if ww < 18 or hh < 18:
                    continue
                ar = ww / max(1, hh)
                area = ww * hh
                if 0.55 <= ar <= 1.8 and area >= max(400, int(0.004 * w * h)):
                    # Expand aggressively because QR finder patterns may be only
                    # a part of the full code.
                    pad = int(max(8, 0.35 * max(ww, hh)))
                    out.append(Box("qr_roi_contour", x - pad, y - pad, x + ww + pad, y + hh + pad, 0.18, "qr_contour_roi"))
        except Exception:
            return []
        out.sort(key=lambda b: (0 if b.x1 > int(w * 0.45) and b.y1 < int(h * 0.72) else 1, -b.area))
        return out[: self.qr_contour_max_rois]

    def _decode_crop(self, image: np.ndarray, offset: Tuple[int, int], kind_hint: str) -> List[DecodedCode]:
        out: List[DecodedCode] = []
        if image is None or image.size == 0:
            return out
        ox, oy = offset
        variants = self._preprocess_variants(image)
        if len(variants) > self.max_variants_per_roi:
            variants = variants[: self.max_variants_per_roi]
        for variant_name, img2, scale in variants:
            self.last_debug.setdefault("attempts", []).append({
                "roi": kind_hint,
                "variant": variant_name,
                "shape": list(img2.shape[:2]),
            })
            if self.try_opencv_qr:
                out.extend(self._decode_opencv_qr(img2, scale=scale, offset=(ox, oy), decoder_suffix=variant_name))
            if self.pyzbar_decode is not None:
                out.extend(self._decode_pyzbar(img2, scale=scale, offset=(ox, oy), decoder_suffix=variant_name))
            if any(r.decoded for r in out):
                # Do not waste time once a payload is found for this crop.
                break
        return out

    def _preprocess_variants(self, image: np.ndarray) -> List[Tuple[str, np.ndarray, float]]:
        variants: List[Tuple[str, np.ndarray, float]] = []
        img2, scale = upscale_if_small(image, min_side=340, max_scale=5)
        if img2.ndim == 2:
            img2_bgr = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        else:
            img2_bgr = img2
        variants.append(("upscaled", img2_bgr, scale))

        norm = normalize_for_ocr(img2_bgr, clahe=True, sharpen=True)
        variants.append(("ocr_norm", norm, scale))

        if not self.preprocessing_variants:
            return variants

        gray = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY) if img2_bgr.ndim == 3 else img2_bgr.copy()
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(6, 6)).apply(gray)
        sharp = cv2.addWeighted(clahe, 1.55, cv2.GaussianBlur(clahe, (0, 0), 1.0), -0.55, 0)
        variants.append(("gray_clahe_sharp", sharp, scale))

        try:
            _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append(("otsu", otsu, scale))
            variants.append(("otsu_inv", 255 - otsu, scale))
        except Exception:
            pass
        try:
            # Block size must be odd and not larger than image side.
            block = max(15, min(51, (min(gray.shape[:2]) // 4) | 1))
            adap = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 5)
            variants.append(("adaptive", adap, scale))
            variants.append(("adaptive_inv", 255 - adap, scale))
        except Exception:
            pass

        # QR-specific SR + denoise + morphology variants.  These are added after
        # the generic OCR variants so normal decoding remains cheap when the QR is
        # already readable, but blurred tracks get a stronger fallback.
        try:
            for name, qr_img, qr_scale in make_qr_preprocess_variants(
                image,
                sr_enabled=self.qr_sr_enabled,
                sr_scale=self.qr_sr_scale,
                sr_min_side=self.qr_sr_min_side,
                sr_max_side=self.qr_sr_max_side,
                sr_method=self.qr_sr_method,
                morphology=self.qr_morphology,
            ):
                if name == "qr_raw":
                    continue
                variants.append((name, qr_img, float(qr_scale)))
        except Exception:
            pass
        return variants

    def _decode_opencv_qr(self, image: np.ndarray, scale: float, offset: Tuple[int, int], decoder_suffix: str = "") -> List[DecodedCode]:
        ox, oy = offset
        out: List[DecodedCode] = []
        decoder_name = "opencv_qr" + (f":{decoder_suffix}" if decoder_suffix else "")
        try:
            retval, decoded_info, points, _ = self.qr_detector.detectAndDecodeMulti(image)
            if retval and points is not None:
                for payload, pts in zip(decoded_info, points):
                    out.append(self._decoded_qr_from_points(payload, pts, scale, (ox, oy), decoder=f"opencv_qr_multi:{decoder_suffix}"))
                    if not payload and self.qr_perspective_warp:
                        out.extend(self._decode_warped_qr(image, pts, scale=scale, offset=(ox, oy), decoder_suffix=decoder_suffix))
            else:
                payload, pts, _ = self.qr_detector.detectAndDecode(image)
                if pts is not None:
                    out.append(self._decoded_qr_from_points(payload, pts, scale, (ox, oy), decoder=decoder_name))
                    if not payload and self.qr_perspective_warp:
                        out.extend(self._decode_warped_qr(image, pts, scale=scale, offset=(ox, oy), decoder_suffix=decoder_suffix))
                # Curved decoder is slower but sometimes helps on perspective/noisy crops.
                if not payload and hasattr(self.qr_detector, "detectAndDecodeCurved"):
                    payload2, pts2, _ = self.qr_detector.detectAndDecodeCurved(image)
                    if pts2 is not None:
                        out.append(self._decoded_qr_from_points(payload2, pts2, scale, (ox, oy), decoder=f"opencv_qr_curved:{decoder_suffix}"))
                        if not payload2 and self.qr_perspective_warp:
                            out.extend(self._decode_warped_qr(image, pts2, scale=scale, offset=(ox, oy), decoder_suffix=f"curved:{decoder_suffix}"))
        except Exception:
            pass
        if not self.keep_undecoded_qr:
            out = [r for r in out if r.decoded]
        return out

    def _decode_warped_qr(self, image: np.ndarray, pts: Any, *, scale: float, offset: Tuple[int, int], decoder_suffix: str = "") -> List[DecodedCode]:
        """Try perspective-normalized QR when OpenCV found points but no payload."""
        out: List[DecodedCode] = []
        ox, oy = offset
        try:
            warped, rect = warp_qr_from_points(image, pts, border=18, min_size=192, max_size=768)
            warp_variants: List[Tuple[str, np.ndarray]] = [("warp_raw", warped)]
            for name, v, _ in make_qr_preprocess_variants(
                warped,
                sr_enabled=False,
                morphology=self.qr_morphology,
            ):
                if name != "qr_raw":
                    warp_variants.append(("warp_" + name, v))
            seen_names = set()
            for name, wimg in warp_variants:
                if name in seen_names:
                    continue
                seen_names.add(name)
                payload, warped_pts, _ = self.qr_detector.detectAndDecode(wimg)
                if payload:
                    out.append(self._decoded_qr_from_points(payload, pts, scale, (ox, oy), decoder=f"opencv_qr_perspective:{decoder_suffix}:{name}"))
                    break
                if self.pyzbar_decode is not None:
                    pyz = self._decode_pyzbar(wimg, scale=1.0, offset=(0, 0), decoder_suffix=f"perspective:{decoder_suffix}:{name}")
                    for r in pyz:
                        if r.decoded:
                            # pyzbar bbox is in warped coordinates.  For the track
                            # result, keep the original QR detector bbox.
                            out.append(self._decoded_qr_from_points(r.payload, pts, scale, (ox, oy), decoder=f"pyzbar_perspective:{decoder_suffix}:{name}"))
                            return out
        except Exception:
            return []
        return out

    def _decoded_qr_from_points(self, payload: Any, pts: Any, scale: float, offset: Tuple[int, int], *, decoder: str) -> DecodedCode:
        ox, oy = offset
        pts2 = np.asarray(pts, dtype=np.float32).reshape(-1, 2) / max(1e-6, float(scale))
        x1, y1 = np.min(pts2, axis=0)
        x2, y2 = np.max(pts2, axis=0)
        p = str(payload or "")
        return DecodedCode(
            kind="qr_code",
            decoded=bool(p),
            payload=p,
            fmt="QR_CODE",
            conf=0.82 if p else 0.32,
            bbox=[max(0, int(x1 + ox)), max(0, int(y1 + oy)), max(0, int(x2 + ox)), max(0, int(y2 + oy))],
            decoder=decoder,
        )

    def _decode_pyzbar(self, image: np.ndarray, scale: float, offset: Tuple[int, int], decoder_suffix: str = "") -> List[DecodedCode]:
        ox, oy = offset
        out: List[DecodedCode] = []
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            decoded = self.pyzbar_decode(gray)
            for d in decoded:
                payload = d.data.decode("utf-8", errors="replace") if d.data else ""
                fmt = str(d.type or "")
                x, y, ww, hh = d.rect.left, d.rect.top, d.rect.width, d.rect.height
                kind = "qr_code" if "QR" in fmt.upper() else "barcode"
                out.append(DecodedCode(
                    kind=kind,
                    decoded=bool(payload),
                    payload=payload,
                    fmt=fmt,
                    conf=0.92 if payload else 0.4,
                    bbox=[max(0, int(x / scale + ox)), max(0, int(y / scale + oy)), max(0, int((x + ww) / scale + ox)), max(0, int((y + hh) / scale + oy))],
                    decoder="pyzbar" + (f":{decoder_suffix}" if decoder_suffix else ""),
                ))
        except Exception:
            pass
        return out

    @staticmethod
    def _points_to_box(pts: Any, *, scale: float, offset: Tuple[int, int], conf: float, source: str) -> Box:
        ox, oy = offset
        p = np.asarray(pts, dtype=np.float32).reshape(-1, 2) / max(1e-6, float(scale))
        x1, y1 = np.min(p, axis=0)
        x2, y2 = np.max(p, axis=0)
        return Box("qr_code", max(0, int(x1 + ox)), max(0, int(y1 + oy)), max(0, int(x2 + ox)), max(0, int(y2 + oy)), float(conf), source)

    @staticmethod
    def _dedupe_boxes(boxes: Sequence[Box]) -> List[Box]:
        out: List[Box] = []
        for b in boxes:
            duplicate = False
            for p in out:
                if _box_iou(b, p) > 0.55:
                    duplicate = True
                    break
            if not duplicate:
                out.append(b)
        return out

    @staticmethod
    def _dedupe(results: List[DecodedCode]) -> List[DecodedCode]:
        seen_payloads = set()
        out: List[DecodedCode] = []
        for r in results:
            if r.decoded:
                key = (r.kind, r.fmt, r.payload)
                if key in seen_payloads:
                    continue
                seen_payloads.add(key)
                out.append(r)
                continue
            # Keep only a few undecoded detections and dedupe by rough bbox.
            if r.bbox:
                key = (r.kind, tuple(round(float(x) / 12) for x in r.bbox))
                if key in seen_payloads:
                    continue
                seen_payloads.add(key)
                out.append(r)
        return out[:12]


def _box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = int(a.x1), int(a.y1), int(a.x2), int(a.y2)
    bx1, by1, bx2, by2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    den = area_a + area_b - inter
    return float(inter / den) if den > 0 else 0.0
