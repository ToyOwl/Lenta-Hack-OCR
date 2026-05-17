"""
Template-aware heuristic layout extraction for already-cropped price tags.
No YOLO/Ultralytics dependency is used here. The input is assumed to be one
already-cropped price tag.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .code_decoder import CodeDecoder
from .image_ops import box_iou, merge_same_class_boxes
from .pipeline_types import Box


class HeuristicLayoutExtractor:

    def __init__(self, code_decoder: Optional[CodeDecoder] = None, debug: bool = False) -> None:
        self.debug = debug
        self.code_decoder = code_decoder or CodeDecoder(use_pyzbar=False, try_opencv_qr=True)

    def extract(self, image: np.ndarray, template: Dict[str, Any]) -> List[Box]:
        h, w = image.shape[:2]
        template_name = str(template.get("template_name", "small_blurry_or_unknown"))

        boxes = self._template_default_boxes(image, template_name)

        qr_boxes = self._detect_qr_boxes(image)
        if qr_boxes:
            boxes = self._replace_low_conf_same_class(boxes, qr_boxes, cls="qr_code")

        if template_name not in {"shelf_red_promo", "hanging_yellow_promo_large", "progressive", "progressive_yellow"}:
            barcode_box = self._detect_barcode_candidate(image)
            if barcode_box is not None:
                boxes = self._replace_low_conf_same_class(boxes, [barcode_box], cls="linear_barcode")

        if template_name not in {"shelf_red_promo", "hanging_yellow_promo_large", "shelf_white_regular", "progressive", "progressive_yellow"}:
            price_box = self._detect_main_price_box(image, qr_boxes=qr_boxes)
            if price_box is not None:
                boxes.append(price_box)
                kp = self._estimate_kopeeks_box(image, price_box)
                if kp is not None:
                    boxes.append(kp)

        clipped: List[Box] = []
        for b in boxes:
            bc = b.clip(w, h)
            if bc.width >= 5 and bc.height >= 5:
                clipped.append(bc)
        return merge_same_class_boxes(clipped, iou_thr=0.55)

    def detect(self, image: np.ndarray, template: Dict[str, Any]) -> List[Box]:
        return self.extract(image, template)

    def _template_default_boxes(self, image: np.ndarray, template_name: str) -> List[Box]:
        h, w = image.shape[:2]
        if template_name == "shelf_red_promo":
            return self._boxes_shelf_red_promo(image)
        if template_name == "hanging_yellow_promo_large":
            return self._boxes_hanging_yellow_promo(w, h)
        if template_name == "shelf_white_regular":
            return self._boxes_shelf_white_regular(w, h)
        if template_name in {"progressive", "progressive_yellow"}:
            return self._boxes_progressive_price_tag(image, template_name)
        return [self._pbox(w, h, "full_tag_text", 0.03, 0.03, 0.97, 0.97, 0.20, "heuristic_template")]

    def _boxes_hanging_yellow_promo(self, w: int, h: int) -> List[Box]:

        """
        Geometry template for large yellow hanging promo sign.
        Target example: yellow header "АКЦИЯ"; product name on left;
        scale number on right; crossed/card old price on left-middle;
        main price on right-lower.
        """
        return [
            self._pbox(w, h, "promo_header", 0.03, 0.02, 0.97, 0.24, 0.95, "template_hanging_yellow_promo"),
            self._pbox(w, h, "product_name", 0.04, 0.24, 0.64, 0.42, 0.90, "template_hanging_yellow_promo"),
            self._pbox(w, h, "scale_number", 0.78, 0.23, 0.98, 0.50, 0.85, "template_hanging_yellow_promo"),
            self._pbox(w, h, "old_price", 0.05, 0.43, 0.43, 0.76, 0.90, "template_hanging_yellow_promo"),
            self._pbox(w, h, "main_price", 0.56, 0.40, 0.98, 0.92, 0.95, "template_hanging_yellow_promo"),
            self._pbox(w, h, "main_price_rubles", 0.60, 0.48, 0.86, 0.90, 0.95, "template_hanging_yellow_promo"),
            self._pbox(w, h, "main_price_kopeeks", 0.83, 0.43, 0.97, 0.74, 0.90, "template_hanging_yellow_promo"),
            self._pbox(w, h, "footer_note", 0.02, 0.84, 0.45, 0.98, 0.75, "template_hanging_yellow_promo"),
        ]

    def _boxes_shelf_red_promo(self, image: np.ndarray) -> List[Box]:
        h, w = image.shape[:2]
        split_y, split_conf = self._estimate_red_promo_split_y(image)
        sy = split_y / max(1, h)
        sy = float(np.clip(sy, 0.42, 0.62))
        _ = split_conf
        return [
            self._pbox(w, h, "product_name", 0.055, 0.055, 0.565, max(0.28, sy - 0.055), 0.72, "template_lenta_red_promo"),
            self._pbox(w, h, "qr_code", 0.595, 0.045, 0.885, max(0.31, sy - 0.09), 0.55, "template_lenta_red_promo"),
            self._pbox(w, h, "card_price_small", 0.345, max(0.24, sy - 0.165), 0.545, min(0.52, sy + 0.015), 0.45, "template_lenta_red_promo"),
            self._pbox(w, h, "no_card_price_small", 0.585, max(0.24, sy - 0.165), 0.875, min(0.52, sy + 0.015), 0.45, "template_lenta_red_promo"),
            self._pbox(w, h, "promo_label", 0.040, sy + 0.030, 0.325, min(0.80, sy + 0.34), 0.62, "template_lenta_red_promo"),
            self._pbox(w, h, "main_price", 0.500, sy + 0.045, 0.910, min(0.790, sy + 0.34), 0.75, "template_lenta_red_promo"),
            self._pbox(w, h, "main_price_rubles", 0.545, sy + 0.075, 0.755, min(0.775, sy + 0.325), 0.68, "template_lenta_red_promo"),
            self._pbox(w, h, "main_price_kopeeks", 0.735, sy + 0.055, 0.910, min(0.700, sy + 0.235), 0.58, "template_lenta_red_promo"),
            self._pbox(w, h, "linear_barcode", 0.420, max(0.780, sy + 0.370), 0.845, 0.915, 0.58, "template_lenta_red_promo"),
            self._pbox(w, h, "barcode_digits", 0.350, 0.855, 0.950, 0.985, 0.32, "template_lenta_red_promo"),
            self._pbox(w, h, "promo_article", 0.045, 0.740, 0.335, 0.905, 0.32, "template_lenta_red_promo"),
        ]

    def _boxes_progressive_price_tag(self, image: np.ndarray, template_name: str) -> List[Box]:
        """Geometry template for the newer/progressive Lenta shelf tag.

        Most crops have a light top half with product/QR and a warm/yellow lower
        price block.  The exact split is estimated from the warm color mask, as
        in the shelf red-promo template, but the price zone is wider and more
        centered because the large price is often the only readable field.
        """
        h, w = image.shape[:2]
        split_y, _ = self._estimate_red_promo_split_y(image)
        sy = float(np.clip(split_y / max(1, h), 0.38, 0.64))
        src = f"template_{template_name}"
        return [
            self._pbox(w, h, "product_name", 0.045, 0.045, 0.620, max(0.30, sy - 0.045), 0.66, src),
            self._pbox(w, h, "qr_code", 0.585, 0.040, 0.900, max(0.30, sy - 0.065), 0.52, src),
            self._pbox(w, h, "card_price_small", 0.060, max(0.28, sy - 0.12), 0.380, min(0.58, sy + 0.06), 0.36, src),
            self._pbox(w, h, "no_card_price_small", 0.380, max(0.28, sy - 0.12), 0.690, min(0.58, sy + 0.06), 0.34, src),
            self._pbox(w, h, "main_price", 0.160, sy + 0.030, 0.900, min(0.86, sy + 0.38), 0.84, src),
            self._pbox(w, h, "main_price_rubles", 0.230, sy + 0.065, 0.720, min(0.84, sy + 0.36), 0.72, src),
            self._pbox(w, h, "main_price_kopeeks", 0.690, sy + 0.040, 0.910, min(0.74, sy + 0.25), 0.52, src),
            self._pbox(w, h, "unit_price_or_article", 0.045, min(0.78, sy + 0.34), 0.485, 0.965, 0.30, src),
            self._pbox(w, h, "linear_barcode", 0.430, min(0.78, sy + 0.34), 0.900, 0.945, 0.30, src),
        ]

    def _boxes_shelf_white_regular(self, w: int, h: int) -> List[Box]:
        return [
            self._pbox(w, h, "product_name", 0.04, 0.04, 0.72, 0.42, 0.45, "heuristic_template"),
            self._pbox(w, h, "qr_code", 0.60, 0.04, 0.90, 0.42, 0.35, "heuristic_template"),
            self._pbox(w, h, "main_price", 0.25, 0.38, 0.82, 0.78, 0.30, "heuristic_template"),
            self._pbox(w, h, "unit_price_or_article", 0.03, 0.65, 0.50, 0.94, 0.25, "heuristic_template"),
            self._pbox(w, h, "linear_barcode", 0.38, 0.72, 0.82, 0.92, 0.25, "heuristic_template"),
        ]

    @staticmethod
    def _pbox(w: int, h: int, cls: str, x1: float, y1: float, x2: float, y2: float, conf: float, source: str) -> Box:
        return Box(cls, int(round(w * x1)), int(round(h * y1)), int(round(w * x2)), int(round(h * y2)), conf, source).clip(w, h)

    @staticmethod
    def _replace_low_conf_same_class(base: List[Box], new_boxes: List[Box], cls: str) -> List[Box]:
        out = list(base)
        for nb in new_boxes:
            replaced = False
            for i, ob in enumerate(out):
                if ob.cls != cls:
                    continue
                if nb.conf >= ob.conf and nb.area <= max(1, ob.area) * 3.0:
                    out[i] = nb
                    replaced = True
                    break
                if box_iou(nb, ob) > 0.25:
                    replaced = True
                    break
            if not replaced:
                out.append(nb)
        return out

    def _estimate_red_promo_split_y(self, image: np.ndarray) -> Tuple[int, float]:
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        warm = (((H <= 24) | (H >= 172)) & (S >= 28) & (V >= 35)) | ((H >= 5) & (H <= 35) & (S >= 22) & (V >= 35))
        x1, x2 = int(w * 0.08), int(w * 0.92)
        row_score = warm[:, x1:x2].mean(axis=1) if x2 > x1 else warm.mean(axis=1)
        row_score = cv2.GaussianBlur(row_score.astype(np.float32).reshape(-1, 1), (1, 21), 0).reshape(-1)
        lo, hi = int(h * 0.34), int(h * 0.70)
        if hi <= lo:
            return int(h * 0.50), 0.0
        thr = max(0.08, min(0.35, float(np.percentile(row_score[lo:hi], 80)) * 0.55))
        min_run = max(5, int(h * 0.025))
        for y in range(lo, hi - min_run):
            if float(np.mean(row_score[y:y + min_run])) >= thr:
                return y, float(np.mean(row_score[y:y + min_run]))
        deriv = np.diff(row_score, prepend=row_score[0])
        y = int(lo + np.argmax(deriv[lo:hi]))
        return y, float(row_score[y])

    def _detect_qr_boxes(self, image: np.ndarray) -> List[Box]:
        h, w = image.shape[:2]
        boxes = self.code_decoder.detect_qr_boxes(image)
        good = []
        for b in boxes:
            ratio = b.width / max(1, b.height)
            area_ratio = b.area / max(1, w * h)
            if 0.55 <= ratio <= 1.65 and 0.005 <= area_ratio <= 0.16:
                good.append(b)
        return good

    def _detect_main_price_box(self, image: np.ndarray, qr_boxes: List[Box]) -> Optional[Box]:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12)
        for qb in qr_boxes:
            th[qb.y1:qb.y2, qb.x1:qb.x2] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        th2 = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(th2, connectivity=8)
        comps = []
        for i in range(1, num_labels):
            x, y, bw, bh, area = stats[i]
            if area < 20 or bh < max(14, h * 0.10) or bw < 3:
                continue
            aspect = bw / max(1, bh)
            if aspect > 1.6:
                continue
            if y < h * 0.16 and bh < h * 0.28:
                continue
            comps.append((x, y, bw, bh, area))
        if not comps:
            return None
        comps_sorted = sorted(comps, key=lambda t: (t[4] * (1.0 + t[3] / max(1, h))), reverse=True)
        seed = comps_sorted[0]
        _, sy, _, sh, _ = seed
        cy_seed = sy + sh / 2
        group = []
        for x, y, bw, bh, area in comps:
            cy = y + bh / 2
            if abs(cy - cy_seed) <= max(sh, bh) * 0.65 and bh >= sh * 0.42:
                group.append((x, y, bw, bh, area))
        if not group:
            group = [seed]
        x1 = min(x for x, y, bw, bh, area in group)
        y1 = min(y for x, y, bw, bh, area in group)
        x2 = max(x + bw for x, y, bw, bh, area in group)
        y2 = max(y + bh for x, y, bw, bh, area in group)
        pad_x, pad_y = int(max(4, (x2 - x1) * 0.06)), int(max(3, (y2 - y1) * 0.08))
        b = Box("main_price_rubles", x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, 0.45, "heuristic_large_digits").clip(w, h)
        if b.width > w * 0.82 or b.height > h * 0.62 or b.area > w * h * 0.42:
            return None
        return b

    def _estimate_kopeeks_box(self, image: np.ndarray, price_box: Box) -> Optional[Box]:
        h, w = image.shape[:2]
        x1 = min(w - 1, price_box.x2 - int(price_box.width * 0.08))
        x2 = min(w, price_box.x2 + int(price_box.width * 0.45))
        y1 = max(0, price_box.y1 - int(price_box.height * 0.20))
        y2 = min(h, price_box.y1 + int(price_box.height * 0.55))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        return Box("main_price_kopeeks", x1, y1, x2, y2, 0.25, "heuristic_price_geometry")

    def _detect_barcode_candidate(self, image: np.ndarray) -> Optional[Box]:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        y0 = int(h * 0.45)
        roi = gray[y0:h, :]
        if roi.size == 0:
            return None
        grad_x = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.convertScaleAbs(np.abs(grad_x) - 0.5 * np.abs(grad_y))
        grad = cv2.blur(grad, (9, 3))
        _, bwimg = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        bwimg = cv2.morphologyEx(bwimg, cv2.MORPH_CLOSE, kernel, iterations=2)
        bwimg = cv2.erode(bwimg, None, iterations=1)
        bwimg = cv2.dilate(bwimg, None, iterations=1)
        contours, _ = cv2.findContours(bwimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            if area < max(100, w * h * 0.002):
                continue
            aspect = bw / max(1, bh)
            area_ratio = area / max(1, w * h)
            if aspect < 2.0 or bw < w * 0.15:
                continue
            if area_ratio > 0.18 or bh > h * 0.28:
                continue
            candidates.append((area, x, y, bw, bh))
        if not candidates:
            return None
        _, x, y, bw, bh = max(candidates, key=lambda t: t[0])
        return Box("linear_barcode", x, y0 + y, x + bw, y0 + y + bh, 0.38, "heuristic_barcode").clip(w, h)
