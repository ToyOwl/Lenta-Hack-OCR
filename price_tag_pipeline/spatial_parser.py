"""
OCR-first spatial parser for already-cropped Lenta price tags.

This module intentionally does not require YOLO. It uses full-tag OCR polygons as
localization primitives and then assigns them to semantic fields with template-aware
geometry rules.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .pipeline_types import Box, OCRItem

DIGIT_REPL = str.maketrans({
    "О": "0", "O": "0", "o": "0", "о": "0",
    "I": "1", "l": "1", "|": "1", "!": "1", "і": "1",
    "S": "5", "s": "5", "Б": "6", "б": "6",
    "З": "3", "з": "3", "В": "8", "в": "8",})

MAX_RUBLES = 99999

@dataclass
class OCRLine:
    text: str
    conf: float
    box: Box
    source_index: int

    @property
    def cx(self) -> float:
        return (self.box.x1 + self.box.x2) * 0.5

    @property
    def cy(self) -> float:
        return (self.box.y1 + self.box.y2) * 0.5

    @property
    def w(self) -> float:
        return float(self.box.width)

    @property
    def h(self) -> float:
        return float(self.box.height)

    @property
    def area(self) -> float:
        return float(self.box.area)


@dataclass
class PricePair:
    value: float
    rubles: int
    kopeeks: int
    rub_line: OCRLine
    kop_line: Optional[OCRLine]
    score: float
    kind: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "rubles": self.rubles,
            "kopeeks": self.kopeeks,
            "score": self.score,
            "kind": self.kind,
            "rub_text": self.rub_line.text,
            "rub_bbox": self.rub_line.box.to_xyxy(),
            "kop_text": self.kop_line.text if self.kop_line else "",
            "kop_bbox": self.kop_line.box.to_xyxy() if self.kop_line else None,
        }

    def merged_box(self, cls: str) -> Box:
        if self.kop_line is None:
            b = self.rub_line.box
            return Box(cls, b.x1, b.y1, b.x2, b.y2, self.score, "ocr_spatial")
        a = self.rub_line.box
        b = self.kop_line.box
        return Box(
            cls,
            min(a.x1, b.x1), min(a.y1, b.y1), max(a.x2, b.x2), max(a.y2, b.y2),
            self.score,
            "ocr_spatial",
        )

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def normalize_digits(s: str) -> str:
    return (s or "").translate(DIGIT_REPL)

def digits_only(s: str) -> str:
    return re.sub(r"\D+", "", normalize_digits(s))

def is_barcode_like(s: str) -> bool:
    d = digits_only(s)
    return len(d) >= 7

def is_percent_like(s: str) -> bool:
    return "%" in (s or "") or "проц" in (s or "").lower()

def item_to_line(item: OCRItem, idx: int, image_w: int, image_h: int) -> Optional[OCRLine]:

    if not item.text or item.conf < 0.05 or item.box is None:
        return None

    pts = np.asarray(item.box, dtype=np.float32).reshape(-1, 2)

    if pts.size == 0:
        return None

    x1, y1 = np.min(pts, axis=0)
    x2, y2 = np.max(pts, axis=0)
    b = Box("ocr", int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)), float(item.conf), "full_tag_ocr").clip(image_w, image_h)

    if b.width < 2 or b.height < 2:
        return None

    return OCRLine(text=normalize_text(item.text), conf=float(item.conf), box=b, source_index=idx)

def items_to_lines(items: Sequence[OCRItem], image_w: int, image_h: int) -> List[OCRLine]:
    lines: List[OCRLine] = []

    for i, item in enumerate(items):
        line = item_to_line(item, i, image_w, image_h)
        if line is not None:
            lines.append(line)

    return sorted(lines, key=lambda l: (l.box.y1, l.box.x1))

def y_overlap_ratio(a: Box, b: Box) -> float:
    inter = max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
    return inter / max(1, min(a.height, b.height))

def box_union(cls: str, boxes: Sequence[Box], conf: float = 1.0, source: str = "ocr_spatial") -> Optional[Box]:

    boxes = [b for b in boxes if b.width > 0 and b.height > 0]
    if not boxes:
        return None

    return Box(cls, min(b.x1 for b in boxes), min(b.y1 for b in boxes), max(b.x2 for b in boxes), max(b.y2 for b in boxes), conf,source,)

def detect_warm_red_panels(image: np.ndarray) -> Dict[str, Any]:
    """
    Find red/orange promo areas.
    Returns total warm bbox and component boxes.
    This is intentionally used only as a geometric prior.
    This stage only estimates where the red promo area starts.
    """
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    b, g, r = cv2.split(image)

    hue_red_or_orange = ((H <= 24) | (H >= 168)) & (S > 35) & (V > 45)
    rgb_warm = (r.astype(np.int16) > g.astype(np.int16) + 8) & (r.astype(np.int16) > b.astype(np.int16) + 12) & (r > 80)
    lower_prior = np.zeros((h, w), dtype=bool)
    lower_prior[int(h * 0.20):, :] = True
    mask = (hue_red_or_orange | rgb_warm) & lower_prior

    m = (mask.astype(np.uint8) * 255)
    kw = max(9, int(w * 0.025) | 1)
    kh = max(5, int(h * 0.015) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    comps: List[Box] = []
    min_area = max(50, int(w * h * 0.01))

    for c in contours:
      x, y, bw, bh = cv2.boundingRect(c)
      area = bw * bh

      if area < min_area:
            continue
      if bw < w * 0.10 or bh < h * 0.06:
            continue
      comps.append(Box("red_panel", x, y, x + bw, y + bh, area / float(w * h), "warm_mask"))

    comps = sorted(comps, key=lambda b: (b.y1, b.x1))
    total = box_union("red_area", comps, conf=sum(b.conf for b in comps), source="warm_mask")

    if total is None:
      total = Box("red_area", 0, int(h * 0.42), w, h, 0.1, "fallback")

    return {"red_area": total, "red_panels": comps, "red_top": int(total.y1), "mask_ratio": float(np.mean(mask)),}


def filter_product_lines(lines: Sequence[OCRLine], red_top: int, w: int, h: int) -> List[OCRLine]:
    bad_words = [
        "цена", "карта", "карт", "лента", "упаков", "колич", "руб", "py6", "pуб", "номер", "весах",
        "акция", "скид", "штрих", "barcode",]

    out: List[OCRLine] = []

    for l in lines:
        t = l.text.lower()
        d = digits_only(t)
        if l.cy > red_top - 5:
            continue
        if l.conf < 0.35:
            continue
        if is_barcode_like(t):
            continue
        if any(bw in t for bw in bad_words):
            continue
        # keep weights like 375г / 900г as part of product block, but not isolated 1/2-digit artifacts
        if d and len(d) <= 2 and len(t) <= 4:
            continue
        # skip QR corner / small numeric code regions
        if l.cx > w * 0.62 and l.cy < h * 0.32 and len(d) >= 3:
            continue
        out.append(l)
    return sorted(out, key=lambda l: (l.box.y1, l.box.x1))

def text_contains(line: OCRLine, *needles: str) -> bool:

    t = line.text.lower().replace("ё", "е")

    return any(n in t for n in needles)


def find_anchor(lines: Sequence[OCRLine], mode: str) -> Optional[OCRLine]:
    candidates: List[OCRLine] = []
    for l in lines:
        t = l.text.lower().replace("ё", "е")
        if mode == "card" and ("цен" in t and "карт" in t and "без" not in t):
            candidates.append(l)
        elif mode == "no_card" and ("без" in t and "карт" in t):
            candidates.append(l)
    if not candidates:
        return None
    # Prefer the most confident and widest anchor.
    return sorted(candidates, key=lambda l: (l.conf, l.w), reverse=True)[0]


def looks_mostly_numeric_price_text(s: str) -> bool:
    """Reject product names that accidentally produce digits after OCR char mapping."""
    raw = (s or "").strip()
    if not raw:
        return False
    d = digits_only(raw)
    if not d:
        return False
    # Remove tolerated currency fragments before checking alphabetic contamination.
    t = raw.lower().replace("руб", "").replace("py6", "").replace("pуб", "")
    t = t.replace("р", "").replace("p", "")
    letters = re.findall(r"[a-zа-яёA-ZА-ЯЁ]", t)
    # If a line is mostly a product/label word, do not treat it as price.
    if len(letters) >= 3:
        return False
    # Require at least one digit and mostly numeric punctuation/currency.
    numericish = re.sub(r"[0-9.,\-\s₽/]+", "", raw.translate(DIGIT_REPL))
    numericish = numericish.lower().replace("руб", "").replace("py6", "").replace("pуб", "")
    numericish = re.sub(r"[рpуyбb]+", "", numericish)
    return len(numericish.strip()) <= 2


def is_ruble_candidate(line: OCRLine, image_w: int, image_h: int) -> bool:
    d = digits_only(line.text)
    if not d or len(d) > 5:
        return False
    if is_percent_like(line.text):
        return False
    if is_barcode_like(line.text):
        return False
    if line.conf < 0.25:
        return False
    if not looks_mostly_numeric_price_text(line.text):
        return False
    # Single digit is almost never a ruble block here.
    if len(d) == 1:
        return False
    return True


def is_kopeeks_candidate(line: OCRLine) -> bool:
    d = digits_only(line.text)
    return len(d) == 2 and line.conf >= 0.25 and not is_percent_like(line.text) and looks_mostly_numeric_price_text(line.text)


def make_price_pair_from_compact(line: OCRLine, score: float, kind: str) -> Optional[PricePair]:
    d = digits_only(line.text)
    if len(d) < 3 or len(d) > 5:
        return None
    rub = int(d[:-2])
    kop = int(d[-2:])
    if rub <= 0 or not (0 <= kop <= 99):
        return None
    return PricePair(float(f"{rub}.{kop:02d}"), rub, kop, line, None, score, kind)


def pair_price(lines: Sequence[OCRLine], rub_line: OCRLine, kind: str = "generic") -> Optional[PricePair]:
    d = digits_only(rub_line.text)
    if not d:
        return None

    # If OCR already merged rubles+kopeeks: 18359 -> 183.59.
    if len(d) >= 4:
        compact = make_price_pair_from_compact(rub_line, score=rub_line.area, kind=kind + ":compact")
        if compact is not None:
            return compact

    rub = int(d)
    if rub <= 0:
        return None

    kop_candidates: List[Tuple[float, OCRLine]] = []
    for k in lines:

       if k is rub_line or not is_kopeeks_candidate(k):
          continue

       kd = digits_only(k.text)

       if not (0 <= int(kd) <= 99):
         continue

       # kopeeks are usually right of the ruble part, with partial y-overlap or slightly above/below

       rightish = k.cx > rub_line.cx
       near_x = k.box.x1 <= rub_line.box.x2 + max(120, int(rub_line.h * 2.4))
       y_close = abs(k.cy - rub_line.cy) <= max(rub_line.h * 0.75, 70)
       height_ok = 0.20 <= (k.h / max(1.0, rub_line.h)) <= 1.20

       if not (rightish and near_x and y_close and height_ok):
           continue

       score = 1000.0
       score -= abs(k.cy - rub_line.cy) * 3.0
       score -= max(0.0, k.box.x1 - rub_line.box.x2) * 1.0
       score += k.conf * 100.0
       score += y_overlap_ratio(rub_line.box, k.box) * 120.0
       kop_candidates.append((score, k))

    if kop_candidates:
        _, best_k = max(kop_candidates, key=lambda x: x[0])
        kop = int(digits_only(best_k.text))
        score = rub_line.area + 0.55 * best_k.area + rub_line.conf * 100 + best_k.conf * 50
        return PricePair(float(f"{rub}.{kop:02d}"), rub, kop, rub_line, best_k, score, kind)

    # If no kopeeks, still return integer price only when block is very large.
    if rub_line.h > 35:
        return PricePair(float(rub), rub, 0, rub_line, None, rub_line.area * 0.45, kind + ":integer")
    return None


def build_price_pairs(lines: Sequence[OCRLine], image_w: int, image_h: int) -> List[PricePair]:
    pairs: List[PricePair] = []
    for l in lines:
        if not is_ruble_candidate(l, image_w, image_h):
            continue
        p = pair_price(lines, l, kind="candidate")
        if p is not None:
            pairs.append(p)

    # dedupe by rub/kop and nearby bbox

    deduped: List[PricePair] = []
    for p in sorted(pairs, key=lambda x: x.score, reverse=True):
        duplicate = False
        pb = p.merged_box("price")
        for q in deduped:
            qb = q.merged_box("price")
            if abs(p.value - q.value) < 1e-6 and box_iou(pb, qb) > 0.25:
                duplicate = True
                break
        if not duplicate:
            deduped.append(p)

    return deduped


def box_iou(a: Box, b: Box) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0

def choose_price_near_anchor(
    pairs: Sequence[PricePair],
    anchor: Optional[OCRLine],
    red_area: Box,
    side: str,
    image_w: int,
    image_h: int,
) -> Optional[PricePair]:
    if not pairs:
        return None
    scored: List[Tuple[float, PricePair]] = []
    for p in pairs:
        b = p.merged_box("tmp")
        if b.y2 < red_area.y1 - image_h * 0.04:
            continue
        if b.area < image_w * image_h * 0.001:
            continue

        # avoid barcode-like compact prices with absurd values for retail price tags
        if p.rubles > 9999:
            continue
        score = p.score
        score += (b.cy if hasattr(b, 'cy') else (b.y1 + b.y2) * 0.5) * 0.10
        bcx = (b.x1 + b.x2) * 0.5
        # Hard side gates: otherwise the left/no-card anchor often steals the large right price.
        if side == "right" and bcx < image_w * 0.34:
            continue
        if side == "left" and bcx > image_w * 0.58:
            continue
        # A main/card price in red Lenta labels is often the largest and lower/right one.
        if side == "right" and bcx > image_w * 0.40:
            score += 400
        if side == "left" and bcx < image_w * 0.50:
            score += 400
        if anchor is not None:
            # Keep prices below or slightly overlapping the anchor.
            if b.y1 >= anchor.box.y1 - image_h * 0.04:
                score += 250
            if b.y1 >= anchor.box.y2 - image_h * 0.02:
                score += 350
            # x proximity to anchor, but allow large price to extend right.
            axc = anchor.cx
            bxc = (b.x1 + b.x2) * 0.5
            score -= abs(bxc - axc) * 0.20
        scored.append((score, p))
    if not scored:
        return None
    return max(scored, key=lambda x: x[0])[1]


def choose_main_red_promo_price(pairs: Sequence[PricePair], red_area: Box, image_w: int, image_h: int) -> Optional[PricePair]:
    scored: List[Tuple[float, PricePair]] = []
    for p in pairs:
        b = p.merged_box("tmp")
        if b.y2 < red_area.y1 - image_h * 0.03:
            continue
        if p.rubles <= 0 or p.rubles > 9999:
            continue
        # skip very small old prices unless there is no alternative
        h_score = b.height * 20.0
        area_score = math.sqrt(max(1, b.area)) * 18.0
        lower_score = ((b.y1 + b.y2) * 0.5 / max(1, image_h)) * 250.0
        right_score = ((b.x1 + b.x2) * 0.5 / max(1, image_w)) * 90.0
        compact_penalty = -120.0 if p.kop_line is None and ":compact" in p.kind else 0.0
        score = p.score + h_score + area_score + lower_score + right_score + compact_penalty
        scored.append((score, p))
    if not scored:
        return None
    return max(scored, key=lambda x: x[0])[1]


def price_to_str(p: Optional[PricePair]) -> str:
    if p is None:
        return ""
    return f"{p.rubles}.{p.kopeeks:02d}"


def line_texts(lines: Sequence[OCRLine]) -> str:
    return " | ".join([l.text for l in sorted(lines, key=lambda x: (x.box.y1, x.box.x1)) if l.text])


def parse_shelf_red_promo_spatial(image: np.ndarray, lines: List[OCRLine]) -> Dict[str, Any]:
    h, w = image.shape[:2]
    red = detect_warm_red_panels(image)
    red_area: Box = red["red_area"]
    red_top = red["red_top"]

    product_lines = filter_product_lines(lines, red_top=red_top, w=w, h=h)
    product_box = box_union("product_name", [l.box for l in product_lines], conf=0.92, source="ocr_spatial")

    pairs = build_price_pairs(lines, w, h)
    anchor_card = find_anchor(lines, "card")
    anchor_no_card = find_anchor(lines, "no_card")

    card_price = choose_price_near_anchor(pairs, anchor_card, red_area, side="right", image_w=w, image_h=h) if anchor_card else None
    no_card_price = choose_price_near_anchor(pairs, anchor_no_card, red_area, side="left", image_w=w, image_h=h) if anchor_no_card else None
    main_price = card_price or choose_main_red_promo_price(pairs, red_area, w, h)

    discount_lines = [l for l in lines if is_percent_like(l.text)]
    discount_box = box_union("discount_percent", [l.box for l in discount_lines], conf=0.75, source="ocr_spatial")
    discount_text = line_texts(discount_lines)

    barcode_lines = [l for l in lines if is_barcode_like(l.text)]
    barcode_text = ""
    if barcode_lines:
        barcode_text = max(barcode_lines, key=lambda l: len(digits_only(l.text))).text

    semantic_boxes: List[Box] = []
    if product_box:
        semantic_boxes.append(product_box)
    if main_price:
        semantic_boxes.append(main_price.merged_box("main_price"))
        semantic_boxes.append(Box("main_price_rubles", *main_price.rub_line.box.to_xyxy(), main_price.score, "ocr_spatial"))
        if main_price.kop_line:
            semantic_boxes.append(Box("main_price_kopeeks", *main_price.kop_line.box.to_xyxy(), main_price.score, "ocr_spatial"))
    if card_price and (main_price is None or card_price is not main_price):
        semantic_boxes.append(card_price.merged_box("card_price"))
    if no_card_price:
        semantic_boxes.append(no_card_price.merged_box("no_card_price"))
    if anchor_card:
        semantic_boxes.append(Box("card_price_label", *anchor_card.box.to_xyxy(), anchor_card.conf, "ocr_spatial"))
    if anchor_no_card:
        semantic_boxes.append(Box("no_card_price_label", *anchor_no_card.box.to_xyxy(), anchor_no_card.conf, "ocr_spatial"))
    if discount_box:
        semantic_boxes.append(discount_box)

    fields = {
        "product_name": line_texts(product_lines),
        "main_price": price_to_str(main_price),
        "card_price": price_to_str(card_price),
        "no_card_price": price_to_str(no_card_price),
        "discount_percent_raw": discount_text,
        "barcode_text_raw": barcode_text,
    }
    if fields["main_price"] and not fields["card_price"]:
        fields["card_price"] = fields["main_price"]

    return {
        "fields": fields,
        "semantic_boxes": semantic_boxes,
        "price_candidates": [p.to_dict() for p in pairs],
        "geometry": {
            "red_area": red_area.to_xyxy(),
            "red_top": red_top,
            "red_panels": [b.to_xyxy() for b in red.get("red_panels", [])],
            "warm_mask_ratio": red.get("mask_ratio", 0.0),
        },
    }


def parse_hanging_yellow_spatial(image: np.ndarray, lines: List[OCRLine]) -> Dict[str, Any]:
    h, w = image.shape[:2]
    pairs = build_price_pairs(lines, w, h)
    # Main price: largest/lower/right price.
    main = None
    if pairs:
        main = max(pairs, key=lambda p: p.score + p.merged_box("tmp").height * 18 + p.merged_box("tmp").x1 * 0.15)

    product_lines: List[OCRLine] = []
    for l in lines:
        t = l.text.lower()
        if l.cy > h * 0.50:
            continue
        if any(x in t for x in ["акция", "цена", "руб", "номер", "весах"]):
            continue
        if is_barcode_like(t) or is_percent_like(t):
            continue
        if digits_only(t) and len(digits_only(t)) <= 2:
            continue
        product_lines.append(l)

    scale_lines = [l for l in lines if text_contains(l, "номер") or (digits_only(l.text) and l.cx > w * 0.72 and l.cy < h * 0.55)]
    old_anchor = find_anchor(lines, "card")
    old = choose_price_near_anchor(pairs, old_anchor, Box("region", 0, int(h * 0.25), int(w * 0.55), h, 1.0, "fallback"), side="left", image_w=w, image_h=h)

    product_box = box_union("product_name", [l.box for l in product_lines], conf=0.90, source="ocr_spatial")
    scale_box = box_union("scale_number", [l.box for l in scale_lines], conf=0.70, source="ocr_spatial")

    semantic_boxes: List[Box] = []
    if product_box:
        semantic_boxes.append(product_box)
    if scale_box:
        semantic_boxes.append(scale_box)
    if old and (main is None or old is not main):
        semantic_boxes.append(old.merged_box("old_price"))
    if main:
        semantic_boxes.append(main.merged_box("main_price"))
        semantic_boxes.append(Box("main_price_rubles", *main.rub_line.box.to_xyxy(), main.score, "ocr_spatial"))
        if main.kop_line:
            semantic_boxes.append(Box("main_price_kopeeks", *main.kop_line.box.to_xyxy(), main.score, "ocr_spatial"))

    scale_number = ""
    for l in scale_lines:
        d = digits_only(l.text)
        if 2 <= len(d) <= 4:
            scale_number = d
            break

    return {
        "fields": {
            "product_name": line_texts(product_lines),
            "main_price": price_to_str(main),
            "old_price": price_to_str(old),
            "scale_number": scale_number,
        },
        "semantic_boxes": semantic_boxes,
        "price_candidates": [p.to_dict() for p in pairs],
        "geometry": {},
    }



def cluster_lines_by_x(lines: Sequence[OCRLine], gap: float) -> List[List[OCRLine]]:
    """Simple 1-D horizontal clustering by x-center."""
    if not lines:
        return []
    ordered = sorted(lines, key=lambda l: l.cx)
    clusters: List[List[OCRLine]] = [[ordered[0]]]
    for line in ordered[1:]:
        prev_center = float(np.mean([x.cx for x in clusters[-1]]))
        if line.cx - prev_center > gap:
            clusters.append([line])
        else:
            clusters[-1].append(line)
    return clusters


def cluster_lines_by_y(lines: Sequence[OCRLine], gap: float) -> List[List[OCRLine]]:
    """Simple 1-D vertical clustering by y-center."""
    if not lines:
        return []
    ordered = sorted(lines, key=lambda l: l.cy)
    clusters: List[List[OCRLine]] = [[ordered[0]]]
    for line in ordered[1:]:
        prev_center = float(np.mean([x.cy for x in clusters[-1]]))
        if line.cy - prev_center > gap:
            clusters.append([line])
        else:
            clusters[-1].append(line)
    return clusters


def parse_shelf_white_regular_spatial(image: np.ndarray, lines: List[OCRLine]) -> Dict[str, Any]:
    """OCR-first parser for white/orange Lenta shelf tags with several columns.

    This path is important for shelf-rail cells: ordinary white labels and
    progressive labels often have several price candidates in one crop.  The
    parser keeps all column prices instead of forcing one premature answer.
    """
    h, w = image.shape[:2]
    price_lines = [l for l in lines if is_ruble_candidate(l, w, h)]
    if not price_lines:
        return parse_generic_spatial(image, lines)

    # Horizontal clustering: price columns left-to-right.  A progressive tag has
    # several price blocks but they should remain grouped by x-position.
    columns = cluster_lines_by_x(price_lines, gap=w * 0.22)

    semantic_boxes: List[Box] = []
    price_candidates: List[PricePair] = []
    column_debug: List[Dict[str, Any]] = []

    for col_idx, col in enumerate(columns):
        if not col:
            continue
        best_in_col = max(col, key=lambda l: (l.h * l.conf, l.area))
        p_pair = pair_price(lines, best_in_col, kind=f"white_col{col_idx}")
        if p_pair is None:
            continue
        price_candidates.append(p_pair)
        semantic_boxes.append(p_pair.merged_box(f"main_price_col{col_idx}"))
        semantic_boxes.append(Box(f"main_price_rubles_col{col_idx}", *p_pair.rub_line.box.to_xyxy(), p_pair.score, "ocr_spatial_white"))
        if p_pair.kop_line:
            semantic_boxes.append(Box(f"main_price_kopeeks_col{col_idx}", *p_pair.kop_line.box.to_xyxy(), p_pair.score, "ocr_spatial_white"))
        cb = box_union(f"price_column_{col_idx}", [l.box for l in col], conf=0.55, source="ocr_spatial_white")
        column_debug.append({
            "column_index": col_idx,
            "line_count": len(col),
            "bbox": cb.to_xyxy() if cb else None,
            "texts": [l.text for l in col],
        })

    red = detect_warm_red_panels(image)
    red_top = red["red_top"]
    product_lines = filter_product_lines(lines, red_top=red_top, w=w, h=h)
    product_box = box_union("product_name", [l.box for l in product_lines], conf=0.88, source="ocr_spatial_white")
    if product_box:
        semantic_boxes.append(product_box)

    unit_lines = [
        l for l in lines
        if l.cy > h * 0.70 and not is_ruble_candidate(l, w, h) and not is_barcode_like(l.text)
    ]
    unit_box = box_union("unit_price_or_article", [l.box for l in unit_lines], conf=0.65, source="ocr_spatial_white")
    if unit_box:
        semantic_boxes.append(unit_box)

    # Vertical clustering: text rows.  Useful for distinguishing progressive
    # multi-row conditions from ordinary single-price labels in the JSON/debug.
    text_rows = cluster_lines_by_y(lines, gap=max(16.0, h * 0.085))
    row_debug: List[Dict[str, Any]] = []
    for row_idx, row in enumerate(text_rows):
        rb = box_union(f"text_row_{row_idx}", [l.box for l in row], conf=0.40, source="ocr_spatial_white")
        row_debug.append({
            "row_index": row_idx,
            "line_count": len(row),
            "bbox": rb.to_xyxy() if rb else None,
            "texts": [l.text for l in row],
        })

    fields = {
        "product_name": line_texts(product_lines),
        "main_price": price_to_str(price_candidates[0] if price_candidates else None),
        "prices_by_column": [price_to_str(p) for p in price_candidates],
        "unit_info": line_texts(unit_lines),
    }

    return {
        "fields": fields,
        "semantic_boxes": semantic_boxes,
        "price_candidates": [p.to_dict() for p in price_candidates],
        "geometry": {
            "red_area": red["red_area"].to_xyxy(),
            "red_top": red_top,
            "column_count": len(columns),
            "columns": column_debug,
            "row_count": len(text_rows),
            "rows": row_debug,
        },
        "ocr_lines": [{"text": l.text, "conf": l.conf, "bbox": l.box.to_xyxy()} for l in lines],
    }


def parse_generic_spatial(image: np.ndarray, lines: List[OCRLine]) -> Dict[str, Any]:
    """Generic OCR-first fallback for unknown or weak templates."""
    h, w = image.shape[:2]
    red = detect_warm_red_panels(image)
    product_lines = filter_product_lines(lines, red_top=red["red_top"], w=w, h=h)
    pairs = build_price_pairs(lines, w, h)
    main = choose_main_red_promo_price(pairs, red["red_area"], w, h) if pairs else None

    semantic_boxes: List[Box] = []
    pbox = box_union("product_name", [l.box for l in product_lines], conf=0.80, source="ocr_spatial")
    if pbox:
        semantic_boxes.append(pbox)
    if main:
        semantic_boxes.append(main.merged_box("main_price"))

    return {
        "fields": {"product_name": line_texts(product_lines), "main_price": price_to_str(main)},
        "semantic_boxes": semantic_boxes,
        "price_candidates": [p.to_dict() for p in pairs],
        "geometry": {"red_area": red["red_area"].to_xyxy(), "red_top": red["red_top"]},
    }

def parse_full_tag_spatial(image: np.ndarray, template_name: str, full_items: Sequence[OCRItem]) -> Dict[str, Any]:
    h, w = image.shape[:2]
    lines = items_to_lines(full_items, w, h)
    if template_name in {"shelf_red_promo", "progressive", "progressive_yellow"}:
        parsed = parse_shelf_red_promo_spatial(image, lines)
    elif template_name == "hanging_yellow_promo_large":
        parsed = parse_hanging_yellow_spatial(image, lines)
    elif template_name == "shelf_white_regular":
        parsed = parse_shelf_white_regular_spatial(image, lines)
    else:
        parsed = parse_generic_spatial(image, lines)

    parsed["ocr_lines"] = [
        {"text": l.text, "conf": l.conf, "bbox": l.box.to_xyxy()} for l in lines
    ]
    return parsed
