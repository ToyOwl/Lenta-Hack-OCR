"""
Debug visualization with Unicode/Cyrillic OCR overlay.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .pipeline_types import Box, DecodedCode, OCRItem

COLOR_BY_CLASS = {
    "promo_header": (0, 220, 255),
    "promo_label": (128, 128, 255),
    "product_name": (0, 200, 255),
    "scale_number": (180, 180, 255),
    "old_price": (0, 100, 255),
    "old_price_or_without_card": (255, 180, 0),
    "card_price_small": (0, 180, 255),
    "no_card_price_small": (255, 180, 0),
    "card_price": (0, 210, 255),
    "no_card_price": (255, 210, 0),
    "card_price_label": (0, 170, 220),
    "no_card_price_label": (220, 170, 0),
    "discount_percent": (0, 128, 255),
    "main_price": (0, 255, 0),
    "main_price_rubles": (50, 255, 50),
    "main_price_kopeeks": (120, 255, 120),
    "footer_note": (200, 200, 200),
    "qr_code": (255, 0, 255),
    "linear_barcode": (255, 0, 0),
    "barcode": (255, 0, 0),
    "datamatrix": (255, 80, 80),
    "unit_price_or_article": (180, 255, 180),
    "full_tag_text": (200, 200, 200),
    "price_rail": (255, 180, 0),
    "price_cell": (0, 200, 0),
    "price_cell_ordinary": (0, 255, 0),
    "price_cell_progressive": (0, 128, 255),
    "price_cell_progressive_yellow": (0, 220, 255),
}

def _bgr_to_rgb(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    b, g, r = color
    return int(r), int(g), int(b)

def _font_candidates() -> List[Path]:
    """Return common font paths with Cyrillic coverage.
    Priority:
      1. PRICE_TAG_FONT_PATH env var;
      2. Windows fonts;
      3. Linux DejaVu/Noto fonts;
      4. macOS Arial/Helvetica-like fonts.
    """
    paths: List[Path] = []
    env_path = os.environ.get("PRICE_TAG_FONT_PATH", "").strip().strip('"')
    if env_path:
        paths.append(Path(env_path))

    paths.extend(
        [
            Path(r"C:\Windows\Fonts\arial.ttf"),
            Path(r"C:\Windows\Fonts\arialbd.ttf"),
            Path(r"C:\Windows\Fonts\segoeui.ttf"),
            Path(r"C:\Windows\Fonts\tahoma.ttf"),
            Path(r"C:\Windows\Fonts\calibri.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
            Path("/Library/Fonts/Arial Unicode.ttf"),
            Path("/Library/Fonts/Arial.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        ]
    )
    return paths

@lru_cache(maxsize=16)
def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = _font_candidates()

    if bold:
        bold_candidates: List[Path] = []
        for p in candidates:
            name = p.name.lower()
            if "bold" in name or "bd" in name or name == "arialbd.ttf":
                bold_candidates.append(p)
        candidates = bold_candidates + candidates

    for p in candidates:
        try:
            if p.exists():
                return ImageFont.truetype(str(p), size=size)
        except Exception:
            continue

    return ImageFont.load_default()

def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int, int, int]:
    try:
        return draw.textbbox((0, 0), text, font=font)
    except Exception:
        w = int(draw.textlength(text, font=font)) if hasattr(draw, "textlength") else max(8, len(text) * 8)
        return 0, 0, w, 14

def _draw_text_pil(img_bgr: np.ndarray,
                   text: str,
                   xy: Tuple[int, int],
                   font_size: int = 16,
                   color_bgr: Tuple[int, int, int] = (0, 0, 0),
                   bold: bool = False,) -> np.ndarray:
    if not text:
      return img_bgr

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    font = _load_font(font_size, bold=bold)
    draw.text(xy, text, fill=_bgr_to_rgb(color_bgr), font=font)

    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

def _draw_label_pil(img_bgr: np.ndarray,
                    text: str,
                    x: int,
                    y: int,
                    bg_bgr: Tuple[int, int, int],
                    font_size: int = 15,
                    max_width: int | None = None,
                   ) -> np.ndarray:

    """Draw colored label with Unicode text.

    Args:
        img_bgr: OpenCV image, modified via returned copy.
        text: Unicode label.
        x, y: Anchor. The label is placed above y when possible.
        bg_bgr: Background label color in BGR.
        font_size: TrueType size.
        max_width: optional hard clipping by characters is already handled outside;
            this parameter only protects against drawing beyond image width.
    """

    if not text:
        return img_bgr

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    font = _load_font(font_size, bold=False)

    bbox = _text_bbox(draw, text, font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad_x = 4
    pad_y = 3

    if max_width is None:
        max_width = w - 2

    x0 = max(0, min(int(x), max(0, w - min(tw + 2 * pad_x, max_width) - 1)))
    y0 = max(0, int(y) - th - 2 * pad_y)
    if y0 < 2 and y + th + 2 * pad_y < h:
        y0 = int(y) + 2

    x1 = min(w - 1, x0 + tw + 2 * pad_x)
    y1 = min(h - 1, y0 + th + 2 * pad_y)

    draw.rectangle([x0, y0, x1, y1], fill=_bgr_to_rgb(bg_bgr))
    draw.text((x0 + pad_x, y0 + pad_y), text, fill=(0, 0, 0), font=font)

    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

def truncate_text(s: str, max_len: int = 42) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= max_len else s[: max_len - 3] + "..."

def draw_label(img: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int]) -> None:
    rendered = _draw_label_pil(img, text, x, y, color, font_size=15)
    img[:, :, :] = rendered

def _zone_text(items: List[OCRItem]) -> str:
    texts = [it.text.strip() for it in items if (it.text or "").strip()]
    return " | ".join(texts)

def _ocr_item_box(item: OCRItem):
    if item.box is None:
        return None
    try:
        pts = np.asarray(item.box, dtype=np.float32).reshape(-1, 2)
        if pts.size == 0:
            return None
        x1, y1 = np.min(pts, axis=0)
        x2, y2 = np.max(pts, axis=0)
        return int(x1), int(y1), int(x2), int(y2)
    except Exception:
        return None

def _draw_full_ocr_items(img: np.ndarray, items: List[OCRItem]) -> np.ndarray:
    # Thin OCR-localization overlay from full-tag OCR.
    out = img

    for it in items:
        b = _ocr_item_box(it)

        if b is None:
            continue
        x1, y1, x2, y2 = b

        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(out, (x1, y1), (x2, y2), (160, 160, 160), 1)

        if it.conf >= 0.80:
            label = truncate_text(it.text, 24)
            out = _draw_text_pil(
                out,
                label,
                (x1, max(2, y1 - 16)),
                font_size=12,
                color_bgr=(80, 80, 80),
                bold=False,
            )
    return out

def _semantic_text_for_box(cls: str, parsed: Mapping[str, Any]) -> str:
    mapping = {"product_name": "product_name",
               "main_price": "main_price",
               "main_price_rubles": "main_price",
               "main_price_kopeeks": "main_price",
               "card_price": "card_price",
               "no_card_price": "no_card_price",
               "old_price": "old_price",
               "discount_percent": "discount_percent_raw",
               "scale_number": "scale_number",
               "card_price_label": "card_price",
               "no_card_price_label": "no_card_price",}

    key = mapping.get(cls, "")

    if not key:
      return ""

    return str(parsed.get(key, "") or "")

def draw_top_panel(image: np.ndarray, lines: List[str]) -> np.ndarray:
    h, w = image.shape[:2]
    canvas_h = max(42, 23 * len(lines) + 10)
    canvas = np.full((canvas_h, w, 3), 255, dtype=np.uint8)

    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    font = _load_font(16, bold=False)

    y = 6
    for line in lines:
        draw.text((8, y), truncate_text(line, 170), fill=(0, 0, 0), font=font)
        y += 23

    canvas = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return np.vstack([canvas, image])

def draw_debug(image: np.ndarray, boxes: List[Box], codes: List[DecodedCode], title_lines: List[str]) -> np.ndarray:

    return draw_debug_with_ocr(image=image, boxes=boxes,codes=codes, title_lines=title_lines, ocr_by_zone={}, parsed={},)

def _draw_semantic_panel(out: np.ndarray, parsed: Mapping[str, Any]) -> np.ndarray:
    """Draw final product card, it shows
    product type, producer/brand, country, scale number, package size, separated
    regular/promo/card/no-card prices, and alcohol-vs-discount percentages.
    """
    semantic: List[str] = []

    def add(label: str, value: Any, *, skip_if_empty: bool = True) -> None:

        if value is None and skip_if_empty:
            return

        val = str(value or "").strip()

        if not val and skip_if_empty:
            return
        semantic.append(f"{label}: {val}")

    product_name = parsed.get("product_name")
    add("Товар", product_name)
    add("Тип", parsed.get("product_type_label") or parsed.get("product_type"))
    add("Производитель", parsed.get("producer") or parsed.get("brand"))
    add("Страна", parsed.get("country"))
    add("Вес/объём", parsed.get("package_size") or parsed.get("volume"))
    add("Ед. продажи", parsed.get("sale_unit"))
    if parsed.get("scale_number"):
        add("Номер весов", parsed.get("scale_number"))

    regular_price = parsed.get("regular_price")
    promo_price = parsed.get("promo_price")
    main_price = parsed.get("main_price")
    card_price = parsed.get("card_price") or parsed.get("card_price_small")
    no_card_price = parsed.get("no_card_price") or parsed.get("no_card_price_small")
    old_price = parsed.get("old_price")

    if regular_price and promo_price and regular_price != promo_price:
        add("Обычная цена", regular_price)
        add("Акционная цена", promo_price)

    elif main_price:
        add("Цена", main_price)

    elif promo_price:
        add("Акционная цена", promo_price)

    elif regular_price:
        add("Цена", regular_price)

    if card_price and card_price not in {main_price, promo_price}:
        add("С картой", card_price)

    elif card_price and no_card_price and card_price != no_card_price:
        add("С картой", card_price)

    if no_card_price:
        add("Без карты", no_card_price)

    if old_price and old_price not in {regular_price, no_card_price}:
        add("Старая цена", old_price)

    add("Алкоголь", parsed.get("alcohol_percent_raw"))
    add("Скидка", parsed.get("discount_percent_raw"))
    add("Акция", parsed.get("promo_condition"))
    add("ШК", parsed.get("barcode_text_raw"))

    if parsed.get("needs_review"):
        reasons = parsed.get("review_reasons") or []
        if isinstance(reasons, list):
            reasons = ", ".join(str(x) for x in reasons[:3])
        add("Проверить", reasons or "true")

    if not semantic:
        return out

    h, w = out.shape[:2]
    font = _load_font(17, bold=False)
    line_h = 24
    panel_w = max(180, w - 10)
    max_lines = max(1, min(len(semantic), 13))
    semantic = semantic[:max_lines]
    panel_h = line_h * len(semantic) + 12
    x0 = 5
    y0 = max(0, h - panel_h - 5)

    cv2.rectangle(out, (x0, y0), (x0 + panel_w, y0 + panel_h), (255, 255, 255), -1)
    cv2.rectangle(out, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 180, 0), 2)

    img_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    yy = y0 + 6
    max_chars = max(24, int(panel_w / 8.8))

    for line in semantic:
        draw.text((x0 + 8, yy), truncate_text(line, max_chars), fill=(0, 0, 0), font=font)
        yy += line_h

    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

def draw_debug_with_ocr(image: np.ndarray,
                        boxes: List[Box], codes: List[DecodedCode],
                        title_lines: List[str],
                        ocr_by_zone: Mapping[str, List[OCRItem]] | None = None,
                        parsed: Mapping[str, Any] | None = None,
                        full_ocr_items: List[OCRItem] | None = None,) -> np.ndarray:

    out = image.copy()
    ocr_by_zone = ocr_by_zone or {}
    parsed = parsed or {}

    if full_ocr_items:
        out = _draw_full_ocr_items(out, list(full_ocr_items))

    for b in boxes:
        color = COLOR_BY_CLASS.get(b.cls, (200, 200, 200))
        cv2.rectangle(out, (b.x1, b.y1), (b.x2, b.y2), color, 2)
        ztxt = truncate_text(_zone_text(list(ocr_by_zone.get(b.cls, []))), 36)

        if not ztxt:
            ztxt = truncate_text(_semantic_text_for_box(b.cls, parsed), 36)

        if ztxt:
            label = f"{b.cls}: {ztxt}"

        else:
            label = f"{b.cls}" if b.source == "ocr_spatial" else f"{b.cls} {b.conf:.2f}"
        out = _draw_label_pil(out, label, b.x1, max(0, b.y1 - 4), color, font_size=15)

    for c in codes:
        if c.bbox:
            x1, y1, x2, y2 = c.bbox
            color = (0, 255, 255) if c.decoded else (0, 128, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            payload = truncate_text(c.payload, 28) if c.decoded else "not_decoded"
            label = f"{c.kind}:{c.fmt}:{payload}"
            out = _draw_label_pil(out, label, x1, max(0, y1 - 4), color, font_size=15)

    out = _draw_semantic_panel(out, parsed)
    return draw_top_panel(out, title_lines)
