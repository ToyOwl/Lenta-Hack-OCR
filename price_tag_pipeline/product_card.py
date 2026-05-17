"""
Retail-domain postprocessing for the final product card.

This module is deliberately deterministic.  It enriches the OCR/LLM output with
stable business fields used by the debug overlay and downstream JSON consumers:
product type, producer/brand, country, package size, regular/promo/card/no-card
prices, scale number and separated alcohol-vs-promo percentage fields.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Mapping, Optional

_EMPTY_PRICE_KEYS = {"", "none", "null", "nan", "-"}

_VISUAL_LATIN_TO_CYR = str.maketrans({
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
    "a": "а", "c": "с", "e": "е", "h": "н", "k": "к", "m": "м", "o": "о", "p": "р", "x": "х", "y": "у",
})

_PRODUCT_REPLACEMENTS = [
    (re.compile(r"\b[НH][аa][пn][кk][тt][оo0][кk]\b", re.I), "Напиток"),
    (re.compile(r"\b[НH][аa][пn][иuі1l]?[тt][оo0][кk]\b", re.I), "Напиток"),
    (re.compile(r"\bб[еeсcз3]{1,3}алк[оo0]г[оo0]льн[ыьi1l][йиu]\w*\b", re.I), "безалкогольный"),
    (re.compile(r"\bLGHT\b", re.I), "LIGHT"),
    (re.compile(r"\bLIGTH\b", re.I), "LIGHT"),
    (re.compile(r"\bP[ОO0]CCH[АA]\b", re.I), "Россия"),
    (re.compile(r"\bC\s*a\s*x\s*a\s*p\b", re.I), "Сахар"),
    (re.compile(r"\bPYCCKOE\s*MOPE\b", re.I), "РУССКОЕ МОРЕ"),
    (re.compile(r"\bРУССКОЕ\s*МОРЕ\b", re.I), "РУССКОЕ МОРЕ"),
    (re.compile(r"\bN\s*[ЕE]\s*S\s*Q\s*[UИІ]\s*[IІ]?\s*[KК]\b", re.I), "NESQUIK"),
    (re.compile(r"\bN[ЕE]S[QҚ][UИІ][IІ]?[KК]\b", re.I), "NESQUIK"),
    (re.compile(r"\bN[ЕE]SQUIK\b", re.I), "NESQUIK"),
    (re.compile(r"\bКек\s+365\s+ДН", re.I), "Хек 365 ДН"),
    (re.compile(r"\bс\s*/\s*м\b", re.I), "с/м"),
    (re.compile(r"\((?:POCa|POCа|РOCа|Роса|Росм|Pocm)\)", re.I), "(Россия)"),
]

_COUNTRY_ALIASES = [
    ("рос", "Россия"),
    ("рф", "Россия"),
    ("беларус", "Беларусь"),
    ("казах", "Казахстан"),
    ("китай", "Китай"),
    ("турц", "Турция"),
]

_KNOWN_BRANDS = [
    "NESQUIK",
    "365 ДНЕЙ",
    "РУССКОЕ МОРЕ",
    "SANTO STEFANO",
    "ABRAU LIGHT",
    "COFFESSO",
    "ЛЕНТА",
]

_ALCOHOL_WORDS_RE = re.compile(r"\b(?:пиво|beer|ale|эль|lager|stout|porter|сидр|cider|вино|водка|коньяк|ликер|алк)\b", re.I)
_PROMO_WORDS_RE = re.compile(r"\b(?:скидк|акци|выгод|эконом|промо|распродаж|sale|discount)\w*\b", re.I)
_STOCKOUT_REPLACEMENTS = str.maketrans({
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
    "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
    "a": "а", "c": "с", "e": "е", "h": "н", "k": "к", "m": "м",
    "o": "о", "p": "р", "x": "х", "y": "у", "0": "о", "3": "з", "4": "ч",
})
_STOCKOUT_PATTERNS = (
    re.compile(r"\bтовар\w*\s+законч\w*", re.I),
    re.compile(r"\bзакончил[асиоь]*\b", re.I),
    re.compile(r"\bзахонч[иы]л\w*", re.I),
    re.compile(r"\bскоро\s+привез\w*", re.I),
    re.compile(r"\bупс\b.*\bтовар", re.I),
    re.compile(r"\bвсе\s+разобрал\w*", re.I),
)
_STOCKOUT_DENSE_PATTERNS = (
    # OCR often removes spaces/punctuation: ``Упс! Товар закончился`` ->
    # ``УпсйТодорзахончился``.  Dense patterns are used only for stockout
    # detection and never for product matching.
    re.compile(r"упс.{0,16}(?:товар|тодор|то[вд]ар).{0,16}за[кх]онч", re.I),
    re.compile(r"(?:товар|тодор|то[вд]ар).{0,16}за[кх]онч", re.I),
    re.compile(r"скоро.{0,8}привез", re.I),
)


def normalize_price_value(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        try:
            f = float(v)
            if not math.isfinite(f):
                return None
            return f"{f:.2f}"
        except Exception:
            return None
    s = str(v).strip().lower()
    if s in _EMPTY_PRICE_KEYS:
        return None
    s = s.replace(",", ".")
    s = re.sub(r"[^0-9.]", "", s)
    if not s:
        return None
    if s.count(".") > 1:
        parts = [p for p in s.split(".") if p]
        s = parts[0] + "." + "".join(parts[1:])[:2]
    if "." not in s and len(s) >= 4:
        s = s[:-2] + "." + s[-2:]
    try:
        return f"{float(s):.2f}"
    except Exception:
        return None


def normalize_percent_raw(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    m = re.search(r"(\d{1,2}(?:[,.]\d{1,2})?)\s*%", s)
    if not m:
        return None
    num = m.group(1).replace(",", ".")
    return f"{num}%"


def detect_stock_status_from_text(text: Any) -> Dict[str, Any]:
    """Detect service tags such as "товар закончился" from noisy OCR text.

    The check is intentionally deterministic and tolerant to mixed Latin/Cyrillic
    OCR symbols.  It does not infer stock status from missing price alone; only
    explicit stockout/service phrases are accepted.
    """
    raw = " ".join(str(text or "").replace("\n", " ").split())
    if not raw:
        return {"stock_status": None, "confidence": 0.0, "matched_text": ""}
    norm = raw.translate(_STOCKOUT_REPLACEMENTS).lower().replace("ё", "е")
    norm = re.sub(r"[^0-9a-zа-я%]+", " ", norm, flags=re.I)
    norm = re.sub(r"\s+", " ", norm).strip()

    matches = [rx.pattern for rx in _STOCKOUT_PATTERNS if rx.search(norm)]
    dense = re.sub(r"[^0-9a-zа-я]+", "", norm, flags=re.I)
    dense_matches = [rx.pattern for rx in _STOCKOUT_DENSE_PATTERNS if rx.search(dense)]
    if not matches and not dense_matches:
        return {"stock_status": None, "confidence": 0.0, "matched_text": ""}

    # A direct "товар закончился" phrase is stronger than isolated service words.
    direct = bool(re.search(r"\bтовар\w*\s+законч\w*", norm, flags=re.I)) or bool(dense_matches)
    support = len(matches) + len(dense_matches)
    confidence = 0.90 if direct else min(0.82, 0.50 + 0.12 * support)
    return {
        "stock_status": "out_of_stock",
        "confidence": round(float(confidence), 4),
        "matched_text": raw[:220],
        "normalized_text": norm[:220],
    }


def detect_stock_status_from_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    parts = []
    ocr = result.get("ocr") if isinstance(result.get("ocr"), Mapping) else {}
    if ocr:
        parts.append(str(ocr.get("all_text_joined") or ""))
        zone_texts = ocr.get("zone_texts") if isinstance(ocr.get("zone_texts"), Mapping) else {}
        parts.extend(str(v) for v in zone_texts.values() if v not in (None, ""))
        for item in ocr.get("full_tag") or []:
            if isinstance(item, Mapping) and item.get("text"):
                parts.append(str(item.get("text")))
    parsed = result.get("parsed") if isinstance(result.get("parsed"), Mapping) else {}
    parts.extend(str(v) for v in parsed.values() if v not in (None, ""))
    return detect_stock_status_from_text(" | ".join(parts))


def smart_visual_fix(s: str) -> str:
    out = str(s or "")
    for rx, repl in _PRODUCT_REPLACEMENTS:
        out = rx.sub(repl, out)

    # Fix common fully-visual OCR tokens without corrupting real Latin brands.
    tokens = []
    for tok in re.split(r"(\W+)", out):
        if not tok or re.fullmatch(r"\W+", tok):
            tokens.append(tok)
            continue
        visual = tok.translate(_VISUAL_LATIN_TO_CYR)
        vlow = visual.lower().replace("ё", "е")
        if vlow in {"сахар", "россия", "беларусь"}:
            tokens.append(visual)
        elif re.fullmatch(r"[A-ZА-ЯЁ0-9]{4,}", tok) and not re.search(r"[a-z]", tok):
            # Uppercase mixed OCR like PYCCKOEMOPE is usually Cyrillic text.
            if "РУССКОЕ" in visual or "МОРЕ" in visual:
                tokens.append("РУССКОЕ МОРЕ")
            else:
                tokens.append(tok)
        else:
            tokens.append(tok)
    out = "".join(tokens)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def clean_product_name(s: str) -> str:
    s = str(s or "").replace("|", " ")
    s = smart_visual_fix(s)

    protected = []

    def protect(m: re.Match[str]) -> str:
        protected.append(m.group(0))
        return f" __KEEP{len(protected) - 1}__ "

    # Keep package/volume fragments and numeric brands such as "365 ДНЕЙ".
    s = re.sub(r"\b365\s*дн(?:ей|я|)\b", protect, s, flags=re.I)
    s = re.sub(r"\b\d+(?:[,.]\d+)?\s*(?:кг|kg|г|гр|g|r|л|l|мл|ml)\b", protect, s, flags=re.I)
    s = re.sub(r"\b\d+(?:[,.]\d+)?\s*%\b", protect, s, flags=re.I)

    # Remove service text and standalone codes/prices.
    s = re.sub(r"\b(?:шт|руб|py6|pуб|акция|цена|карта|картой|лент[аы]|номер|весах|код|qr|ean|товар|законч\w*|захонч\w*|скоро|привез\w*|упс)\b", " ", s, flags=re.I)
    s = re.sub(r"\b\d{1,6}(?:[.,]\d{1,2})?\b", " ", s)

    for i, val in enumerate(protected):
        s = s.replace(f"__KEEP{i}__", val)

    # Cosmetic fixes.
    s = re.sub(r"\b(\d+(?:[,.]\d+)?)\s*r\b", r"\1г", s, flags=re.I)
    s = re.sub(r"\s*\(\s*", " (", s)
    s = re.sub(r"\s*\)\s*", ") ", s)
    s = re.sub(r"\s+", " ", s).strip(" -;,.|:")
    s = _cleanup_known_brand_context(s)
    return s


def _cleanup_known_brand_context(s: str) -> str:
    """Trim OCR garbage around known brand anchors without inventing a SKU.

    This is intentionally a product-text cleanup, not catalog matching.  It keeps
    the observed brand/variant/volume tokens and drops long noisy prefixes such
    as ``Coanuramimu иамиток`` when a strong brand anchor is present.
    """
    text = re.sub(r"\s+", " ", str(s or "")).strip()
    if not text:
        return ""
    upper = text.upper().replace("Ё", "Е")

    def rebuild(prefix: str, brand: str, tail_re: str = r"") -> str:
        tail = ""
        m = re.search(re.escape(brand) + r"(?P<tail>.{0,80})", text, flags=re.I)
        if m:
            tail_src = m.group("tail")
            keep = []
            for tok in re.findall(r"[A-Za-zА-Яа-яЁё0-9.,%/-]+", tail_src)[:8]:
                tl = tok.lower().replace("ё", "е")
                if re.fullmatch(r"(?:rosso|rose|brut|bianco|zero|mokka|crema|massimo|мокка|крема|массимо|розовый|красный|белый|брют|полусладкий|сухой|0[,.]?\d+л?|\d+(?:[,.]\d+)?\s*(?:л|г|кг|мл))", tl, flags=re.I):
                    keep.append(tok)
            tail = " ".join(keep)
        return re.sub(r"\s+", " ", f"{prefix} {brand} {tail}".strip())

    if "SANTO STEFANO" in upper:
        prefix = "Напиток безалкогольный" if re.search(r"безалк|напит", text, flags=re.I) else ""
        if not prefix and re.search(r"вино|игрист|сидр", text, flags=re.I):
            prefix = "Вино игристое"
        return rebuild(prefix, "SANTO STEFANO")
    if "ABRAU LIGHT" in upper:
        prefix = "Напиток безалкогольный" if re.search(r"безалк|напит", text, flags=re.I) else ""
        return rebuild(prefix, "ABRAU LIGHT")
    if "COFFESSO" in upper:
        prefix = "Кофе" if re.search(r"коф|молот|зерн", text, flags=re.I) else ""
        return rebuild(prefix, "COFFESSO")
    return text


def product_text_quality(text: Any) -> Dict[str, Any]:
    """Lightweight OCR-product quality score.

    Returns a score in [0, 1].  It is used to treat spatial product text as a
    prior instead of unconditional ground truth.  The function is conservative:
    brands and product-category words increase the score; stockout/service text,
    mostly numeric text and dense OCR garbage decrease it.
    """
    raw = str(text or "").strip()
    cleaned = clean_product_name(raw) if raw else ""
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", cleaned)
    digits = re.findall(r"\d", cleaned)
    words = re.findall(r"[A-Za-zА-Яа-яЁё]{2,}", cleaned)
    upper = cleaned.upper().replace("Ё", "Е")
    low = cleaned.lower().replace("ё", "е")
    stock = detect_stock_status_from_text(raw)
    score = 0.0
    if letters:
        score += min(0.38, len(letters) / 80.0)
    if words:
        score += min(0.28, len(words) * 0.055)
    has_known_brand = any(b in upper for b in _KNOWN_BRANDS)
    has_product_category = bool(re.search(r"\b(?:напиток|вино|кофе|чай|молоко|сыр|сок|пиво|сидр|шоколад|печенье|пластины|инсектицид|носки|мыло|шампунь|йогурт|кефир|хлеб|сырок|колбас|сосиск|масло|крупа|рис|макарон|соус|конфет|батончик)\b", low, flags=re.I))
    if has_known_brand:
        score += 0.32
    if has_product_category:
        score += 0.16
    if stock.get("stock_status") == "out_of_stock":
        score -= 0.85
    if not letters or (digits and len(digits) >= len(letters)):
        score -= 0.25
    dense_tokens = [w for w in words if len(w) >= 14]
    if dense_tokens and len(words) <= 2 and not has_known_brand:
        score -= 0.35
    has_mixed_scripts = bool(re.search(r"[a-z]", cleaned, flags=re.I) and re.search(r"[а-яё]", cleaned, flags=re.I))
    has_digits_inside_words = bool(re.search(r"[a-zа-яё]+\d+|\d+[a-zа-яё]+", cleaned, flags=re.I))
    if not has_known_brand and not has_product_category and (has_mixed_scripts or has_digits_inside_words):
        score -= 0.30
    if re.search(r"\b(?:росом|рес10|res10|qr|ean|штрих|номер|весах|карта|цена|руб)\b", low):
        score -= 0.20
    return {
        "score": round(max(0.0, min(1.0, score)), 4),
        "cleaned": cleaned,
        "letters": len(letters),
        "words": len(words),
        "stock_status": stock.get("stock_status"),
    }


def is_plausible_product_name_text(text: Any, *, min_score: float = 0.34) -> bool:
    q = product_text_quality(text)
    return bool(q.get("score", 0.0) >= float(min_score) and q.get("letters", 0) >= 4 and not q.get("stock_status"))



def extract_volume(text: str) -> Optional[str]:
    s = smart_visual_fix(str(text or ""))
    # OCR often turns Cyrillic "г" into Latin "r" in 900г.
    m = re.search(r"\b(\d+(?:[,.]\d+)?)\s*(кг|kg|г|гр|g|r|л|l|мл|ml)\b", s, flags=re.I)
    if not m:
        return None
    value = m.group(1).replace(",", ".")
    unit = m.group(2).lower()
    unit = {"kg": "кг", "g": "г", "r": "г", "гр": "г", "l": "л", "ml": "мл"}.get(unit, unit)
    return f"{value} {unit}"


def extract_country(text: str) -> Optional[str]:
    s = smart_visual_fix(str(text or ""))
    # Prefer parenthesized country fragments.
    for frag in re.findall(r"\(([^)]{2,32})\)", s):
        low = frag.lower().replace("ё", "е")
        for needle, country in _COUNTRY_ALIASES:
            if needle in low:
                return country
    low = s.lower().replace("ё", "е")
    for needle, country in _COUNTRY_ALIASES:
        if needle in low:
            return country
    return None


def extract_alcohol_percent(text: str) -> Optional[str]:
    s = str(text or "")
    percents = list(re.finditer(r"(\d{1,2}(?:[,.]\d{1,2})?)\s*%", s))
    if not percents:
        return None
    has_alcohol_context = bool(_ALCOHOL_WORDS_RE.search(s)) or bool(re.search(r"\b\d+(?:[,.]\d+)?\s*(?:л|l|мл|ml)\b", s, re.I))
    for m in percents:
        window = s[max(0, m.start() - 40): min(len(s), m.end() + 80)]
        if _PROMO_WORDS_RE.search(window):
            continue
        if has_alcohol_context or re.search(r"\b\d+(?:[,.]\d+)?\s*(?:л|l|мл|ml)\b", window, re.I):
            return f"{m.group(1).replace(',', '.')}%"
    return None


def extract_discount_percent(text: str, *, alcohol_percent: Optional[str] = None, promo_context: bool = False) -> Optional[str]:
    s = str(text or "")
    fallback: Optional[str] = None
    for m in re.finditer(r"([-−]?\s*)(\d{1,2}(?:[,.]\d{1,2})?)\s*%", s):
        raw_num = m.group(2).replace(',', '.')
        val = f"{raw_num}%"
        try:
            fval = float(raw_num)
        except Exception:
            fval = -1.0
        window = s[max(0, m.start() - 45): min(len(s), m.end() + 45)]
        has_minus = bool(m.group(1).strip()) or bool(re.search(r"[-−]\s*\d{1,2}(?:[,.]\d{1,2})?\s*%", window))
        if val == alcohol_percent and not (_PROMO_WORDS_RE.search(window) or has_minus):
            continue
        if _PROMO_WORDS_RE.search(window) or has_minus:
            return val
        # Red promo templates often contain only a round "-17%" badge, where
        # OCR keeps just "17%".  Accept such values only in promo context and
        # only in a realistic discount interval.  Alcohol strength is filtered
        # above when it is detectable as a product attribute.
        if promo_context and 3.0 <= fval <= 95.0 and val != alcohol_percent:
            fallback = fallback or val
    return fallback


def extract_producer_or_brand(product_name: str, all_text: str = "", catalog_match: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    if catalog_match:
        for key in ("brand", "manufacturer", "producer"):
            v = str(catalog_match.get(key, "") or "").strip()
            if v:
                return v
    # Known brands may be outside the cleaned product name, but the conservative
    # uppercase fallback must use product_name only to avoid service headers like АКЦИЯ.
    src = smart_visual_fix(f"{product_name} {all_text}")
    upper = src.upper().replace("Ё", "Е")
    for brand in _KNOWN_BRANDS:
        if brand in upper:
            return brand
    # Conservative fallback: a long all-uppercase token near the beginning.
    head = " ".join(smart_visual_fix(product_name).split()[:5])
    for tok in re.findall(r"[A-ZА-ЯЁ][A-ZА-ЯЁ0-9-]{2,}", head):
        if tok.upper() not in {"РОССИЯ", "БЕЛАРУСЬ", "АКЦИЯ", "ЛЕНТА", "ЦЕНА"}:
            return tok
    return None


def infer_product_type(*, product_name: str, all_text: str, scale_number: Optional[str], volume: Optional[str], alcohol_percent: Optional[str]) -> str:
    s = f"{product_name} {all_text}".lower().replace("ё", "е")
    if alcohol_percent or _ALCOHOL_WORDS_RE.search(s):
        return "alcohol"
    if scale_number:
        return "weighted_scale"
    if volume:
        return "packaged"
    if re.search(r"\b(?:шт|штук|упак|упаковк)\b", s):
        return "piece"
    return "unknown"


def product_type_label(product_type: str) -> str:
    return {
        "weighted_scale": "весовой товар",
        "packaged": "упаковка/штука",
        "piece": "штучный товар",
        "alcohol": "алкоголь",
        "unknown": "не определён",
    }.get(str(product_type or "unknown"), "не определён")



def _product_name_score(text: str) -> float:
    if detect_stock_status_from_text(text).get("stock_status") == "out_of_stock":
        return -1e9
    t = clean_product_name(text)
    if not t:
        return -1e9
    low = t.lower().replace("ё", "е")
    service_penalty = 0.0
    if any(x in low for x in ["весах", "номер", "цен", "карт", "крт", "акци", "руб", "покуп", "неболее", "указаны", "законч", "захонч", "упс"]):
        service_penalty += 220.0
    letters = len(re.findall(r"[A-Za-zА-Яа-яЁё]", t))
    cyr = len(re.findall(r"[А-Яа-яЁё]", t))
    words = len(re.findall(r"[A-Za-zА-Яа-яЁё]{2,}", t))
    upper = t.upper().replace("Ё", "Е")
    brand_bonus = 12.0 if re.search(r"\b365\s*д", t, flags=re.I) else 0.0
    for brand in _KNOWN_BRANDS:
        if brand in upper:
            brand_bonus += 18.0 if len(brand) >= 8 else 10.0
    if words <= 2 and any(len(w) >= 14 for w in re.findall(r"[A-Za-zА-Яа-яЁё]{2,}", t)) and brand_bonus <= 0:
        service_penalty += 45.0
    return letters + cyr * 0.4 + words * 5.0 + brand_bonus - service_penalty


def _split_ocr_joined(text: str) -> list[str]:
    return [x.strip() for x in str(text or "").split("|") if x.strip()]


def select_product_name(result: Mapping[str, Any], merged: Mapping[str, Any], final: Mapping[str, Any]) -> str:
    candidates: list[str] = []
    spatial_fields_src = (result.get("spatial") or {}).get("fields") or {}
    for src in (final, merged, result.get("parsed") or {}):
        if isinstance(src, Mapping) and src.get("product_name"):
            candidates.append(str(src.get("product_name")))
    if isinstance(spatial_fields_src, Mapping) and spatial_fields_src.get("product_name"):
        sp = str(spatial_fields_src.get("product_name"))
        if is_plausible_product_name_text(sp):
            candidates.append(sp)
    zone_texts = (result.get("ocr") or {}).get("zone_texts") or {}
    if isinstance(zone_texts, Mapping):
        for key in ("product_name", "old_price", "promo_label"):
            if zone_texts.get(key):
                val = str(zone_texts.get(key))
                val = re.split(r"\b(?:цена|карта|номер|весах)\b", val, maxsplit=1, flags=re.I)[0]
                candidates.append(val)
    all_text = str((result.get("ocr") or {}).get("all_text_joined") or "")
    parts = _split_ocr_joined(all_text)
    for i, part in enumerate(parts[:28]):
        low = part.lower().replace("ё", "е")
        if any(x in low for x in ["цена", "карта", "номер", "весах", "руб", "qr", "ean", "штрих"]):
            continue
        if re.fullmatch(r"[\d\W]+", part):
            continue
        # Keep short weight fragments only if previous line is a real product line.
        if re.fullmatch(r"\d+\s*(?:г|гр|r|кг|л|l|мл|ml)", part, flags=re.I) and candidates:
            candidates[-1] = candidates[-1] + " " + part
            continue
        candidates.append(part)
    if not candidates:
        return ""
    best = max(candidates, key=_product_name_score)
    cleaned = clean_product_name(best)
    # Remove repeated brand phrase caused by multiple OCR lines.
    cleaned = re.sub(r"\b(РУССКОЕ МОРЕ)(?:\s+\1)+\b", r"\1", cleaned, flags=re.I)
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", cleaned)
    if len(letters) < 4 or len(cleaned) <= 3:
        return ""
    if _product_name_score(cleaned) < 4.0:
        return ""
    if not is_plausible_product_name_text(cleaned, min_score=0.26):
        return ""
    return cleaned


_PRICE_OCR_CHAR_MAP = str.maketrans({
    "O": "0", "О": "0", "о": "0", "o": "0",
    "I": "1", "l": "1", "|": "1", "!": "1", "і": "1",
    "S": "5", "s": "5", "Б": "6", "б": "6",
    "З": "3", "з": "3",
    "В": "8", "в": "8",
})


def normalize_price_integer_as_rubles(v: Any) -> Optional[str]:
    """Normalize OCR integer price as rubles, not as compact kopecks.

    ``normalize_price_value('179')`` already returns ``179.00`` while
    ``normalize_price_value('1799')`` returns ``17.99`` by design for compact
    OCR.  For full-tag OCR of a single large price field this is often wrong:
    the detector crop may contain only ``179``/``99``/``1299`` as rubles.  This
    helper is therefore used only by the price-only fallback path.
    """
    if v is None:
        return None
    s = str(v).strip().translate(_PRICE_OCR_CHAR_MAP)
    s = s.replace("₽", " ").replace("р", " ").replace("Р", " ")
    s = re.sub(r"[^0-9]", "", s)
    if not re.fullmatch(r"\d{1,5}", s or ""):
        return None
    try:
        val = int(s)
    except Exception:
        return None
    if val <= 0 or val > 99999:
        return None
    return f"{float(val):.2f}"


def resolve_decimal_shift_with_price_only(primary: Any, price_only: Optional[Mapping[str, Any]], *, min_larger_price: float = 5.0) -> Optional[str]:
    """Prefer ruble-scale price when OCR alternates between 179 and 1.79.

    The parser can interpret a raw ``179`` as compact kopecks (1.79) in some
    template zones, while full-tag price-only OCR usually means 179 rubles.  If
    both interpretations share the same visible digit sequence, prefer the
    larger integer-rubles value.
    """
    base = normalize_price_value(primary)
    if not base:
        return None
    if not isinstance(price_only, Mapping):
        return base
    try:
        base_f = float(base)
    except Exception:
        base_f = 0.0
    key = _decimal_shift_key(base)
    candidates = price_only.get("candidates") if isinstance(price_only.get("candidates"), list) else []
    larger = []
    for c in candidates:
        if not isinstance(c, Mapping):
            continue
        val = normalize_price_value(c.get("value"))
        if not val:
            continue
        if _decimal_shift_key(val) == key and float(val) > base_f and float(val) >= float(min_larger_price):
            if val.endswith(".00") or ".integer_rubles" in str(c.get("source") or ""):
                larger.append(val)
    if larger:
        return max(larger, key=lambda x: float(x))
    return base


def _decimal_shift_key(price: Any) -> str:
    norm = normalize_price_value(price)
    if not norm:
        return ""
    ip, fp = norm.split(".", 1)
    ip = ip.lstrip("0") or "0"
    if fp == "00" and float(norm) >= 5.0:
        return ip
    if 0.0 < float(norm) < 100.0 and fp != "00":
        return (ip + fp[:2]).lstrip("0") or "0"
    return (ip + ("" if fp == "00" else fp[:2])).lstrip("0") or "0"


def extract_price_only_candidate_from_result(result: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract the main price when the only readable field is a big number.

    This closes the common mobile-video failure mode where zone OCR/spatial OCR
    do not fill ``prices.main`` but full-tag OCR contains exactly ``99`` or
    ``179``.  The function is intentionally conservative:
    - explicit decimal forms still win when present;
    - standalone 1-5 digit numbers are accepted as rubles only from OCR lines;
    - obvious percentages, barcodes and service lines are skipped.
    """
    candidates: list[Dict[str, Any]] = []

    def add(value: Optional[str], raw: str, source: str, score: float) -> None:
        norm = normalize_price_value(value)
        if not norm:
            return
        try:
            f = float(norm)
        except Exception:
            return
        if f <= 0 or f > 99999:
            return
        candidates.append({"value": norm, "raw": raw, "source": source, "score": float(score)})

    # Existing parser output: keep it, but allow later OCR-integer evidence to
    # override suspicious compact prices such as 1.09 from a raw ``109`` line.
    prices = result.get("prices") if isinstance(result.get("prices"), Mapping) else {}
    main = prices.get("main") if isinstance(prices.get("main"), Mapping) else {}
    if main.get("value") not in (None, ""):
        add(str(main.get("value")), str(main.get("raw_match") or main.get("value")), "prices.main", 0.74)

    by_zone = prices.get("by_zone") if isinstance(prices.get("by_zone"), Mapping) else {}
    for zone_name, vals in by_zone.items():
        if not isinstance(vals, list):
            continue
        for c in vals[:3]:
            if isinstance(c, Mapping) and c.get("value") not in (None, ""):
                zscore = 0.70 if str(zone_name).startswith("main_price") else 0.58
                add(str(c.get("value")), str(c.get("raw_match") or c.get("value")), f"prices.by_zone.{zone_name}", zscore)

    # Spatial candidates are already geometry-aware.
    spatial = result.get("spatial") if isinstance(result.get("spatial"), Mapping) else {}
    for c in spatial.get("price_candidates") or []:
        if isinstance(c, Mapping) and c.get("value") not in (None, ""):
            add(str(c.get("value")), str(c.get("raw_match") or c.get("value")), "spatial.price_candidates", 0.80)

    # Direct fields.
    for src_name, src in (
        ("parsed", result.get("parsed") if isinstance(result.get("parsed"), Mapping) else {}),
        ("spatial.fields", spatial.get("fields") if isinstance(spatial.get("fields"), Mapping) else {}),
    ):
        for key in ("main_price", "card_price", "no_card_price", "old_price"):
            if src.get(key) not in (None, ""):
                add(str(src.get(key)), str(src.get(key)), f"{src_name}.{key}", 0.72)

    ocr = result.get("ocr") if isinstance(result.get("ocr"), Mapping) else {}
    texts: list[tuple[str, str, float]] = []
    zone_texts = ocr.get("zone_texts") if isinstance(ocr.get("zone_texts"), Mapping) else {}
    for zone in ("main_price", "main_price_rubles", "main_price_kopeeks", "card_price", "no_card_price"):
        t = str(zone_texts.get(zone) or "").strip()
        if t:
            texts.append((t, f"ocr.zone_texts.{zone}", 0.86 if zone.startswith("main_price") else 0.68))
    for item in ocr.get("full_tag") or []:
        if isinstance(item, Mapping):
            t = str(item.get("text") or "").strip()
            conf = 0.0
            try:
                conf = float(item.get("conf") or 0.0)
            except Exception:
                pass
            if t:
                texts.append((t, "ocr.full_tag", 0.78 + min(0.12, max(0.0, conf - 0.70) * 0.20)))
    all_text = str(ocr.get("all_text_joined") or "").strip()
    if all_text:
        for part in [x.strip() for x in all_text.split("|") if x.strip()]:
            texts.append((part, "ocr.all_text_joined", 0.62))

    for raw, source, base_score in texts:
        low = raw.lower().replace("ё", "е")
        if any(x in low for x in ("qr", "ean", "штрих", "баркод", "barcode", "номер", "код")):
            continue
        if "%" in raw:
            continue
        t = raw.translate(_PRICE_OCR_CHAR_MAP)
        t = t.replace("₽", " ").replace("р", " ").replace("Р", " ")

        # Explicit decimal/split price: 179.99, 179,99, 179 99, 179-99.
        for m in re.finditer(r"(?<!\d)(\d{1,5})\s*([,.\-])\s*(\d{2})(?!\d)", t):
            rub = int(m.group(1))
            kop = int(m.group(3))
            if rub > 0 and 0 <= kop <= 99:
                add(f"{rub}.{kop:02d}", m.group(0), source + ".decimal", base_score + 0.05)

        # Standalone integer price as rubles.  This is the critical path for
        # detections where only the large ``99``/``179`` field is readable.
        compact = re.sub(r"[^0-9]", " ", t)
        for m in re.finditer(r"(?<!\d)(\d{1,5})(?!\d)", compact):
            digits = m.group(1)
            # Barcodes/articles are longer and have lower OCR relevance here.
            if len(digits) > 5:
                continue
            val = normalize_price_integer_as_rubles(digits)
            if val:
                score = base_score + (0.08 if len(digits) in (2, 3) else 0.02)
                add(val, digits, source + ".integer_rubles", score)

    if not candidates:
        return None

    # Collapse by value, preserving the strongest evidence.
    best_by_value: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        key = str(c["value"])
        prev = best_by_value.get(key)
        if prev is None or float(c["score"]) > float(prev["score"]):
            best_by_value[key] = dict(c)
    ordered = sorted(best_by_value.values(), key=lambda c: float(c["score"]), reverse=True)

    # If the top parser candidate is a tiny compact price but OCR also saw a
    # plausible integer-rubles value, prefer the integer value.  This handles
    # ``109`` -> wrong ``1.09`` on blurred 99/109 crops.
    top = ordered[0]
    try:
        top_f = float(top["value"])
    except Exception:
        top_f = 0.0
    if top_f < 5.0:
        integer_candidates = [c for c in ordered if ".integer_rubles" in str(c.get("source", "")) and float(c["value"]) >= 5.0]
        if integer_candidates:
            top = integer_candidates[0]

    return {
        "value": str(top["value"]),
        "raw_match": str(top.get("raw") or top["value"]),
        "parser": "price_only_ocr_fallback",
        "confidence": round(min(0.96, max(0.50, float(top.get("score", 0.0)))), 4),
        "zone": str(top.get("source") or "ocr"),
        "type": "main_price_price_only",
        "candidates": ordered[:6],
    }


def _merge_parsed_with_spatial_product_prior(parsed: Mapping[str, Any], spatial_fields: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(parsed or {})
    for k, v in dict(spatial_fields or {}).items():
        if v in (None, ""):
            continue
        if str(k) == "product_name":
            # Spatial parser is a prior: use it only if it looks like product
            # text.  Do not let OCR garbage or service tags overwrite parsed
            # fields and later drive CSV matching.
            if is_plausible_product_name_text(v):
                merged.setdefault("product_name", v)
                merged.setdefault("spatial_product_name_prior", v)
            else:
                merged.setdefault("spatial_product_name_rejected", v)
            continue
        merged[k] = v
    return merged


def baseline_final_from_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    parsed = dict(result.get("parsed") or {})
    spatial_fields = dict((result.get("spatial") or {}).get("fields") or {})
    merged = _merge_parsed_with_spatial_product_prior(parsed, spatial_fields)
    price_only = extract_price_only_candidate_from_result(result)
    main = ((result.get("prices") or {}).get("main") or {}).get("value") or merged.get("main_price")
    if (not normalize_price_value(main)) and price_only:
        main = price_only.get("value")
    final = {
        "product_name": select_product_name(result, merged, {} ) or None,
        "main_price": normalize_price_value(main),
        "card_price": normalize_price_value(merged.get("card_price") or (main if main else None)),
        "no_card_price": normalize_price_value(merged.get("no_card_price") or merged.get("no_card_price_small")),
        "old_price": normalize_price_value(merged.get("old_price") or merged.get("old_price_raw")),
        "unit": merged.get("unit") or merged.get("unit_info"),
        "scale_number": merged.get("scale_number") or None,
        "barcode_text_raw": merged.get("barcode_text_raw") or None,
        "discount_percent_raw": merged.get("discount_percent_raw") or None,
        "promo_condition": merged.get("promo_condition") or None,
    }
    if price_only and final.get("main_price"):
        resolved_main = resolve_decimal_shift_with_price_only(final.get("main_price"), price_only)
        if resolved_main and resolved_main != final.get("main_price"):
            final["main_price"] = resolved_main
            final["card_price"] = resolved_main if final.get("card_price") == normalize_price_value(main) else final.get("card_price")
            final["price_source"] = "decimal_shift_resolved_from_price_only"
            final["price_confidence"] = max(float(price_only.get("confidence") or 0.0), 0.90)
    if price_only and final.get("main_price") == price_only.get("value"):
        final["price_source"] = price_only.get("zone")
        final["price_confidence"] = price_only.get("confidence")
    return enrich_final_fields(result, final)


def enrich_final_fields(result: Mapping[str, Any], final_in: Mapping[str, Any]) -> Dict[str, Any]:
    final: Dict[str, Any] = dict(final_in or {})
    parsed = dict(result.get("parsed") or {})
    spatial_fields = dict((result.get("spatial") or {}).get("fields") or {})
    merged = _merge_parsed_with_spatial_product_prior(parsed, spatial_fields)
    merged.update({k: v for k, v in final.items() if v not in (None, "")})
    template_name = str(((result.get("template") or {}).get("template_name") or ""))
    all_text = str(((result.get("ocr") or {}).get("all_text_joined") or ""))
    stock_status = detect_stock_status_from_result(result)

    raw_product = str(merged.get("product_name") or "")
    product_name = select_product_name(result, merged, final)
    if not product_name:
        fallback_product = clean_product_name(raw_product)
        if is_plausible_product_name_text(fallback_product, min_score=0.30):
            product_name = fallback_product
    if product_name:
        final["product_name"] = product_name
    else:
        final.pop("product_name", None)

    source_text = " | ".join([str(x) for x in [raw_product, product_name, all_text, merged.get("discount_percent_raw"), merged.get("promo_condition")] if x])
    volume = final.get("volume") or extract_volume(source_text)
    country = final.get("country") or extract_country(source_text)
    alcohol_percent = normalize_percent_raw(final.get("alcohol_percent_raw")) or extract_alcohol_percent(source_text)

    discount_raw_source = str(final.get("discount_percent_raw") or merged.get("discount_percent_raw") or "")
    explicit_discount = normalize_percent_raw(discount_raw_source)
    promo_context = bool("promo" in template_name.lower() or "акц" in source_text.lower() or "скид" in source_text.lower())
    discount_percent = extract_discount_percent(source_text, alcohol_percent=alcohol_percent, promo_context=promo_context)
    if explicit_discount and explicit_discount != alcohol_percent and (_PROMO_WORDS_RE.search(discount_raw_source) or discount_percent or promo_context):
        discount_percent = explicit_discount
    elif explicit_discount == alcohol_percent and _PROMO_WORDS_RE.search(discount_raw_source):
        discount_percent = explicit_discount

    scale_number = str(final.get("scale_number") or merged.get("scale_number") or "").strip() or None
    product_type = str(final.get("product_type") or "").strip() or infer_product_type(
        product_name=product_name,
        all_text=all_text,
        scale_number=scale_number,
        volume=volume,
        alcohol_percent=alcohol_percent,
    )

    match = final.get("product_match") if isinstance(final.get("product_match"), Mapping) else {}
    producer = final.get("producer") or final.get("brand") or extract_producer_or_brand(product_name, all_text, match)

    price_only = extract_price_only_candidate_from_result(result)
    for key in ("main_price", "card_price", "no_card_price", "old_price"):
        if key in final:
            final[key] = normalize_price_value(final.get(key))
        else:
            final[key] = normalize_price_value(merged.get(key))
    if not final.get("main_price") and price_only:
        final["main_price"] = price_only.get("value")
        final["price_source"] = price_only.get("zone")
        final["price_confidence"] = price_only.get("confidence")
    elif final.get("main_price") and price_only:
        resolved_main = resolve_decimal_shift_with_price_only(final.get("main_price"), price_only)
        if resolved_main and resolved_main != final.get("main_price"):
            final["main_price"] = resolved_main
            if final.get("card_price") and _decimal_shift_key(final.get("card_price")) == _decimal_shift_key(resolved_main):
                final["card_price"] = resolved_main
            final["price_source"] = "decimal_shift_resolved_from_price_only"
            final["price_confidence"] = max(float(price_only.get("confidence") or 0.0), 0.90)

    main_price = final.get("main_price")
    card_price = final.get("card_price")
    no_card_price = final.get("no_card_price")
    old_price = final.get("old_price")

    promo_price = normalize_price_value(final.get("promo_price"))
    regular_price = normalize_price_value(final.get("regular_price"))
    if not promo_price:
        if "promo" in template_name or "акц" in source_text.lower():
            promo_price = card_price or main_price
        elif card_price and no_card_price and card_price != no_card_price:
            promo_price = card_price
    if not regular_price:
        regular_price = no_card_price or old_price
        if not regular_price and not promo_price:
            regular_price = main_price

    # Populate final fields in stable order-ish. Existing keys are preserved where valid.
    final["product_type"] = product_type
    final["product_type_label"] = product_type_label(product_type)
    if producer:
        final["producer"] = str(producer)
        final.setdefault("brand", str(producer))
    if country:
        final["country"] = country
    if volume:
        final["volume"] = volume
        final.setdefault("package_size", volume)
    if scale_number:
        final["scale_number"] = scale_number
    if alcohol_percent:
        final["alcohol_percent_raw"] = alcohol_percent
    final["discount_percent_raw"] = discount_percent
    if promo_price:
        final["promo_price"] = promo_price
    if regular_price:
        final["regular_price"] = regular_price
    if product_type == "weighted_scale":
        final.setdefault("sale_unit", "за кг/весовой")
    elif product_type == "alcohol" and volume:
        final.setdefault("sale_unit", "за бутылку")
    elif volume:
        final.setdefault("sale_unit", "за упаковку")
    elif product_type == "piece":
        final.setdefault("sale_unit", "за штуку")

    if stock_status.get("stock_status") == "out_of_stock":
        final["stock_status"] = "out_of_stock"
        final["stock_status_label"] = "товар закончился"
        final["stock_status_confidence"] = stock_status.get("confidence", 0.0)
        final["stock_status_text"] = stock_status.get("matched_text", "")
        reasons = list(final.get("review_reasons") or []) if isinstance(final.get("review_reasons"), list) else []
        if "stockout_service_tag" not in reasons:
            reasons.append("stockout_service_tag")
        final["review_reasons"] = reasons

    return final
