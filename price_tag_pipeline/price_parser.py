"""
Price parsing and numeric normalization.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

OCR_PRICE_CHAR_MAP = str.maketrans({
    "O": "0", "О": "0", "о": "0", "o": "0",
    "I": "1", "l": "1", "|": "1", "!": "1", "і": "1",
    "S": "5", "s": "5", "Б": "6",
    "З": "3", "з": "3",
    "В": "8", "в": "8",
})


def normalize_price_text(text: str) -> str:
    t = text.strip().translate(OCR_PRICE_CHAR_MAP)
    t = t.replace("₽", " ").replace("р", " ").replace("Р", " ")
    t = re.sub(r"[^0-9,\.\-\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def parse_prices_from_texts(texts: Sequence[str], allow_compact: bool = True) -> List[Dict[str, Any]]:
    joined = " | ".join([str(t) for t in texts if str(t).strip()])
    norm = normalize_price_text(joined)
    candidates: List[Dict[str, Any]] = []

    # Explicit decimal forms: 119.99, 119,99, 119 99, 119-99.
    pattern = re.compile(r"(?<!\d)(\d{1,5})\s*[\.,\- ]\s*(\d{2})(?!\d)")
    for m in pattern.finditer(norm):
        rub = int(m.group(1))
        kop = int(m.group(2))
        if 0 <= kop <= 99 and rub > 0:
            candidates.append({
                "value": float(f"{rub}.{kop:02d}"),
                "rubles": rub,
                "kopeeks": kop,
                "raw_match": m.group(0),
                "parser": "regex_decimal_or_split",
                "confidence": 0.75,
            })

    if allow_compact:
        # Compact form: 11999 -> 119.99, only safe in price zones.
        compact = re.findall(r"(?<!\d)(\d{3,7})(?!\d)", norm)
        for s in compact:
            if len(s) >= 3:
                rub = int(s[:-2])
                kop = int(s[-2:])
                if rub > 0 and 0 <= kop <= 99:
                    candidates.append({
                        "value": float(f"{rub}.{kop:02d}"),
                        "rubles": rub,
                        "kopeeks": kop,
                        "raw_match": s,
                        "parser": "regex_compact",
                        "confidence": 0.55,
                    })

    return dedupe_price_candidates(candidates)


def dedupe_price_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[float, Dict[str, Any]] = {}
    for c in candidates:
        v = float(c["value"])
        if v not in best or c.get("confidence", 0) > best[v].get("confidence", 0):
            best[v] = c
    return sorted(best.values(), key=lambda x: x.get("confidence", 0), reverse=True)


def choose_main_price(zone_prices: Dict[str, List[Dict[str, Any]]], template_name: str) -> Optional[Dict[str, Any]]:
    priority = [
        "main_price",
        "main_price_rubles",
        "main_price_kopeeks",
        "old_price_or_without_card",
        "unit_price_or_article",
        "full_tag",
    ]
    for z in priority:
        vals = zone_prices.get(z) or []
        if vals:
            cand = vals[0].copy()
            cand["zone"] = z
            if template_name in {"shelf_red_promo", "progressive", "progressive_yellow"} and z.startswith("main_price"):
                cand["type"] = "with_card_or_promo_main"
            elif z == "old_price_or_without_card":
                cand["type"] = "old_or_without_card"
            else:
                cand["type"] = "main_or_unknown"
            return cand
    return None
