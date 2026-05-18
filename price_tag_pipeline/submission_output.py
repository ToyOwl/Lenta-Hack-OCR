"""
Task-level CSV/JSON export for detected price-tag tracks.

The hackathon output schema is intentionally separated from internal diagnostic
schemas.  Internal track aggregation keeps rich evidence, while this module
converts one aggregated output track into one flat record:

- fields read from the visible price tag;
- fields read from the decoded QR payload;
- frame timestamp and bbox metadata.

Policy:
- if a QR payload is not decoded, QR fields are left empty;
- if a QR payload is decoded but a specific QR parameter is absent, the field is
  filled with ``absent_value`` (default: "нет");
- for visible price-tag fields, unknown text fields are left empty, while
  optional absent fields such as discounts/additional info use ``absent_value``.
"""

from __future__ import annotations

import csv
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, unquote_plus, urlsplit

from .product_card import normalize_percent_raw, normalize_price_value

TASK_OUTPUT_FIELDS: List[str] = [
    "filename",
    "product_name",
    "price_default",
    "price_card",
    "price_discount",
    "barcode",
    "discount_amount",
    "id_sku",
    "print_datetime",
    "code",
    "additional_info",
    "color",
    "special_symbols",
    "frame_timestamp",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "qr_code_barcode",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_count",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
    "action_price_qr",
    "action_code_qr",
]

QR_FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "qr_code_barcode": ("barcode", "b", "qr_code_barcode", "ean", "gtin"),
    "price1_qr": ("price1", "p1", "price_1", "price", "p"),
    "price2_qr": ("price2", "p2", "price_2"),
    "price3_qr": ("price3", "p3", "price_3"),
    "price4_qr": ("price4", "p4", "price_4"),
    "wholesale_level_1_count": ("wholesalelevel1count", "wL1C", "wl1c", "wholesale_level_1_count"),
    "wholesale_level_1_price": ("wholesalelevel1price", "wL1P", "wl1p", "wholesale_level_1_price"),
    "wholesale_level_2_count": ("wholesalelevel2count", "wL2C", "wl2c", "wholesale_level_2_count"),
    "wholesale_level_2_price": ("wholesalelevel2price", "wL2P", "wl2p", "wholesale_level_2_price"),
    "action_price_qr": ("actionprice", "aP", "ap", "action_price"),
    "action_code_qr": ("actioncode", "aC", "ac", "action_code"),
}

OPTIONAL_VISIBLE_FIELDS = {
    "price_default",
    "price_card",
    "price_discount",
    "discount_amount",
    "print_datetime",
    "code",
    "additional_info",
    "color",
    "special_symbols",
}

PRICE_FIELDS = {
    "price_default",
    "price_card",
    "price_discount",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_price",
    "wholesale_level_2_price",
    "action_price_qr",
}


def write_task_outputs(
    tracks: Sequence[Mapping[str, Any]],
    out_dir: Path,
    cfg: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Write task CSV/JSON files and return output metadata."""
    cfg = dict(cfg or {})
    enabled = bool(cfg.get("enabled", True))
    if not enabled:
        return {"enabled": False}

    csv_name = str(cfg.get("csv_name") or "detected_tracks_task_output.csv")
    json_name = str(cfg.get("json_name") or "detected_tracks_task_output.json")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_task_output_records(tracks, cfg)
    csv_path = out_dir / csv_name
    json_path = out_dir / json_name

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TASK_OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k, "")) for k in TASK_OUTPUT_FIELDS})

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"schema": TASK_OUTPUT_FIELDS, "count": len(rows), "rows": rows}, f, ensure_ascii=False, indent=2)

    return {"enabled": True, "csv": str(csv_path), "json": str(json_path), "count": len(rows)}


def build_task_output_records(tracks: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any] | None = None) -> List[Dict[str, Any]]:
    cfg = dict(cfg or {})
    rows: List[Dict[str, Any]] = []
    for track in tracks:
        if not isinstance(track, Mapping):
            continue
        rows.append(build_task_output_record(track, cfg))
    return rows


def build_task_output_record(track: Mapping[str, Any], cfg: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    cfg = dict(cfg or {})
    absent_value = str(cfg.get("absent_value", "нет"))
    empty_value = str(cfg.get("empty_value", ""))
    main_price_target = str(cfg.get("main_price_target", "price_card") or "price_card")
    video_fps = _to_float(cfg.get("video_fps"), None)

    final = track.get("aggregated_final") if isinstance(track.get("aggregated_final"), Mapping) else {}
    best = track.get("best_observation") if isinstance(track.get("best_observation"), Mapping) else {}
    template_name = _get_template_name(track)
    decoded_codes = _collect_decoded_codes(track)
    qr_payload = _select_qr_payload(decoded_codes)
    qr_map = parse_qr_payload(qr_payload) if qr_payload else {}
    qr_was_decoded = bool(qr_payload)

    row: Dict[str, Any] = {field: empty_value for field in TASK_OUTPUT_FIELDS}
    row["filename"] = _filename_for_track(track, best)
    row["product_name"] = _first_nonempty(final.get("product_name"), final.get("product"), empty_value)
    row["barcode"] = _first_barcode(decoded_codes, final)
    row["discount_amount"] = _discount_amount_value(final, track, best, absent_value=absent_value)
    row["id_sku"] = _sku_from_final(final)
    row["print_datetime"] = _first_nonempty(final.get("print_datetime"), final.get("printed_at"), absent_value)
    row["code"] = _first_nonempty(final.get("code"), final.get("scale_code"), final.get("scale_number"), absent_value)
    row["additional_info"] = _additional_info(final, track, absent_value=absent_value)
    row["color"] = _color_value(final, template_name, track=track, best=best, cfg=cfg, absent_value=absent_value)
    row["special_symbols"] = _special_symbols(final, track, absent_value=absent_value)
    row["frame_timestamp"] = _frame_timestamp(best, cfg, video_fps=video_fps)

    x1, y1, x2, y2 = _bbox_from_best(best)
    row["x_min"], row["y_min"], row["x_max"], row["y_max"] = x1, y1, x2, y2

    prices = _visible_prices(final, track, main_price_target=main_price_target, template_name=template_name)
    for key in ("price_default", "price_card", "price_discount"):
        row[key] = prices.get(key) or absent_value

    for field, aliases in QR_FIELD_ALIASES.items():
        value = _lookup_alias(qr_map, aliases)
        if value not in (None, ""):
            row[field] = _normalize_field_value(field, value)
        elif qr_was_decoded:
            row[field] = absent_value
        else:
            row[field] = empty_value

    # Keep product/sku/barcode empty if genuinely not recognized. Optional visible
    # fields use "нет" when not present in the visible tag evidence.
    for key in OPTIONAL_VISIBLE_FIELDS:
        if row.get(key) in (None, ""):
            row[key] = absent_value
    for key in PRICE_FIELDS:
        if row.get(key) not in (None, "", absent_value):
            row[key] = _normalize_field_value(key, row[key])
    return {k: row.get(k, empty_value) for k in TASK_OUTPUT_FIELDS}


def parse_qr_payload(payload: Any) -> Dict[str, str]:
    """Parse QR payload into a case-insensitive map.

    Supports common forms:
    - ``barcode=...;price1=...``;
    - ``b:...;p1:...``;
    - URL query string with ``?b=...&p1=...``.
    """
    raw = str(payload or "").strip()
    if not raw:
        return {}
    raw = unquote_plus(raw)
    candidates = [raw]
    try:
        parsed = urlsplit(raw)
        if parsed.query:
            candidates.append(parsed.query)
    except Exception:
        pass

    out: Dict[str, str] = {}
    for candidate in candidates:
        for k, v in parse_qsl(candidate, keep_blank_values=True, strict_parsing=False):
            if k:
                out[_norm_key(k)] = str(v).strip()
        # Semicolon/comma/pipe separated key-value tokens.
        for token in re.split(r"[;|\n\r]+", candidate):
            token = token.strip()
            if not token:
                continue
            if "=" in token:
                k, v = token.split("=", 1)
            elif ":" in token:
                k, v = token.split(":", 1)
            else:
                continue
            k = k.strip()
            v = v.strip()
            if k:
                out[_norm_key(k)] = v
    return out



def _visible_prices(final: Mapping[str, Any], track: Mapping[str, Any], *, main_price_target: str, template_name: str) -> Dict[str, str]:
    """Resolve task prices with promo-aware postprocessing.

    Internal OCR/parsers can leak barcode fragments into ``price_default``
    (for example ``4.01`` from ``460140...``) or confuse old price with the
    action price.  The task schema needs semantic prices:

    - ``price_default``  — regular / without-card / old visible price;
    - ``price_card``     — card price, usually the large visible price;
    - ``price_discount`` — action/promo price when the tag is promotional.

    This function therefore prefers explicit semantic fields, but then validates
    them against the visible main price, discount badge and OCR text context.
    """
    out: Dict[str, str] = {}
    text_sources = _collect_text_sources(final, track)
    promo = _looks_promotional(final, track, template_name)

    main = _repair_kopeeks(_norm_price(final.get("main_price")), text_sources)
    badge_price = _extract_price_from_discount_badge(text_sources)
    if badge_price:
        badge_price = _repair_kopeeks(badge_price, text_sources)

    explicit_discount = _first_price_from_keys(
        final,
        ["price_discount", "discount_price", "action_price", "promo_price", "price_promo"],
        text_sources=text_sources,
    )
    explicit_card = _first_price_from_keys(
        final,
        ["price_card", "card_price", "with_card_price", "loyalty_price"],
        text_sources=text_sources,
    )
    explicit_default = _first_price_from_keys(
        final,
        ["price_default", "default_price", "regular_price", "price_regular", "price_without_card", "no_card_price", "old_price"],
        text_sources=text_sources,
        prefer_integer_rubles=True,
    )

    # The discount badge sometimes contains both percent and promo price, e.g.
    # "-28%1199".  If the current main price is actually the old price, this
    # badge-derived price is a better action/card candidate.
    if badge_price and (_price_f(main) <= 0 or _price_f(main) > _price_f(badge_price) * 1.12):
        main = badge_price

    # Main/card price is the large price unless a more specific card field is present.
    card = explicit_card or main
    if card:
        out["price_card"] = card

    discount = explicit_discount or (badge_price if promo else "") or (main if promo else "")
    if discount:
        out["price_discount"] = discount

    # Resolve regular/default price.  Prefer explicit fields, then old/no-card
    # price near keywords such as "без карты".
    context_default = _extract_default_price_from_context(text_sources, reference_price=card or discount or main)
    default = explicit_default or context_default
    if default:
        default = _repair_kopeeks(default, text_sources)

    if default and _is_suspicious_default_price(default, card or discount or main, promo=promo):
        # Do not export barcode fragments such as 4.01 / 116.84 as regular price.
        default = context_default if context_default and not _is_suspicious_default_price(context_default, card or discount or main, promo=promo) else ""

    if default:
        out["price_default"] = default

    # If the tag is promotional and price_card is missing, use discount/main as
    # the card price.  This matches Lenta red labels where the large price is
    # normally the card/action price.
    if promo and not out.get("price_card") and out.get("price_discount"):
        out["price_card"] = out["price_discount"]
    if promo and not out.get("price_discount") and out.get("price_card"):
        out["price_discount"] = out["price_card"]

    return out


def _looks_promotional(final: Mapping[str, Any], track: Mapping[str, Any], template_name: str) -> bool:
    if str(final.get("discount_percent_raw") or final.get("discount_amount") or "").strip():
        return True
    tmpl = template_name.lower()
    if any(x in tmpl for x in ("promo", "red", "progressive", "yellow")):
        return True
    warnings = " ".join(str(w) for w in (track.get("warnings") or []))
    if "promo" in warnings.lower() or "progressive" in warnings.lower():
        return True
    final_text = " ".join(str(final.get(k) or "") for k in ("special_symbols", "layout_type", "promo_header"))
    return bool(re.search(r"[-−]?\s*\d{1,2}\s*%", final_text))


def _collect_text_sources(final: Mapping[str, Any], track: Mapping[str, Any]) -> List[str]:
    """Collect OCR/debug strings used only for conservative output cleanup."""
    sources: List[str] = []

    def add(v: Any) -> None:
        if v is None:
            return
        if isinstance(v, str):
            s = v.strip()
            if s:
                sources.append(s)
            return
        if isinstance(v, (int, float)):
            sources.append(str(v))
            return
        if isinstance(v, Mapping):
            for vv in v.values():
                add(vv)
            return
        if isinstance(v, (list, tuple)):
            for vv in v:
                add(vv)

    for k in (
        "all_text_joined", "raw_text", "ocr_text", "product_name", "main_price_raw",
        "main_price_rubles_raw", "main_price_kopeeks_raw", "old_price_raw",
        "discount_percent_raw", "discount_amount", "special_symbols", "promo_header",
    ):
        add(final.get(k))

    for obs in track.get("observations") or []:
        if not isinstance(obs, Mapping):
            continue
        add(obs.get("final") if isinstance(obs.get("final"), Mapping) else {})
        raw = obs.get("raw_summary") if isinstance(obs.get("raw_summary"), Mapping) else {}
        add(raw.get("parsed") if isinstance(raw.get("parsed"), Mapping) else {})
        ocr = raw.get("ocr") if isinstance(raw.get("ocr"), Mapping) else {}
        add(ocr.get("all_text_joined"))
        add(ocr.get("texts"))
        add(ocr.get("items"))

    best = track.get("best_observation") if isinstance(track.get("best_observation"), Mapping) else {}
    add(best.get("final") if isinstance(best.get("final"), Mapping) else {})
    raw = best.get("raw_summary") if isinstance(best.get("raw_summary"), Mapping) else {}
    add(raw.get("parsed") if isinstance(raw.get("parsed"), Mapping) else {})
    ocr = raw.get("ocr") if isinstance(raw.get("ocr"), Mapping) else {}
    add(ocr.get("all_text_joined"))
    add(ocr.get("texts"))
    add(ocr.get("items"))

    # Deduplicate, keep order.
    out: List[str] = []
    seen = set()
    for s in sources:
        key = s[:500]
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _norm_price(value: Any, *, prefer_integer_rubles: bool = False) -> str:
    if value in (None, ""):
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    if prefer_integer_rubles:
        m = re.fullmatch(r"\s*(\d{2,5})\s*", raw)
        if m:
            try:
                return f"{float(int(m.group(1))):.2f}"
            except Exception:
                pass
    norm = normalize_price_value(raw)
    return str(norm or "").strip()


def _first_price_from_keys(final: Mapping[str, Any], keys: Sequence[str], *, text_sources: Sequence[str], prefer_integer_rubles: bool = False) -> str:
    for k in keys:
        v = final.get(k)
        p = _norm_price(v, prefer_integer_rubles=prefer_integer_rubles)
        if p:
            return _repair_kopeeks(p, text_sources)
    return ""


def _price_f(value: Any) -> float:
    try:
        return float(str(value).replace(",", "."))
    except Exception:
        return 0.0


def _repair_kopeeks(price: str, text_sources: Sequence[str]) -> str:
    """Repair integer ruble prices when OCR text contains adjacent kopecks.

    Example: final main price is ``129.00`` but OCR text contains ``129 99``;
    return ``129.99``.  The regex requires digit boundaries to avoid matching
    inside barcodes.
    """
    if not price:
        return ""
    try:
        f = float(price)
    except Exception:
        return price
    if abs(f - round(f)) > 1e-6:
        return price
    rub = str(int(round(f)))
    if len(rub) < 2:
        return price
    pat = re.compile(rf"(?<!\d){re.escape(rub)}\s*(?:[.,'’`´\-]?\s*)?(\d{{2}})(?!\d)")
    for src in text_sources:
        s = str(src or "")
        m = pat.search(s)
        if not m:
            continue
        kop = m.group(1)
        try:
            kval = int(kop)
        except Exception:
            continue
        if 0 <= kval <= 99:
            return f"{int(rub)}.{kop}"
    return price


def _price_tokens_from_text(text: str, *, prefer_integer_rubles: bool = False) -> List[str]:
    s = str(text or "")
    out: List[str] = []

    # 129 99 / 144 99 / 1199 99
    for m in re.finditer(r"(?<!\d)(\d{2,5})\s+(\d{2})(?!\d)", s):
        rub, kop = m.group(1), m.group(2)
        try:
            out.append(f"{int(rub)}.{kop}")
        except Exception:
            pass

    # 129.99 / 129,99
    for m in re.finditer(r"(?<!\d)(\d{1,5})[,.](\d{2})(?!\d)", s):
        try:
            out.append(f"{int(m.group(1))}.{m.group(2)}")
        except Exception:
            pass

    # compact 14499 / 119999
    for m in re.finditer(r"(?<!\d)(\d{4,7})(?!\d)", s):
        raw = m.group(1)
        p = normalize_price_value(raw)
        if p:
            out.append(str(p))

    if prefer_integer_rubles:
        for m in re.finditer(r"(?<!\d)(\d{2,5})(?!\d)", s):
            try:
                out.append(f"{float(int(m.group(1))):.2f}")
            except Exception:
                pass

    dedup: List[str] = []
    seen = set()
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        # avoid absurd output prices from barcodes or parsing garbage
        val = _price_f(p)
        if 0.01 <= val <= 99999.99:
            dedup.append(p)
    return dedup


def _extract_default_price_from_context(text_sources: Sequence[str], *, reference_price: str) -> str:
    ref = _price_f(reference_price)
    key_re = re.compile(r"(?:без\s*карт\w*|безкарты|обычн\w*|старая\s*цена|до\s*скидк\w*|цена\s*без)", re.I)
    candidates: List[Tuple[float, str]] = []
    for src in text_sources:
        s = str(src or "")
        for km in key_re.finditer(s):
            window = s[max(0, km.start() - 35): min(len(s), km.end() + 80)]
            for p in _price_tokens_from_text(window, prefer_integer_rubles=True):
                val = _price_f(p)
                if ref > 0 and val <= ref * 1.01:
                    continue
                if ref > 0 and val > ref * 4.5:
                    continue
                candidates.append((val, p))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: (abs(x[0] - ref * 1.25) if ref > 0 else x[0], -x[0]))
    return candidates[0][1]


def _extract_price_from_discount_badge(text_sources: Sequence[str]) -> str:
    """Extract promo price accidentally glued to a discount badge.

    Examples: ``28%1199`` -> ``1199.00``; ``-17%399`` -> ``399.00``.
    """
    for src in text_sources:
        s = str(src or "")
        for m in re.finditer(r"[-−]?\s*\d{1,2}(?:[,.]\d{1,2})?\s*%\s*([^\dА-Яа-яA-Za-z]{0,4})(\d{3,5})(?:\s+(\d{2}))?", s):
            rub = m.group(2)
            kop = m.group(3)
            try:
                if kop:
                    return f"{int(rub)}.{kop[:2]}"
                return f"{float(int(rub)):.2f}"
            except Exception:
                continue
    return ""


def _is_suspicious_default_price(default_price: str, reference_price: str, *, promo: bool) -> bool:
    d = _price_f(default_price)
    r = _price_f(reference_price)
    if d <= 0:
        return True
    if d > 99999:
        return True
    if not promo or r <= 0:
        return False
    # On promo labels default/old price should usually be above the card/action
    # price.  Values like 4.01, 5.00, 116.84 from barcodes are invalid here.
    if d < max(10.0, r * 0.70):
        return True
    return False

def _collect_decoded_codes(track: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    out: List[Mapping[str, Any]] = []
    for obs in track.get("observations") or []:
        if not isinstance(obs, Mapping):
            continue
        raw = obs.get("raw_summary") if isinstance(obs.get("raw_summary"), Mapping) else {}
        for code in raw.get("codes") or []:
            if isinstance(code, Mapping) and code.get("decoded") and str(code.get("payload") or ""):
                out.append(code)
    fused = track.get("fused_image") if isinstance(track.get("fused_image"), Mapping) else {}
    fb = fused.get("code_fallback") if isinstance(fused.get("code_fallback"), Mapping) else {}
    for code in fb.get("codes") or []:
        if isinstance(code, Mapping) and code.get("decoded") and str(code.get("payload") or ""):
            out.append(code)
    # Deduplicate by payload.
    seen = set()
    deduped: List[Mapping[str, Any]] = []
    for code in out:
        payload = str(code.get("payload") or "")
        if payload in seen:
            continue
        seen.add(payload)
        deduped.append(code)
    return deduped


def _select_qr_payload(codes: Sequence[Mapping[str, Any]]) -> str:
    for code in codes:
        fmt = str(code.get("fmt") or code.get("kind") or code.get("decoder") or "").lower()
        payload = str(code.get("payload") or "")
        if payload and ("qr" in fmt or "datamatrix" in fmt or any(a in _norm_key(payload) for a in ("price1", "p1", "wholesale", "action"))):
            return payload
    return str(codes[0].get("payload") or "") if codes else ""


def _first_barcode(codes: Sequence[Mapping[str, Any]], final: Mapping[str, Any]) -> str:
    for key in ("barcode", "ean", "gtin"):
        v = str(final.get(key) or "").strip()
        if v:
            return v
    for code in codes:
        payload = str(code.get("payload") or "").strip()
        fmt = str(code.get("fmt") or code.get("kind") or "").lower()
        if payload and ("ean" in fmt or "barcode" in fmt or re.fullmatch(r"\d{8,14}", payload)):
            return payload
    qr = parse_qr_payload(_select_qr_payload(codes))
    return _lookup_alias(qr, QR_FIELD_ALIASES["qr_code_barcode"]) or ""


def _filename_for_track(track: Mapping[str, Any], best: Mapping[str, Any]) -> str:
    seq = str(track.get("sequence_name") or "").strip()
    if seq:
        return seq
    image_path = str(best.get("image_path") or best.get("saved_crop") or "").split("#", 1)[0]
    if image_path:
        return Path(image_path).name
    return str(track.get("track_key") or "")


def _bbox_from_best(best: Mapping[str, Any]) -> Tuple[Any, Any, Any, Any]:
    bbox = best.get("bbox") if isinstance(best.get("bbox"), (list, tuple)) else []
    if len(bbox) >= 4:
        return tuple(int(float(x)) for x in bbox[:4])  # type: ignore[return-value]
    return "", "", "", ""


def _frame_timestamp(best: Mapping[str, Any], cfg: Mapping[str, Any], *, video_fps: Optional[float]) -> Any:
    # Prefer explicit timestamp fields when runner/detector provided them.
    for key in ("frame_timestamp", "timestamp_ms", "frame_timestamp_ms", "time_ms"):
        if best.get(key) not in (None, ""):
            return best.get(key)
    raw = " ".join(str(best.get(k) or "") for k in ("image_path", "saved_crop"))
    m = re.search(r"(?:timestamp|ts|time|ms)[_=-]?(\d{2,12})", raw, flags=re.I)
    if m:
        return int(m.group(1))
    frame = best.get("frame_index")
    if frame is not None and video_fps and video_fps > 0:
        try:
            return int(round(1000.0 * int(frame) / float(video_fps)))
        except Exception:
            pass
    if bool(cfg.get("fallback_frame_index_as_timestamp", False)) and frame is not None:
        try:
            return int(frame)
        except Exception:
            return frame
    return ""


def _get_template_name(track: Mapping[str, Any]) -> str:
    best = track.get("best_observation") if isinstance(track.get("best_observation"), Mapping) else {}
    return str(best.get("template_name") or best.get("template") or "")


def _sku_from_final(final: Mapping[str, Any]) -> str:
    for key in ("id_sku", "sku", "item_id", "product_id"):
        v = str(final.get(key) or "").strip()
        if v:
            return v
    pm = final.get("product_match") if isinstance(final.get("product_match"), Mapping) else {}
    return str(pm.get("item_id") or pm.get("sku") or "").strip()


def _additional_info(final: Mapping[str, Any], track: Mapping[str, Any], *, absent_value: str) -> str:
    for key in ("additional_info", "info", "note", "comment"):
        v = str(final.get(key) or "").strip()
        if v:
            return v
    if final.get("stock_status") == "out_of_stock":
        return "товар закончился"
    return absent_value


def _discount_amount_value(
    final: Mapping[str, Any],
    track: Mapping[str, Any],
    best: Mapping[str, Any],
    *,
    absent_value: str,
) -> str:
    """Return a clean discount value for the task schema.

    OCR often merges the discount badge with the large price, for example
    ``28%1199`` or ``-17%399``.  The required field is the discount amount,
    not the following price, so prefer a normalized percent token.
    """
    sources: List[Any] = [
        final.get("discount_amount"),
        final.get("discount_percent_raw"),
        final.get("discount_percent"),
        final.get("special_symbols"),
    ]

    # Best-observation sources are useful when the track vote dropped the field.
    best_final = best.get("final") if isinstance(best.get("final"), Mapping) else {}
    sources.extend([
        best_final.get("discount_amount"),
        best_final.get("discount_percent_raw"),
        best_final.get("discount_percent"),
    ])
    raw = best.get("raw_summary") if isinstance(best.get("raw_summary"), Mapping) else {}
    parsed = raw.get("parsed") if isinstance(raw.get("parsed"), Mapping) else {}
    ocr = raw.get("ocr") if isinstance(raw.get("ocr"), Mapping) else {}
    sources.extend([
        parsed.get("discount_percent_raw"),
        parsed.get("discount_amount"),
        ocr.get("all_text_joined"),
    ])

    for obs in track.get("observations") or []:
        if not isinstance(obs, Mapping):
            continue
        f = obs.get("final") if isinstance(obs.get("final"), Mapping) else {}
        sources.extend([f.get("discount_amount"), f.get("discount_percent_raw"), f.get("discount_percent")])
        rs = obs.get("raw_summary") if isinstance(obs.get("raw_summary"), Mapping) else {}
        po = rs.get("parsed") if isinstance(rs.get("parsed"), Mapping) else {}
        oo = rs.get("ocr") if isinstance(rs.get("ocr"), Mapping) else {}
        sources.extend([po.get("discount_percent_raw"), po.get("discount_amount"), oo.get("all_text_joined")])

    for value in sources:
        pct = normalize_percent_raw(value)
        if pct:
            return pct
    for value in sources:
        s = str(value or "").strip()
        if s:
            return s
    return absent_value



def _color_value(
    final: Mapping[str, Any],
    template_name: str,
    *,
    track: Mapping[str, Any],
    best: Mapping[str, Any],
    cfg: Mapping[str, Any],
    absent_value: str,
) -> str:
    """Resolve visible tag color with yellow/progressive override.

    The template classifier can label progressive yellow shelf tags as
    ``red_promo`` because both are promotional.  For task output, color must be
    the visible label color, so yellow/progressive evidence has priority over a
    generic red promo field.
    """
    visual_cfg = cfg.get("visible_color", {}) if isinstance(cfg.get("visible_color"), Mapping) else {}
    use_visual = bool(visual_cfg.get("infer_from_best_image", True))
    prefer_visual_yellow = bool(visual_cfg.get("prefer_visual_yellow", True))

    tmpl = template_name.lower()
    warnings = " ".join(str(w) for w in (track.get("warnings") or [])).lower()
    final_text = " ".join(str(final.get(k) or "") for k in ("color", "template", "layout_type", "special_symbols", "promo_header")).lower()
    image_color = _infer_color_from_best_image(best, track, visual_cfg) if use_visual else ""
    explicit = str(final.get("color") or "").strip()

    yellow_evidence = any(x in tmpl for x in ("yellow", "progressive")) or any(x in warnings for x in ("yellow", "progressive")) or "желт" in final_text
    red_evidence = any(x in tmpl for x in ("red", "promo")) or "красн" in final_text or "promo" in final_text

    if image_color and ("желт" in image_color or prefer_visual_yellow):
        if "желт" in image_color:
            return "желтый"
    if yellow_evidence:
        return "желтый"
    if image_color:
        return image_color
    if explicit:
        # Do not let generic red/promo survive when other evidence says yellow;
        # yellow case returned above.
        return explicit
    if red_evidence:
        return "красный/промо"
    if tmpl:
        return tmpl
    return absent_value

def _special_symbols(final: Mapping[str, Any], track: Mapping[str, Any], *, absent_value: str) -> str:
    values: List[str] = []
    for key in ("special_symbols", "special_symbol", "layout_type"):
        v = str(final.get(key) or "").strip()
        if v:
            values.append(v)
    if final.get("stock_status") == "out_of_stock":
        values.append("товар закончился")
    disc = normalize_percent_raw(final.get("discount_percent_raw") or final.get("discount_amount") or "")
    if disc:
        values.append(disc)
    warnings = [str(w) for w in (track.get("warnings") or [])]
    if any("progressive" in w.lower() for w in warnings):
        values.append("progressive")
    return ", ".join(list(dict.fromkeys(values))) if values else absent_value


def _infer_color_from_best_image(best: Mapping[str, Any], track: Mapping[str, Any], cfg: Mapping[str, Any]) -> str:
    paths: List[str] = []
    for src in (best, track):
        for key in ("image_path", "saved_crop", "best_image", "best_debug_image"):
            v = str(src.get(key) or "").strip() if isinstance(src, Mapping) else ""
            if v:
                paths.append(v)
    # Also check best observation nested in the track.
    b2 = track.get("best_observation") if isinstance(track.get("best_observation"), Mapping) else {}
    for key in ("image_path", "saved_crop"):
        v = str(b2.get(key) or "").strip()
        if v:
            paths.append(v)

    for raw_path in paths:
        color = _infer_color_from_image_path(raw_path, cfg)
        if color:
            return color
    return ""


@lru_cache(maxsize=4096)
def _infer_color_from_image_path_cached(path_str: str, yellow_min_ratio: float, red_min_ratio: float, yellow_over_red: float) -> str:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return ""
    path = Path(path_str.split("#", 1)[0])
    if not path.exists():
        return ""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return ""

    h, w = img.shape[:2]
    # Most Lenta promo/progressive color is in the lower half.  Ignore the top
    # white product-name area and shelf background where possible.
    y0 = int(round(h * 0.42)) if h >= 80 else 0
    roi = img[y0:h, :]
    if roi.size == 0:
        roi = img
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hh = hsv[:, :, 0].astype("int32")
    ss = hsv[:, :, 1]
    vv = hsv[:, :, 2]
    valid = (ss > 45) & (vv > 70)
    denom = float(valid.sum())
    if denom < max(20.0, 0.02 * float(roi.shape[0] * roi.shape[1])):
        return ""
    # OpenCV hue is 0..179.  Yellow tags have a stronger 20..45 component;
    # orange/red Lenta promo labels are usually 0..19 plus occasional wrap-around.
    yellow = valid & (hh >= 20) & (hh <= 45)
    red_orange = valid & (((hh >= 0) & (hh <= 19)) | (hh >= 165))
    yellow_ratio = float(yellow.sum()) / denom
    red_ratio = float(red_orange.sum()) / denom
    if yellow_ratio >= yellow_min_ratio and yellow_ratio >= red_ratio * yellow_over_red:
        return "желтый"
    if red_ratio >= red_min_ratio:
        return "красный/промо"
    return ""


def _infer_color_from_image_path(raw_path: str, cfg: Mapping[str, Any]) -> str:
    yellow_min_ratio = float(cfg.get("yellow_min_ratio", 0.24) or 0.24)
    red_min_ratio = float(cfg.get("red_min_ratio", 0.22) or 0.22)
    yellow_over_red = float(cfg.get("yellow_over_red", 1.08) or 1.08)
    return _infer_color_from_image_path_cached(str(raw_path), yellow_min_ratio, red_min_ratio, yellow_over_red)


def _lookup_alias(data: Mapping[str, str], aliases: Iterable[str]) -> str:
    for alias in aliases:
        key = _norm_key(alias)
        if key in data and str(data[key]).strip() != "":
            return str(data[key]).strip()
    return ""


def _norm_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key or "").strip().lower())


def _normalize_field_value(field: str, value: Any) -> str:
    if field in PRICE_FIELDS:
        norm = normalize_price_value(value)
        return norm if norm is not None else str(value or "").strip()
    return str(value or "").strip()


def _first_nonempty(*values: Any) -> str:
    for v in values:
        if v not in (None, ""):
            s = str(v).strip()
            if s:
                return s
    return ""


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _to_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default
