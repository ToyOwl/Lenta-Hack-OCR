"""Deterministic OCR correction from a structured product CSV.

This module is intentionally independent from the LLM corrector.  It uses only
OCR evidence, parsed prices and a structured catalog CSV such as the Lenta
``goods.csv`` file with columns like ``item_id``, ``name``, ``unit``, ``price``
and ``price_regular``.

Typical use case:
- OCR sees a noisy product name or only a partially readable tag.
- The tag price is still readable across one or several frames.
- We match OCR text + price against the catalog and replace only fields whose
  catalog evidence is strong enough.
"""

from __future__ import annotations

import csv
import math
import re
import time
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .product_card import baseline_final_from_result, enrich_final_fields, detect_stock_status_from_result
from .product_card import normalize_price_value as product_normalize_price_value

_EMPTY_PRICE_KEYS = {"", "none", "null", "nan", "-", "—"}
_VISUAL_FIX = str.maketrans({
    "О": "0", "O": "0", "o": "0", "о": "0",
    "I": "1", "l": "1", "|": "1", "!": "1",
    "S": "5", "s": "5", ",": ".",
})
_SERVICE_WORDS = {
    "цена", "ценник", "руб", "рубль", "рублей", "коп", "копейка", "шт", "кг",
    "г", "л", "мл", "карта", "картой", "лентой", "лента", "акция", "скидка",
    "выгодно", "код", "qr", "ean", "barcode", "штрихкод", "за", "ед", "изм",
}

# Visual OCR substitutions for product-text matching.  Do not reuse the price
# substitution table here: mapping Latin ``l`` to digit ``1`` is useful for
# prices but harmful for brands like LIGHT.  The table below is intentionally
# conservative and is used only by catalog matching / OCR text repair.

_TEXT_VISUAL_LATIN_TO_CYR = str.maketrans({
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
    "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
    "a": "а", "c": "с", "e": "е", "h": "н", "k": "к", "m": "м",
    "o": "о", "p": "р", "x": "х", "y": "у",
})

_TEXT_OCR_REPLACEMENTS: List[Tuple[re.Pattern[str], str]] = [
    # Common mixed Latin/Cyrillic OCR errors on Russian price tags.
    (re.compile(r"\b[нh][аa][пn][иuі1l]?[тt][оo0][кk]\b", re.I), "напиток"),
    (re.compile(r"\b[нh][аa][пn][кk][тt][оo0][кk]\b", re.I), "напиток"),
    (re.compile(r"\bб[еeсcз3]{1,3}алк[оo0]г[оo0]льн[ыьi1l][йиu]\w*\b", re.I), "безалкогольный"),
    (re.compile(r"\blght\b", re.I), "light"),
    (re.compile(r"\bligth\b", re.I), "light"),
    (re.compile(r"\bро[сc][сc][иu][яa]\b", re.I), "россия"),
    (re.compile(r"\bp[оo0]cch[аa]\b", re.I), "россия"),
    (re.compile(r"\bp[оo0]cc[иu][яa]\b", re.I), "россия"),
    # Wine tags: frequent short OCR fragments and lost first letters.
    (re.compile(r"\b[оo0]настырск\w*\b", re.I), "монастырская"),
    (re.compile(r"\bм[оo0]настырск\w*\b", re.I), "монастырская"),
    (re.compile(r"\bрапез[а-яa-z]*\b", re.I), "трапеза"),
    (re.compile(r"\bтр[аa]пез[а-яa-z]*\b", re.I), "трапеза"),
    (re.compile(r"\bкр\.?(?=\s|$)", re.I), "красное"),
    (re.compile(r"\bбел\.?(?=\s|$)", re.I), "белое"),
    (re.compile(r"\bсух\.?(?=\s|$)", re.I), "сухое"),
    (re.compile(r"\bп\s*[/\\]?\s*с\b", re.I), "полусладкое"),
    (re.compile(r"\bп\s*[/\\]?\s*сл\.?(?=\s|$)", re.I), "полусладкое"),
]


@dataclass
class StructuredCatalogCandidate:
    item_id: str = ""
    category_id: str = ""
    name: str = ""
    brand: str = ""
    category_name: str = ""
    unit: str = ""
    quantity: Optional[float] = None
    price: Optional[float] = None
    price_regular: Optional[float] = None
    cost: Optional[float] = None
    cost_regular: Optional[float] = None
    is_promo: Optional[bool] = None
    is_loyalty: Optional[bool] = None
    text_score: float = 0.0
    price_score: float = 0.0
    score: float = 0.0
    match_status: str = "candidate"
    matched_price_column: str = ""
    raw_row: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_raw_row: bool = False) -> Dict[str, Any]:
        d = asdict(self)
        if not include_raw_row:
            d.pop("raw_row", None)
        return d


class StructuredCatalogMatcher:
    """Small in-process matcher for product catalog CSV files.

    The matcher keeps the implementation simple and transparent.  It avoids
    external dependencies so it can run on Rockchip boards together with the OCR
    runtime.  For very large catalogs the first optimization point is replacing
    the linear scan with a token inverted index; for the current Lenta-scale CSV
    a capped linear scan is acceptable for offline/batch correction.
    """

    def __init__(
        self,
        csv_path: str | Path | None = None,
        name_column: str = "name",
        item_id_column: str = "item_id",
        category_id_column: str = "category_id",
        brand_column: str = "brand",
        category_column: str = "category_name",
        unit_column: str = "unit",
        quantity_column: str = "quantity",
        price_columns: Sequence[str] = ("price", "price_regular", "cost", "cost_regular"),
        min_text_score: float = 0.26,
        min_accept_score: float = 0.58,
        max_rows: int = 250_000,
        fuzzy_token_match: bool = True,
        min_fuzzy_token_score: float = 0.74,
        max_candidate_rows: int = 20_000,
    ) -> None:
        self.csv_path = Path(csv_path).expanduser() if csv_path else None
        self.name_column = str(name_column or "name")
        self.item_id_column = str(item_id_column or "item_id")
        self.category_id_column = str(category_id_column or "category_id")
        self.brand_column = str(brand_column or "brand")
        self.category_column = str(category_column or "category_name")
        self.unit_column = str(unit_column or "unit")
        self.quantity_column = str(quantity_column or "quantity")
        self.price_columns = tuple(str(c) for c in price_columns if str(c or "").strip())
        self.min_text_score = float(min_text_score)
        self.min_accept_score = float(min_accept_score)
        self.max_rows = int(max_rows)
        self.fuzzy_token_match = bool(fuzzy_token_match)
        self.min_fuzzy_token_score = float(min_fuzzy_token_score)
        self.max_candidate_rows = int(max_candidate_rows)
        self.rows: List[Dict[str, Any]] = []
        self._norm_names: List[str] = []
        self._token_sets: List[set[str]] = []
        self._token_to_indices: Dict[str, List[int]] = {}
        self._price_int_to_indices: Dict[int, List[int]] = {}
        self.load_error = ""
        if self.csv_path:
            self.load(self.csv_path)

    @property
    def enabled(self) -> bool:
        return bool(self.rows)

    def load(self, csv_path: str | Path) -> None:
        p = Path(csv_path).expanduser()
        self.load_error = ""
        if not p.exists():
            self.load_error = f"catalog_not_found:{p}"
            return

        rows: List[Dict[str, Any]] = []
        try:
            with open(p, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i >= self.max_rows:
                        break
                    name = str(row.get(self.name_column, "") or "").strip()
                    if not name:
                        continue
                    rows.append(dict(row))
        except UnicodeDecodeError:
            with open(p, "r", encoding="cp1251", errors="replace", newline="") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i >= self.max_rows:
                        break
                    name = str(row.get(self.name_column, "") or "").strip()
                    if not name:
                        continue
                    rows.append(dict(row))
        except Exception as e:
            self.load_error = f"catalog_load_failed:{type(e).__name__}:{e}"
            rows = []

        self.rows = rows
        self._norm_names = [_normalize_text_for_match(str(r.get(self.name_column, "") or "")) for r in self.rows]
        self._token_sets = [set(_content_tokens(n)) for n in self._norm_names]
        self._build_indexes()

    def _build_indexes(self) -> None:
        token_to_indices: Dict[str, List[int]] = {}
        price_int_to_indices: Dict[int, List[int]] = {}
        for idx, (row, tokens) in enumerate(zip(self.rows, self._token_sets)):
            for tok in tokens:
                if len(tok) >= 3:
                    token_to_indices.setdefault(tok, []).append(idx)
            for col in self.price_columns:
                val = _to_float_or_none(row.get(col))
                if val is None or val <= 0:
                    continue
                for bucket in {int(round(val)), int(math.floor(val)), int(math.ceil(val))}:
                    price_int_to_indices.setdefault(bucket, []).append(idx)
        self._token_to_indices = token_to_indices
        self._price_int_to_indices = price_int_to_indices

    def _candidate_indices(self, q_tokens: Sequence[str], observed: Sequence[float]) -> List[int]:
        if not self.rows:
            return []
        scores: Dict[int, float] = {}
        for tok in dict.fromkeys(q_tokens):
            if len(tok) < 3:
                continue
            for idx in self._token_to_indices.get(tok, [])[: self.max_candidate_rows]:
                scores[idx] = scores.get(idx, 0.0) + (2.0 if len(tok) >= 5 else 1.0)
        for obs in observed:
            if not _is_positive_number(obs):
                continue
            center = int(round(float(obs)))
            for bucket in range(center - 2, center + 3):
                for idx in self._price_int_to_indices.get(bucket, [])[: self.max_candidate_rows]:
                    scores[idx] = scores.get(idx, 0.0) + 1.4
        if not scores:
            # Fall back to a bounded scan.  This keeps behavior deterministic for
            # very noisy OCR while preventing quadratic fuzzy matching on large catalogs.
            return list(range(min(len(self.rows), self.max_candidate_rows)))
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [idx for idx, _ in ordered[: self.max_candidate_rows]]

    def find_candidates(
        self,
        query_text: str,
        prices: Sequence[float] = (),
        top_k: int = 8,
        allow_price_only: bool = True,
        max_price_only_candidates: int = 5,
    ) -> List[StructuredCatalogCandidate]:
        if not self.rows:
            return []

        q_norm = _normalize_text_for_match(query_text)
        q_tokens = _content_tokens(q_norm)
        observed = [float(p) for p in prices if _is_positive_number(p)]
        if not q_tokens and not observed:
            return []
        scored: List[Tuple[float, StructuredCatalogCandidate]] = []
        price_only: List[Tuple[float, StructuredCatalogCandidate]] = []

        for idx in self._candidate_indices(q_tokens, observed):
            row = self.rows[idx]
            n_norm = self._norm_names[idx]
            n_tokens = self._token_sets[idx]
            text_score = _text_similarity(
                q_norm,
                q_tokens,
                n_norm,
                n_tokens,
                fuzzy_token_match=self.fuzzy_token_match,
                min_fuzzy_token_score=self.min_fuzzy_token_score,
            )
            price_score, price_col = self._price_similarity(row, observed)

            if text_score >= self.min_text_score:
                combined = 0.78 * text_score + 0.22 * price_score
                cand = self._row_to_candidate(row)
                cand.text_score = float(text_score)
                cand.price_score = float(price_score)
                cand.score = float(combined)
                cand.matched_price_column = price_col
                cand.match_status = "candidate" if cand.score >= self.min_accept_score else "weak_candidate"
                scored.append((combined, cand))
            elif allow_price_only and price_score >= 0.98:
                cand = self._row_to_candidate(row)
                cand.text_score = float(text_score)
                cand.price_score = float(price_score)
                cand.score = float(0.22 + 0.60 * price_score + 0.18 * text_score)
                cand.matched_price_column = price_col
                cand.match_status = "price_only_candidate"
                price_only.append((cand.score, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        price_only.sort(key=lambda x: x[0], reverse=True)
        out = [c for _, c in scored[: max(0, int(top_k))]]
        if allow_price_only and len(out) < int(top_k):
            out.extend([c for _, c in price_only[: max(0, min(max_price_only_candidates, int(top_k) - len(out)))]] )
        return out[: max(0, int(top_k))]

    def _row_to_candidate(self, row: Mapping[str, Any]) -> StructuredCatalogCandidate:
        return StructuredCatalogCandidate(
            item_id=str(row.get(self.item_id_column, row.get("item_id", "")) or ""),
            category_id=str(row.get(self.category_id_column, row.get("category_id", "")) or ""),
            name=str(row.get(self.name_column, "") or ""),
            brand=str(row.get(self.brand_column, row.get("brand", "")) or ""),
            category_name=str(row.get(self.category_column, row.get("category_name", "")) or ""),
            unit=str(row.get(self.unit_column, row.get("unit", "")) or ""),
            quantity=_to_float_or_none(row.get(self.quantity_column, row.get("quantity"))),
            price=_to_float_or_none(row.get("price")),
            price_regular=_to_float_or_none(row.get("price_regular")),
            cost=_to_float_or_none(row.get("cost")),
            cost_regular=_to_float_or_none(row.get("cost_regular")),
            is_promo=_to_bool_or_none(row.get("is_promo")),
            is_loyalty=_to_bool_or_none(row.get("is_loyalty")),
            raw_row=dict(row),
        )

    def _price_similarity(self, row: Mapping[str, Any], observed: Sequence[float]) -> Tuple[float, str]:
        if not observed:
            return 0.0, ""
        best = 0.0
        best_col = ""
        for col in self.price_columns:
            val = _to_float_or_none(row.get(col))
            if val is None or val <= 0:
                continue
            for obs in observed:
                denom = max(1.0, abs(val), abs(obs))
                rel = abs(val - obs) / denom
                score = 0.0
                if rel <= 0.005:
                    score = 1.0
                elif rel <= 0.015:
                    score = 0.92
                elif rel <= 0.05:
                    score = 0.74
                elif rel <= 0.12:
                    score = 0.42
                if score > best:
                    best = score
                    best_col = col
        return best, best_col

    def debug_info(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "csv_path": str(self.csv_path) if self.csv_path else "",
            "rows": len(self.rows),
            "load_error": self.load_error,
            "name_column": self.name_column,
            "item_id_column": self.item_id_column,
            "price_columns": list(self.price_columns),
            "min_text_score": self.min_text_score,
            "min_accept_score": self.min_accept_score,
            "fuzzy_token_match": self.fuzzy_token_match,
            "min_fuzzy_token_score": self.min_fuzzy_token_score,
            "max_candidate_rows": self.max_candidate_rows,
        }


class StructuredCSVOCRCorrector:
    def __init__(
        self,
        enabled: bool = False,
        matcher: Optional[StructuredCatalogMatcher] = None,
        top_k: int = 8,
        min_accept_score: float = 0.58,
        allow_price_correction: bool = True,
        allow_price_only_match: bool = True,
        allow_price_only_autofill: bool = False,
        force_catalog_price_when_matched: bool = False,
        max_price_conflict_ratio: float = 0.12,
        max_price_only_candidates: int = 5,
        include_raw_row: bool = False,
        allow_close_family_match: bool = True,
        family_match_min_name_overlap: float = 0.72,
        family_match_min_price_score: float = 0.92,
        min_text_score_with_price: float = 0.42,
        min_price_score_for_soft_accept: float = 0.90,
        reject_weak_query_text: bool = True,
        min_query_content_tokens: int = 2,
        min_query_alpha_chars: int = 8,
        min_product_text_score_for_autofill: float = 0.50,
        retain_candidate_match: bool = True,
        strong_text_accept_score: float = 0.48,
        debug: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.matcher = matcher or StructuredCatalogMatcher()
        self.top_k = int(top_k)
        self.min_accept_score = float(min_accept_score)
        self.allow_price_correction = bool(allow_price_correction)
        self.allow_price_only_match = bool(allow_price_only_match)
        self.allow_price_only_autofill = bool(allow_price_only_autofill)
        self.force_catalog_price_when_matched = bool(force_catalog_price_when_matched)
        self.max_price_conflict_ratio = float(max_price_conflict_ratio)
        self.max_price_only_candidates = int(max_price_only_candidates)
        self.include_raw_row = bool(include_raw_row)
        self.allow_close_family_match = bool(allow_close_family_match)
        self.family_match_min_name_overlap = float(family_match_min_name_overlap)
        self.family_match_min_price_score = float(family_match_min_price_score)
        self.min_text_score_with_price = float(min_text_score_with_price)
        self.min_price_score_for_soft_accept = float(min_price_score_for_soft_accept)
        self.reject_weak_query_text = bool(reject_weak_query_text)
        self.min_query_content_tokens = int(min_query_content_tokens)
        self.min_query_alpha_chars = int(min_query_alpha_chars)
        self.min_product_text_score_for_autofill = float(min_product_text_score_for_autofill)
        self.retain_candidate_match = bool(retain_candidate_match)
        self.strong_text_accept_score = float(strong_text_accept_score)
        self.debug = bool(debug)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "StructuredCSVOCRCorrector":
        ccfg = cfg.get("csv_corrector", {}) if isinstance(cfg, Mapping) else {}
        catalog_cfg = ccfg.get("catalog", {}) if isinstance(ccfg, Mapping) else {}
        path = str(catalog_cfg.get("path", ccfg.get("path", "")) or "")
        matcher = StructuredCatalogMatcher(
            csv_path=path or None,
            name_column=str(catalog_cfg.get("name_column", "name") or "name"),
            item_id_column=str(catalog_cfg.get("item_id_column", "item_id") or "item_id"),
            category_id_column=str(catalog_cfg.get("category_id_column", "category_id") or "category_id"),
            brand_column=str(catalog_cfg.get("brand_column", "brand") or "brand"),
            category_column=str(catalog_cfg.get("category_column", "category_name") or "category_name"),
            unit_column=str(catalog_cfg.get("unit_column", "unit") or "unit"),
            quantity_column=str(catalog_cfg.get("quantity_column", "quantity") or "quantity"),
            price_columns=catalog_cfg.get("price_columns", ["price", "price_regular", "cost", "cost_regular"]),
            min_text_score=float(catalog_cfg.get("min_text_score", ccfg.get("min_text_score", 0.26))),
            min_accept_score=float(catalog_cfg.get("min_accept_score", ccfg.get("min_accept_score", 0.58))),
            max_rows=int(catalog_cfg.get("max_rows", ccfg.get("max_rows", 250000))),
            fuzzy_token_match=bool(catalog_cfg.get("fuzzy_token_match", ccfg.get("fuzzy_token_match", True))),
            min_fuzzy_token_score=float(catalog_cfg.get("min_fuzzy_token_score", ccfg.get("min_fuzzy_token_score", 0.74))),
            max_candidate_rows=int(catalog_cfg.get("max_candidate_rows", ccfg.get("max_candidate_rows", 20000))),
        )
        return cls(
            enabled=bool(ccfg.get("enabled", False)),
            matcher=matcher,
            top_k=int(ccfg.get("top_k", catalog_cfg.get("top_k", 8))),
            min_accept_score=float(ccfg.get("min_accept_score", catalog_cfg.get("min_accept_score", 0.58))),
            allow_price_correction=bool(ccfg.get("allow_price_correction", True)),
            allow_price_only_match=bool(ccfg.get("allow_price_only_match", True)),
            allow_price_only_autofill=bool(ccfg.get("allow_price_only_autofill", False)),
            force_catalog_price_when_matched=bool(ccfg.get("force_catalog_price_when_matched", False)),
            max_price_conflict_ratio=float(ccfg.get("max_price_conflict_ratio", 0.12)),
            max_price_only_candidates=int(ccfg.get("max_price_only_candidates", 5)),
            include_raw_row=bool(ccfg.get("include_raw_row", False)),
            allow_close_family_match=bool(ccfg.get("allow_close_family_match", True)),
            family_match_min_name_overlap=float(ccfg.get("family_match_min_name_overlap", 0.72)),
            family_match_min_price_score=float(ccfg.get("family_match_min_price_score", 0.92)),
            min_text_score_with_price=float(ccfg.get("min_text_score_with_price", 0.42)),
            min_price_score_for_soft_accept=float(ccfg.get("min_price_score_for_soft_accept", 0.90)),
            reject_weak_query_text=bool(ccfg.get("reject_weak_query_text", True)),
            min_query_content_tokens=int(ccfg.get("min_query_content_tokens", 2)),
            min_query_alpha_chars=int(ccfg.get("min_query_alpha_chars", 8)),
            min_product_text_score_for_autofill=float(ccfg.get("min_product_text_score_for_autofill", 0.50)),
            retain_candidate_match=bool(ccfg.get("retain_candidate_match", True)),
            strong_text_accept_score=float(ccfg.get("strong_text_accept_score", 0.48)),
            debug=bool(ccfg.get("debug", False)),
        )

    def correct_result(self, result: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        t0 = time.perf_counter()
        final = dict(result.get("final") or baseline_final_from_result(result))
        query_text = _catalog_query_text(result)
        observed_prices = _collect_observed_prices(result, final)
        stockout_hint = detect_stock_status_from_result(result)
        query_guard = _catalog_query_guard(
            query_text,
            min_content_tokens=self.min_query_content_tokens,
            min_alpha_chars=self.min_query_alpha_chars,
        )
        query_rejected = bool(self.reject_weak_query_text and query_guard.get("reject"))
        if stockout_hint.get("stock_status") == "out_of_stock":
            query_rejected = True
            query_guard = dict(query_guard)
            query_guard["reject"] = True
            query_guard["reject_reason"] = "stockout_service_tag"
            query_guard["stockout_hint"] = stockout_hint
        candidates = [] if query_rejected else (self.matcher.find_candidates(
            query_text,
            prices=observed_prices,
            top_k=self.top_k,
            allow_price_only=self.allow_price_only_match,
            max_price_only_candidates=self.max_price_only_candidates,
        ) if self.matcher.enabled else [])

        warnings: List[str] = []
        if query_rejected:
            if query_guard.get("reject_reason") == "stockout_service_tag":
                warnings.append("catalog_query_blocked_for_stockout_service_tag")
            else:
                warnings.append("catalog_query_too_weak_for_product_autofill")
        status = "disabled"
        best: Optional[StructuredCatalogCandidate] = candidates[0] if candidates else None
        accepted = False
        ambiguous = False
        if not self.matcher.enabled:
            status = "catalog_unavailable"
            if self.matcher.load_error:
                warnings.append(self.matcher.load_error)
        elif query_rejected:
            status = "stockout_service_tag" if query_guard.get("reject_reason") == "stockout_service_tag" else "query_too_weak"
        elif best is None:
            status = "not_found"
        else:
            second = candidates[1] if len(candidates) > 1 else None
            close_family_accepted = False
            if second is not None and abs(float(best.score) - float(second.score)) < 0.035:
                ambiguous = True
                warnings.append("catalog_top_candidates_too_close")
                if self.allow_close_family_match and _same_catalog_family(
                    best,
                    second,
                    min_name_overlap=self.family_match_min_name_overlap,
                    min_price_score=self.family_match_min_price_score,
                ):
                    close_family_accepted = True
                    ambiguous = False
                    warnings.append("catalog_close_family_accepted")
            soft_price_text_accept = (
                float(best.price_score) >= self.min_price_score_for_soft_accept
                and float(best.text_score) >= self.min_text_score_with_price
            )
            # Goods CSV may lag the real shelf price.  For product identity we
            # therefore allow a text-dominant accept when OCR contains a strong
            # brand/product family anchor, but catalog price correction below
            # remains conflict-aware and will not overwrite the OCR price.
            strong_text_accept = (
                float(best.text_score) >= self.strong_text_accept_score
                and (close_family_accepted or float(best.text_score) >= self.strong_text_accept_score + 0.08)
            )
            accepted = (float(best.score) >= self.min_accept_score or soft_price_text_accept or strong_text_accept) and (not ambiguous or close_family_accepted or strong_text_accept)
            if best.match_status == "price_only_candidate" and not self.allow_price_only_autofill:
                accepted = False
                warnings.append("catalog_price_only_candidate_not_autofilled")
            if accepted and float(best.text_score) < self.min_product_text_score_for_autofill:
                accepted = False
                warnings.append("catalog_text_score_too_low_for_product_autofill")
            if accepted and soft_price_text_accept and float(best.score) < self.min_accept_score:
                warnings.append("catalog_soft_accept_by_price_and_text")
            if accepted and strong_text_accept and float(best.score) < self.min_accept_score:
                warnings.append("catalog_soft_accept_by_strong_text_family")
            if best.match_status == "price_only_candidate" and not self.allow_price_only_autofill:
                status = "price_only_candidates"
            else:
                status = "matched" if accepted else "ambiguous" if ambiguous else "weak_match"

        if accepted and best is not None:
            final = self._apply_candidate(final, result, best, warnings)
        else:
            if best is not None and self.retain_candidate_match and status in {"ambiguous", "weak_match", "price_only_candidates"}:
                final["product_match"] = _candidate_to_product_match(best, status=status)
                final.setdefault("catalog_product_prior", final["product_match"])
            else:
                final.setdefault("product_match", {
                    "status": "not_found" if self.matcher.enabled else "not_used",
                    "item_id": None,
                    "catalog_name": None,
                    "score": 0.0,
                })
            if status in {"ambiguous", "weak_match", "price_only_candidates"}:
                final["needs_review"] = True
                reasons = list(final.get("review_reasons") or []) if isinstance(final.get("review_reasons"), list) else []
                reason = "csv_catalog_ambiguous" if status == "ambiguous" else "csv_catalog_weak_match"
                if status == "price_only_candidates":
                    reason = "csv_catalog_price_only_candidates"
                if reason not in reasons:
                    reasons.append(reason)
                final["review_reasons"] = reasons

        final = enrich_final_fields(result, final)
        return {
            "enabled": True,
            "status": status,
            "backend": "structured_csv",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            "catalog": self.matcher.debug_info(),
            "query_text": query_text,
            "query_guard": query_guard,
            "observed_prices": [round(float(p), 2) for p in observed_prices],
            "candidates": [c.to_dict(include_raw_row=self.include_raw_row) for c in candidates],
            "selected": best.to_dict(include_raw_row=self.include_raw_row) if accepted and best is not None else None,
            "final": final,
            "warnings": warnings,
        }

    def _apply_candidate(
        self,
        final: Dict[str, Any],
        result: Mapping[str, Any],
        cand: StructuredCatalogCandidate,
        warnings: List[str],
    ) -> Dict[str, Any]:
        out = dict(final)
        out["product_name"] = cand.name or out.get("product_name")
        out["unit"] = cand.unit or out.get("unit")
        out["product_match"] = {
            "status": "matched",
            "item_id": cand.item_id or None,
            "catalog_name": cand.name or None,
            "score": round(float(cand.score), 4),
            "source": "structured_csv",
            "text_score": round(float(cand.text_score), 4),
            "price_score": round(float(cand.price_score), 4),
            "matched_price_column": cand.matched_price_column or None,
            "brand": cand.brand or None,
            "category_name": cand.category_name or None,
        }
        if cand.brand:
            out.setdefault("producer_or_brand", cand.brand)
        if cand.category_name:
            out.setdefault("category_name", cand.category_name)
        if cand.is_promo is not None:
            out["is_promo"] = cand.is_promo
        if cand.is_loyalty is not None:
            out["is_loyalty"] = cand.is_loyalty

        if self.allow_price_correction:
            self._apply_prices(out, result, cand, warnings)

        out["confidence"] = max(_to_float_or_none(out.get("confidence")) or 0.0, min(0.94, 0.72 + 0.22 * cand.score))
        return out

    def _apply_prices(self, out: Dict[str, Any], result: Mapping[str, Any], cand: StructuredCatalogCandidate, warnings: List[str]) -> None:
        observed = _collect_observed_prices(result, out)
        current_price = cand.price if cand.price is not None else cand.cost
        regular_price = cand.price_regular if cand.price_regular is not None else cand.cost_regular
        current_s = _fmt_price(current_price)
        regular_s = _fmt_price(regular_price)

        existing_main = _price_to_float(out.get("main_price"))
        conflict = False
        if current_price is not None and existing_main is not None:
            denom = max(1.0, abs(current_price), abs(existing_main))
            conflict = abs(current_price - existing_main) / denom > self.max_price_conflict_ratio

        if current_s:
            exact_or_missing = out.get("main_price") in (None, "") or _price_matches_any(current_price, observed, 0.015)
            if self.force_catalog_price_when_matched or exact_or_missing:
                out["main_price"] = current_s
                out["price_selection_status"] = "csv_catalog_corrected"
            else:
                # Catalog prices are often stale relative to shelf labels.  Do
                # not overwrite a stable OCR price merely because the product
                # identity matched.  Keep catalog prices as secondary evidence.
                out.setdefault("catalog_price", current_s)
                out["price_selection_status"] = "csv_catalog_price_conflict"
                out["needs_review"] = True
                reasons = list(out.get("review_reasons") or []) if isinstance(out.get("review_reasons"), list) else []
                if "csv_catalog_price_conflict" not in reasons:
                    reasons.append("csv_catalog_price_conflict")
                out["review_reasons"] = reasons
                warnings.append("catalog_price_differs_from_ocr_price")

        if regular_s and regular_s != out.get("main_price"):
            out.setdefault("old_price", regular_s)
            if bool(cand.is_loyalty):
                out.setdefault("no_card_price", regular_s)
                if current_s:
                    out.setdefault("card_price", current_s)

        if current_s and "price_candidates" in out and isinstance(out.get("price_candidates"), list):
            exists = any(product_normalize_price_value(p.get("value") if isinstance(p, Mapping) else p) == current_s for p in out["price_candidates"])
            if not exists:
                out["price_candidates"].append({"value": current_s, "source": "structured_csv", "confidence": round(0.72 + 0.22 * float(cand.score), 4)})
        elif current_s:
            out["price_candidates"] = [{"value": current_s, "source": "structured_csv", "confidence": round(0.72 + 0.22 * float(cand.score), 4)}]


def _candidate_to_product_match(cand: StructuredCatalogCandidate, *, status: str) -> Dict[str, Any]:
    """Represent the best catalog candidate even when it is only a prior.

    This is intentionally different from ``_apply_candidate``: it does not
    overwrite OCR product text.  It only exposes that DB/CSV was tried and what
    the strongest candidate was, so the track-level gate and debug plots can
    reason about it.
    """
    return {
        "status": str(status or cand.match_status or "candidate"),
        "item_id": cand.item_id or None,
        "catalog_name": cand.name or None,
        "score": round(float(cand.score), 4),
        "source": "structured_csv",
        "text_score": round(float(cand.text_score), 4),
        "price_score": round(float(cand.price_score), 4),
        "matched_price_column": cand.matched_price_column or None,
        "brand": cand.brand or None,
        "category_name": cand.category_name or None,
        "candidate_only": status not in {"matched", "accepted"},
    }


def _catalog_query_guard(query_text: str, *, min_content_tokens: int = 2, min_alpha_chars: int = 8) -> Dict[str, Any]:
    """Decide whether catalog autofill has enough OCR text evidence.

    Price-only matching is useful for candidate display, but it is unsafe for
    automatic product replacement: many unrelated products share prices.  This
    guard blocks DB/CSV product autofill when OCR text is mostly digits, service
    words, or short noisy fragments such as ``69 | 99 | ттоа``.
    """
    norm = _normalize_text_for_match(query_text)
    tokens_all = _content_tokens(norm)
    alpha_tokens = [t for t in tokens_all if re.search(r"[a-zа-я]", t, flags=re.I)]
    alpha_chars = sum(len(re.sub(r"[^a-zа-я]+", "", t, flags=re.I)) for t in alpha_tokens)
    long_alpha_tokens = [t for t in alpha_tokens if len(re.sub(r"[^a-zа-я]+", "", t, flags=re.I)) >= 5]
    reject = (len(alpha_tokens) < int(min_content_tokens)) or (alpha_chars < int(min_alpha_chars)) or (not long_alpha_tokens and len(alpha_tokens) < 3)
    return {
        "reject": bool(reject),
        "normalized": norm[:240],
        "tokens": tokens_all[:40],
        "alpha_tokens": alpha_tokens[:40],
        "alpha_chars": int(alpha_chars),
        "long_alpha_token_count": int(len(long_alpha_tokens)),
        "min_content_tokens": int(min_content_tokens),
        "min_alpha_chars": int(min_alpha_chars),
    }


def _catalog_query_text(result: Mapping[str, Any]) -> str:
    parts: List[str] = []
    for key_path in (
        ("final", "product_name"),
        ("parsed", "product_name"),
        ("ocr", "zone_texts", "product_name"),
        ("ocr", "all_text_joined"),
    ):
        v = _get_nested(result, key_path)
        if v not in (None, ""):
            parts.append(str(v))
    for item in (result.get("ocr", {}) or {}).get("full_tag", []) if isinstance(result.get("ocr"), Mapping) else []:
        if isinstance(item, Mapping) and item.get("text"):
            parts.append(str(item.get("text")))
    return " | ".join(parts)


def _collect_observed_prices(result: Mapping[str, Any], final: Optional[Mapping[str, Any]] = None) -> List[float]:
    vals: List[float] = []
    for v in ((result.get("prices") or {}).get("main") or {}).values() if isinstance((result.get("prices") or {}).get("main"), Mapping) else []:
        f = _price_to_float(v)
        if f is not None:
            vals.append(f)
    by_zone = (result.get("prices") or {}).get("by_zone", {}) if isinstance(result.get("prices"), Mapping) else {}
    if isinstance(by_zone, Mapping):
        for cand_list in by_zone.values():
            if isinstance(cand_list, Sequence) and not isinstance(cand_list, (str, bytes)):
                for c in cand_list:
                    if isinstance(c, Mapping):
                        f = _price_to_float(c.get("value") or c.get("raw_match"))
                    else:
                        f = _price_to_float(c)
                    if f is not None:
                        vals.append(f)
    if final:
        for k in ("main_price", "card_price", "no_card_price", "old_price"):
            f = _price_to_float(final.get(k))
            if f is not None:
                vals.append(f)
    # De-duplicate with cents precision.
    uniq: Dict[str, float] = {}
    for v in vals:
        if _is_positive_number(v):
            uniq[f"{float(v):.2f}"] = float(v)
    return list(uniq.values())


def _get_nested(d: Mapping[str, Any], path: Sequence[str]) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(p)
    return cur


def _normalize_text_for_match(s: str) -> str:
    s = _repair_ocr_text_for_catalog(str(s or ""))
    s = s.lower().replace("ё", "е")
    # Preserve Latin brand words.  Convert only highly ambiguous visual symbols
    # that are common inside Russian words; do not turn LIGHT into 1GHT.
    s = re.sub(r"[|!]+", "1", s)
    s = re.sub(r"(?<=\d)[oо](?=\d)", "0", s, flags=re.I)
    s = re.sub(r"(?<=\d)[зЗ](?=\d)", "3", s)
    s = re.sub(r"(\d+[,.]\d+)\s*[lл]\b", r"\1 л", s, flags=re.I)
    s = re.sub(r"(\d+)\s*[lл]\b", r"\1 л", s, flags=re.I)
    s = re.sub(r"[^0-9a-zа-я]+", " ", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _repair_ocr_text_for_catalog(s: str) -> str:
    """Repair typical OCR homoglyph errors before catalog matching.

    This is deliberately not a generic transliteration layer.  It fixes only
    high-frequency retail OCR distortions and leaves Latin brand names intact.
    Example handled by this function: ``Hаnкток бсзалкогальный ABRAU LGHT`` ->
    ``напиток безалкогольный ABRAU LIGHT``.
    """
    out = str(s or "")
    out = out.replace("0,25L", "0,25л").replace("0.25L", "0.25л")
    for rx, repl in _TEXT_OCR_REPLACEMENTS:
        out = rx.sub(repl, out)

    # Token-level visual normalization for mostly-Cyrillic tokens.
    fixed: List[str] = []
    for tok in re.split(r"(\W+)", out):
        if not tok or re.fullmatch(r"\W+", tok):
            fixed.append(tok)
            continue
        has_cyr = bool(re.search(r"[А-Яа-яЁё]", tok))
        has_lat = bool(re.search(r"[A-Za-z]", tok))
        if has_cyr and has_lat:
            t = tok.translate(_TEXT_VISUAL_LATIN_TO_CYR)
            # Lowercase Latin n is often OCR for Cyrillic п in words like Hаnкток.
            t = re.sub(r"(?i)n", "п", t)
            fixed.append(t)
        else:
            fixed.append(tok)
    out = "".join(fixed)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _content_tokens(s: str) -> List[str]:
    tokens = []
    for t in re.findall(r"[0-9a-zа-я]+", s.lower(), flags=re.I):
        if len(t) < 2:
            continue
        if t in _SERVICE_WORDS:
            continue
        # Do not let one/two-digit garbage dominate catalog matching, but keep
        # package fragments like 25л / 025л as useful evidence.
        if t.isdigit() and len(t) < 3:
            continue
        tokens.append(t)
    return tokens


def _text_similarity(
    q_norm: str,
    q_tokens: Sequence[str],
    n_norm: str,
    n_tokens: set[str],
    *,
    fuzzy_token_match: bool = True,
    min_fuzzy_token_score: float = 0.74,
) -> float:
    if not q_tokens or not n_norm:
        return 0.0
    q_set = set(q_tokens)
    inter = q_set & n_tokens
    fuzzy_cov = 0.0
    fuzzy_hits = 0
    if fuzzy_token_match:
        fuzzy_cov, fuzzy_hits = _fuzzy_token_coverage(q_tokens, n_tokens, min_score=min_fuzzy_token_score)
    exact_coverage = len(inter) / max(1, len(q_set))
    coverage = max(exact_coverage, fuzzy_cov)
    jaccard = len(inter) / max(1, len(q_set | n_tokens))
    if not inter and fuzzy_hits <= 0:
        if len(q_norm) >= 7 and (q_norm in n_norm or n_norm in q_norm):
            return 0.34
        return 0.0
    seq = 0.0
    if coverage >= 0.16 or len(inter) >= 2 or fuzzy_hits >= 2:
        seq = SequenceMatcher(None, q_norm[:220], n_norm[:220]).ratio()
    # Fuzzy token coverage is the main signal for OCR-corrected catalog match;
    # full sequence ratio is secondary because line order and service text vary.
    return float(max(0.0, min(1.0, 0.66 * coverage + 0.14 * jaccard + 0.20 * seq)))


def _fuzzy_token_coverage(q_tokens: Sequence[str], n_tokens: Iterable[str], *, min_score: float) -> Tuple[float, int]:
    name_tokens = [t for t in n_tokens if len(t) >= 2]
    if not q_tokens or not name_tokens:
        return 0.0, 0
    total = 0.0
    matched = 0.0
    hits = 0
    for qt in dict.fromkeys(q_tokens):
        if qt in _SERVICE_WORDS:
            continue
        weight = 1.25 if len(qt) >= 5 else 1.0
        total += weight
        if qt in name_tokens:
            matched += weight
            hits += 1
            continue
        best = 0.0
        for nt in name_tokens:
            if abs(len(qt) - len(nt)) > max(3, int(0.55 * max(len(qt), len(nt)))):
                continue
            # Exact substring fragments such as lght/light or 025/025л.
            if len(qt) >= 3 and (qt in nt or nt in qt):
                score = min(0.96, 0.72 + 0.04 * min(len(qt), len(nt)))
            else:
                score = SequenceMatcher(None, qt, nt).ratio()
            if score > best:
                best = score
        if best >= float(min_score):
            matched += weight * min(1.0, best)
            hits += 1
    if total <= 0:
        return 0.0, 0
    return float(matched / total), int(hits)


def _same_catalog_family(
    a: StructuredCatalogCandidate,
    b: StructuredCatalogCandidate,
    *,
    min_name_overlap: float,
    min_price_score: float,
) -> bool:
    if a.item_id and b.item_id and str(a.item_id) == str(b.item_id):
        return True
    if min(float(a.price_score), float(b.price_score)) < float(min_price_score):
        return False
    an = set(_content_tokens(_normalize_text_for_match(a.name)))
    bn = set(_content_tokens(_normalize_text_for_match(b.name)))
    if not an or not bn:
        return False
    overlap = len(an & bn) / max(1, min(len(an), len(bn)))
    same_brand = bool(str(a.brand or "").strip() and str(a.brand or "").strip().lower() == str(b.brand or "").strip().lower())
    return bool(overlap >= float(min_name_overlap) or (same_brand and overlap >= float(min_name_overlap) - 0.10))


def _to_float_or_none(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    s = str(v).strip().lower().replace(",", ".")
    if s in _EMPTY_PRICE_KEYS:
        return None
    s = re.sub(r"[^0-9.+\-]", "", s)
    if not s or s in {".", "+", "-"}:
        return None
    try:
        f = float(s)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def _to_bool_or_none(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "да"}:
        return True
    if s in {"0", "false", "no", "n", "нет"}:
        return False
    return None


def _is_positive_number(v: Any) -> bool:
    try:
        f = float(v)
        return math.isfinite(f) and f > 0
    except Exception:
        return False


def _price_to_float(v: Any) -> Optional[float]:
    norm = product_normalize_price_value(v)
    if norm is None:
        return None
    try:
        return float(norm)
    except Exception:
        return None


def _fmt_price(v: Any) -> Optional[str]:
    f = _to_float_or_none(v)
    if f is None or f <= 0:
        return None
    return f"{f:.2f}"


def _price_matches_any(value: Optional[float], observed: Iterable[float], rel_tol: float) -> bool:
    if value is None:
        return False
    for obs in observed:
        denom = max(1.0, abs(float(value)), abs(float(obs)))
        if abs(float(value) - float(obs)) / denom <= rel_tol:
            return True
    return False
