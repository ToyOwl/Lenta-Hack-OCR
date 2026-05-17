"""
LLM/catalog post-correction for OCR results of Russian retail price tags.
The pipeline can run without a PaddleNLPmodel and without a product catalog.
When no model is available it still produces a safe rule/catalog based correction block, so JSON schema and debug overlays
remain stable in production. PaddleNLP is used instead of HuggingFace Transformers so the OCR and correction stages can stay in one Paddle/PaddleOCR GPU runtime without importing PyTorch.
"""

from __future__ import annotations

import csv
import json
import math
import re
import time
import urllib.request
from dataclasses import asdict, dataclass, field

from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .product_card import baseline_final_from_result
from .product_card import clean_product_name as retail_clean_product_name
from .product_card import enrich_final_fields
from .product_card import extract_volume as retail_extract_volume

_EMPTY_PRICE_KEYS = {"", "none", "null", "nan", "-"}

@dataclass
class ProductCandidate:

    """Catalog candidate returned to the prompt and to diagnostics."""

    item_id: str = ""
    category_id: str = ""
    name: str = ""
    slug: str = ""
    brand: str = ""
    manufacturer: str = ""
    country: str = ""
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

    def to_prompt_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "name": self.name,
            "brand": self.brand,
            "manufacturer": self.manufacturer,
            "country": self.country,
            "price": _fmt_float(self.price),
            "price_regular": _fmt_float(self.price_regular),
            "cost": _fmt_float(self.cost),
            "cost_regular": _fmt_float(self.cost_regular),
            "is_promo": self.is_promo,
            "is_loyalty": self.is_loyalty,
            "score": round(float(self.score), 4),
            "text_score": round(float(self.text_score), 4),
            "price_score": round(float(self.price_score), 4),
            "match_status": self.match_status,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LLMCorrectionResult:

    """Stable serializable result of the correction stage."""

    enabled: bool
    status: str
    model_name: str
    model_loaded: bool
    backend: str
    latency_ms: float
    model_path: str = ""
    load_errors: List[str] = field(default_factory=list)
    load_debug: Dict[str, Any] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_response: str = ""
    structured: Dict[str, Any] = field(default_factory=dict)
    final: Dict[str, Any] = field(default_factory=dict)
    catalog_candidates: List[Dict[str, Any]] = field(default_factory=list)
    evidence_summary: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class ProductCatalogMatcher:
    """Lightweight fuzzy matcher over a CSV product catalog.

    Expected columns are flexible.  The Lenta-like CSV columns used by the current
    project are supported directly: category_id, item_id, name, slug, price,
    price_regular, cost, cost_regular, is_promo, is_loyalty.
    """

    def __init__(
        self,
        csv_path: str | Path | None = None,
        name_column: str = "name",
        item_id_column: str = "item_id",
        brand_column: str = "brand",
        manufacturer_column: str = "manufacturer",
        country_column: str = "country",
        price_columns: Sequence[str] = ("price", "price_regular", "cost", "cost_regular"),
        min_text_score: float = 0.30,
        min_accept_score: float = 0.56,
        max_rows: int = 200_000,
    ) -> None:
        self.csv_path = Path(csv_path) if csv_path else None
        self.name_column = name_column
        self.item_id_column = item_id_column
        self.brand_column = brand_column
        self.manufacturer_column = manufacturer_column
        self.country_column = country_column
        self.price_columns = tuple(price_columns)
        self.min_text_score = float(min_text_score)
        self.min_accept_score = float(min_accept_score)
        self.max_rows = int(max_rows)
        self.rows: List[Dict[str, Any]] = []
        self._norm_names: List[str] = []
        if self.csv_path:
            self.load(self.csv_path)

    @property
    def enabled(self) -> bool:
        return bool(self.rows)

    def load(self, csv_path: str | Path) -> None:
        p = Path(csv_path)
        if not p.exists():
            return
        rows: List[Dict[str, Any]] = []

        with open(p, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= self.max_rows:
                    break
                name = str(row.get(self.name_column, "") or "").strip()
                if not name:
                    continue
                rows.append(row)
        self.rows = rows
        self._norm_names = [_normalize_text_for_match(str(r.get(self.name_column, ""))) for r in self.rows]

    def find_candidates(self, query_text: str, prices: Sequence[float] = (), top_k: int = 5,) -> List[ProductCandidate]:

        if not self.rows or not query_text.strip():
            return []
        q_norm = _normalize_text_for_match(query_text)
        q_tokens = _content_tokens(q_norm)

        if not q_tokens:
            return []

        scored: List[Tuple[float, ProductCandidate]] = []
        for row, n_norm in zip(self.rows, self._norm_names):
            text_score = _text_similarity(q_norm, q_tokens, n_norm)
            if text_score < self.min_text_score:
                continue
            price_score = self._price_similarity(row, prices)

            # Text is primary. Price helps only after textual evidence exists.
            combined = 0.82 * text_score + 0.18 * price_score
            cand = self._row_to_candidate(row)
            cand.text_score = float(text_score)
            cand.price_score = float(price_score)
            cand.score = float(combined)
            cand.match_status = "candidate" if cand.score >= self.min_accept_score else "weak_candidate"
            scored.append((combined, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[: max(0, int(top_k))]]

    def _row_to_candidate(self, row: Mapping[str, Any]) -> ProductCandidate:
        return ProductCandidate(
            item_id=str(row.get("item_id", row.get(self.item_id_column, "")) or ""),
            category_id=str(row.get("category_id", "") or ""),
            name=str(row.get(self.name_column, "") or ""),
            slug=str(row.get("slug", "") or ""),
            brand=str(row.get(self.brand_column, row.get("brand", "")) or ""),
            manufacturer=str(row.get(self.manufacturer_column, row.get("manufacturer", row.get("producer", ""))) or ""),
            country=str(row.get(self.country_column, row.get("country", "")) or ""),
            price=_to_float_or_none(row.get("price")),
            price_regular=_to_float_or_none(row.get("price_regular")),
            cost=_to_float_or_none(row.get("cost")),
            cost_regular=_to_float_or_none(row.get("cost_regular")),
            is_promo=_to_bool_or_none(row.get("is_promo")),
            is_loyalty=_to_bool_or_none(row.get("is_loyalty")),
        )

    def _price_similarity(self, row: Mapping[str, Any], prices: Sequence[float]) -> float:
        vals = [_to_float_or_none(row.get(c)) for c in self.price_columns]
        vals = [v for v in vals if v is not None and v > 0]
        observed = [float(p) for p in prices if p is not None and p > 0]

        if not vals or not observed:
            return 0.0

        best = 0.0
        for a in vals:
            for b in observed:
                denom = max(1.0, abs(a), abs(b))
                rel = abs(a - b) / denom
                if rel <= 0.01:
                    best = max(best, 1.0)
                elif rel <= 0.05:
                    best = max(best, 0.80)
                elif rel <= 0.12:
                    best = max(best, 0.45)
        return best

    def debug_info(self) -> Dict[str, Any]:
        return {"enabled": self.enabled, "csv_path": str(self.csv_path) if self.csv_path else "", "rows": len(self.rows),
            "name_column": self.name_column, "brand_column": self.brand_column, "manufacturer_column": self.manufacturer_column,
            "country_column": self.country_column, "min_accept_score": self.min_accept_score, }



def _install_aistudio_sdk_download_compat_shim() -> str:
    """Install a minimal compatibility shim for PaddleNLP 3.x beta.
    Some PaddleNLP builds import ``download`` from ``aistudio_sdk.hub`` during
    package import.  Several aistudio-sdk releases do not expose this symbol,
    which breaks PaddleNLP even when we load a fully local model and never need
    AiStudio downloads.  The shim is intentionally minimal: it only makes the
    import succeed; if PaddleNLP later actually tries to download from AiStudio,
    the shim raises a clear error instead of failing with an opaque ImportError.
    """
    try:
        import aistudio_sdk.hub as hub  # type: ignore
    except Exception as e:
        return f"aistudio_sdk_import_failed:{type(e).__name__}:{e}"

    if hasattr(hub, "download"):
        return "aistudio_sdk.hub.download_exists"

    def _missing_download(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "aistudio_sdk.hub.download is missing in the installed aistudio-sdk. "
            "Install a compatible package, for example: python -m pip install --force-reinstall aistudio-sdk==0.2.6. "
            "For offline inference, use a local model_path and paddlenlp_local_files_only=true."
        )

    setattr(hub, "download", _missing_download)
    return "aistudio_sdk.hub.download_shim_installed"

class LLMPriceTagCorrector:
    """Second-stage OCR corrector with optional product-catalog grounding."""

    SYSTEM_INSTRUCTION = """Ты — постпроцессор OCR российских ценников. Исправляй только по доказательствам из OCR, геометрии и каталога. Не выдумывай товар из каталога: если кандидат не подходит, верни product_match.status='not_found'. Отвечай только JSON."""
    PROMPT_TEMPLATE = """Задача: исправить OCR ценника и вернуть безопасный JSON для товарной карточки. Важные правила:
1. Не склеивай вертикальные ценовые строки как рубли/копейки. Если видишь последовательность вроде 70 | 80 | 89, расположенную столбцом, это несколько цен/порогов, а не 89.80.
2. main_price заполняй только если уверенно понятно, какая цена является итоговой. Если цена неоднозначна, main_price=null, price_selection_status='ambiguous', needs_review=true.
3. Разделяй regular_price, promo_price, card_price, no_card_price, old_price. Для красных промо-ценников цена по карте обычно является promo_price.
4. Для весовых товаров обязательно возвращай product_type='weighted_scale', scale_number и sale_unit='за кг/весовой', если есть «номер на весах».
5. Для алкоголя отличай крепость от скидки: 4.5%, 5%, 6% рядом с названием/объёмом 0.5л — это alcohol_percent_raw, а не discount_percent_raw. discount_percent_raw заполняй только при явных словах скидка/акция/выгода/экономия или явном промо-проценте.
6. Если есть каталог товаров, используй его только как подсказку. Если ни один кандидат не подходит, product_match.status='not_found'. Не выдумывай товар из каталога.
7. Исправляй OCR-ошибки в названии: латинские визуальные аналоги могут означать кириллицу, например PYCCKOE MOPE -> РУССКОЕ МОРЕ, Caxap -> Сахар. Но не порть реальные латинские бренды/названия алкоголя.
8. По возможности выделяй producer/brand и country. Страну часто видно в скобках: (Россия), (Беларусь).
9. Ответ строго JSON без markdown.

Шаблон: {template_name}
Текущий baseline parse:
{baseline_json}

OCR/geometry evidence:
{evidence_json}

Каталоговые кандидаты, top-k:
{catalog_json}

Схема ответа:
{{
  "product_name": string|null,
  "product_type": "weighted_scale"|"packaged"|"piece"|"alcohol"|"unknown",
  "producer": string|null,
  "brand": string|null,
  "country": string|null,
  "product_match": {{
    "status": "matched"|"not_found"|"not_used"|"ambiguous",
    "item_id": string|null,
    "catalog_name": string|null,
    "score": number
  }},
  "main_price": string|null,
  "regular_price": string|null,
  "promo_price": string|null,
  "card_price": string|null,
  "no_card_price": string|null,
  "old_price": string|null,
  "unit": string|null,
  "sale_unit": string|null,
  "volume": string|null,
  "package_size": string|null,
  "scale_number": string|null,
  "barcode_text_raw": string|null,
  "alcohol_percent_raw": string|null,
  "discount_percent_raw": string|null,
  "promo_condition": string|null,
  "price_candidates": [{{"value": string, "role": string, "source": string, "confidence": number}}],
  "price_selection_status": "ok"|"ambiguous"|"not_found",
  "needs_review": boolean,
  "review_reasons": [string],
  "confidence": number
}}

JSON:"""

    def __init__(
        self,
        enabled: bool = False,
        backend: str = "paddlenlp",
        model_path: str | Path | None = None,
        endpoint_url: str = "",
        api_key: str = "",
        model_name: str = "Qwen/Qwen2-0.5B",
        n_ctx: int = 4096,
        n_batch: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        max_tokens: int = 384,
        timeout_s: float = 20.0,
        force_run: bool = False,
        run_on_templates: Sequence[str] = ("shelf_red_promo", "hanging_yellow_promo_large"),
        run_on_warning_quality: bool = True,
        min_ocr_confidence: float = 0.65,
        fallback_when_unavailable: bool = True,
        catalog: Optional[ProductCatalogMatcher] = None,
        catalog_top_k: int = 8,
        min_catalog_match_score: float = 0.56,
        paddlenlp_device: str = "auto",
        paddlenlp_dtype: str = "auto",
        paddlenlp_local_files_only: bool = False,
        paddlenlp_trust_remote_code: bool = False,
        debug: bool = False,
        verbose: bool = False,
    ) -> None:
        self.backend = str(backend or "none").lower().strip()
        self.enabled = bool(enabled) and self.backend != "none"
        self.model_path = Path(model_path) if model_path else None
        self.endpoint_url = str(endpoint_url or "")
        self.api_key = str(api_key or "")
        self.model_name = str(model_name or (self.model_path.name if self.model_path else "Qwen/Qwen2-0.5B"))
        self.n_ctx = int(n_ctx)
        self.n_batch = int(n_batch)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)
        self.timeout_s = float(timeout_s)
        self.force_run = bool(force_run)
        self.run_on_templates = set(str(x) for x in run_on_templates)
        self.run_on_warning_quality = bool(run_on_warning_quality)
        self.min_ocr_confidence = float(min_ocr_confidence)
        self.fallback_when_unavailable = bool(fallback_when_unavailable)
        self.catalog = catalog or ProductCatalogMatcher()
        self.catalog_top_k = int(catalog_top_k)
        self.min_catalog_match_score = float(min_catalog_match_score)
        self.paddlenlp_device = str(paddlenlp_device or "auto").lower().strip()
        self.paddlenlp_dtype = str(paddlenlp_dtype or "auto").lower().strip()
        self.paddlenlp_local_files_only = bool(paddlenlp_local_files_only)
        self.paddlenlp_trust_remote_code = bool(paddlenlp_trust_remote_code)
        self.debug = bool(debug)
        self.verbose = bool(verbose)
        self.llm: Any = None
        self.tokenizer: Any = None
        self._paddle: Any = None
        self.load_errors: List[str] = []
        self.load_debug: Dict[str, Any] = {}

        if self.backend not in {"none", "paddlenlp", "openai_compatible", "http"}:
            if self.verbose:
                print(f"[LLM] unsupported backend={self.backend!r}; correction disabled")
            self.backend = "none"
            self.enabled = False

        if self.enabled and self.backend == "paddlenlp":
            self._load_paddlenlp()

    @staticmethod
    def _resolve_model_path_from_config(cfg: Mapping[str, Any], raw_path: str) -> str:
        """Resolve model_path robustly for Windows/dev workflows.
        Priority:
        1. absolute path as-is;
        2. path relative to current working directory;
        3. path relative to the YAML directory;
        4. path relative to the parent of the YAML directory.
        The last case is important when config is in `configs/` and models are
        stored in project-root `models/`. If nothing exists, return the first
        deterministic candidate and expose all candidates in load_errors/debug.
        """
        raw = str(raw_path or "").strip().strip('"').strip("'")

        if not raw:
            return ""
        p = Path(raw).expanduser()

        if p.is_absolute():
            return str(p)

        meta = cfg.get("_meta", {}) if isinstance(cfg, Mapping) else {}
        roots: List[Path] = [Path.cwd()]
        config_dir_raw = meta.get("config_dir") if isinstance(meta, Mapping) else None
        if config_dir_raw:
            config_dir = Path(str(config_dir_raw))
            roots.append(config_dir)
            roots.append(config_dir.parent)

        seen = set()
        candidates: List[Path] = []
        for root in roots:
            cand = (root / p).resolve()
            key = str(cand).lower()
            if key not in seen:
                seen.add(key)
                candidates.append(cand)
        for cand in candidates:
            if cand.exists():
                return str(cand)
        return str(candidates[0] if candidates else p)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "LLMPriceTagCorrector":
        c = cfg.get("llm_corrector", {}) if isinstance(cfg, Mapping) else {}
        catalog_cfg = c.get("catalog", {}) if isinstance(c, Mapping) else {}
        price_cols = catalog_cfg.get("price_columns", ["price", "price_regular", "cost", "cost_regular"])
        if isinstance(price_cols, str):
            price_cols = [x.strip() for x in price_cols.split(",") if x.strip()]
        catalog = ProductCatalogMatcher(
            csv_path=str(catalog_cfg.get("path", "") or "") or None,
            name_column=str(catalog_cfg.get("name_column", "name") or "name"),
            item_id_column=str(catalog_cfg.get("item_id_column", "item_id") or "item_id"),
            brand_column=str(catalog_cfg.get("brand_column", "brand") or "brand"),
            manufacturer_column=str(catalog_cfg.get("manufacturer_column", "manufacturer") or "manufacturer"),
            country_column=str(catalog_cfg.get("country_column", "country") or "country"),
            price_columns=tuple(price_cols),
            min_text_score=float(catalog_cfg.get("min_text_score", 0.30)),
            min_accept_score=float(catalog_cfg.get("min_accept_score", c.get("min_catalog_match_score", 0.56))),
            max_rows=int(catalog_cfg.get("max_rows", 200000)),
        )
        run_templates = c.get("run_on_templates", ["shelf_red_promo", "hanging_yellow_promo_large"])
        if isinstance(run_templates, str):
            run_templates = [x.strip() for x in run_templates.split(",") if x.strip()]
        backend = str(c.get("backend", "none") or "none").lower().strip()
        legacy_enabled = c.get("enabled", None)
        if legacy_enabled is False:
            backend = "none"
        elif legacy_enabled is True and backend == "none":
            backend = "paddlenlp"
        raw_model_path = str(c.get("model_path", "") or "")
        resolved_model_path = cls._resolve_model_path_from_config(cfg, raw_model_path) if raw_model_path else ""
        raw_model_name = c.get("model_name", None)

        if raw_model_name is None:
            model_name = "Qwen/Qwen2-0.5B" if not resolved_model_path else ""
        else:
            model_name = str(raw_model_name or "")

        return cls(
            enabled=(backend != "none"),
            backend=backend,
            model_path=resolved_model_path or None,
            endpoint_url=str(c.get("endpoint_url", "") or ""),
            api_key=str(c.get("api_key", "") or ""),
            model_name=model_name,
            n_ctx=int(c.get("n_ctx", 4096)),
            n_batch=int(c.get("n_batch", 512)),
            temperature=float(c.get("temperature", 0.0)),
            top_p=float(c.get("top_p", 0.95)),
            max_tokens=int(c.get("max_tokens", 384)),
            timeout_s=float(c.get("timeout_s", 20.0)),
            force_run=bool(c.get("force_run", False)),
            run_on_templates=run_templates,
            run_on_warning_quality=bool(c.get("run_on_warning_quality", True)),
            min_ocr_confidence=float(c.get("min_ocr_confidence", 0.65)),
            fallback_when_unavailable=bool(c.get("fallback_when_unavailable", True)),
            catalog=catalog,
            catalog_top_k=int(catalog_cfg.get("top_k", c.get("catalog_top_k", 8))),
            min_catalog_match_score=float(c.get("min_catalog_match_score", catalog_cfg.get("min_accept_score", 0.56))),
            paddlenlp_device=str(c.get("paddlenlp_device", "auto") or "auto"),
            paddlenlp_dtype=str(c.get("paddlenlp_dtype", "auto") or "auto"),
            paddlenlp_local_files_only=bool(c.get("paddlenlp_local_files_only", False)),
            paddlenlp_trust_remote_code=bool(c.get("paddlenlp_trust_remote_code", False)),
            debug=bool(c.get("debug", False)),
            verbose=bool(c.get("verbose", False)),
        )


    def _load_paddlenlp(self) -> None:
        """
        Load a small PaddleNLP mode.
        """
        model_id = str(self.model_path) if self.model_path else self.model_name
        self.load_debug = {
            "backend": self.backend, "model_id": model_id, "model_path": str(self.model_path) if self.model_path else "",
            "model_name": self.model_name, "local_files_only": self.paddlenlp_local_files_only,}

        if not model_id:
            msg = "model_name/model_path is empty; backend disabled"
            self.load_errors.append(msg)
            if self.verbose:
                print(f"[LLM/PaddleNLP] {msg}")
            return

        if self.model_path and not self.model_path.exists():
            msg = f"model_path not found: {self.model_path}"
            self.load_errors.append(msg)
            self.load_debug["model_path_exists"] = False
            if self.verbose:
                print(f"[LLM/PaddleNLP] {msg}")
            return

        if self.model_path and self.model_path.exists():
            try:
                files = sorted([x.name for x in self.model_path.iterdir()])
                self.load_debug["model_path_files"] = files[:80]
                expected_any = {"config.json", "tokenizer_config.json", "model_config.json", "tokenizer.json", "vocab.json", "merges.txt"}
                if not any(x in files for x in expected_any):
                    self.load_errors.append("model_path_exists_but_no_common_config_or_tokenizer_files")
            except Exception as e:
                self.load_errors.append(f"model_path_scan_error:{type(e).__name__}:{e}")
        shim_status = _install_aistudio_sdk_download_compat_shim()
        self.load_debug["aistudio_sdk_download_compat"] = shim_status

        try:
            import paddle
            from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            msg = f"paddle/paddlenlp are not available: {type(e).__name__}: {e}"
            self.load_errors.append(msg)
            if self.verbose:
                print(f"[LLM/PaddleNLP] {msg}")
            return

        self._paddle = paddle
        device = self._resolve_paddlenlp_device(paddle)
        try:
            paddle.set_device(device)
        except Exception as e:
            msg = f"failed to set device={device!r}: {type(e).__name__}: {e}; fallback to cpu"
            self.load_errors.append(msg)
            if self.verbose:
                print(f"[LLM/PaddleNLP] {msg}")
            device = "cpu"
            try:
                paddle.set_device("cpu")
            except Exception:
                pass
        self.paddlenlp_device = str(device)

        tokenizer_kwargs: Dict[str, Any] = {}
        if self.paddlenlp_local_files_only:
          tokenizer_kwargs["local_files_only"] = True
        if self.paddlenlp_trust_remote_code:
          tokenizer_kwargs["trust_remote_code"] = True
        try:
          self.tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        except TypeError as e:
          if self.paddlenlp_local_files_only and not self.model_path:
             self.tokenizer = None
             msg = f"local_files_only is unsupported by this PaddleNLP release; use model_path instead: {type(e).__name__}: {e}"
             self.load_errors.append(msg)
             if self.verbose:
                print(f"[LLM/PaddleNLP] {msg}")
                return
             self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception as e:  # pragma: no cover
            self.tokenizer = None
            msg = f"failed to load tokenizer: {type(e).__name__}: {e}"
            self.load_errors.append(msg)
            if self.verbose:
                print(f"[LLM/PaddleNLP] {msg}")
            return

        dtype = self._resolve_paddlenlp_dtype(device)
        model_kwargs: Dict[str, Any] = {}
        if dtype:
            model_kwargs["dtype"] = dtype
        if self.paddlenlp_local_files_only:
            model_kwargs["local_files_only"] = True
        if self.paddlenlp_trust_remote_code:
            model_kwargs["trust_remote_code"] = True

        try:
            self.llm = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        except TypeError as e:
            # Some PaddleNLP releases do not accept dtype/local_files_only/trust_remote_code.
            # In strict local mode with model_name, do not retry in a way that can download.
            if self.paddlenlp_local_files_only and not self.model_path:
                self.llm = None
                msg = f"local_files_only is unsupported by this PaddleNLP release; use model_path instead: {type(e).__name__}: {e}"
                self.load_errors.append(msg)
                if self.verbose:
                    print(f"[LLM/PaddleNLP] {msg}")
                return
            try:
                self.llm = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype) if dtype else AutoModelForCausalLM.from_pretrained(model_id)
            except TypeError:
                self.llm = AutoModelForCausalLM.from_pretrained(model_id)
        except Exception as e:  # pragma: no cover
            self.llm = None
            msg = f"failed to load model: {type(e).__name__}: {e}"
            self.load_errors.append(msg)
            if self.verbose:
                print(f"[LLM/PaddleNLP] {msg}")
            return

        try:
            self.llm.eval()
        except Exception:
            pass
        self.model_name = model_id
        self.load_debug.update({
            "loaded": True,
            "device": self.paddlenlp_device,
            "dtype": dtype or "default",
            "tokenizer_class": type(self.tokenizer).__name__,
            "model_class": type(self.llm).__name__,
        })
        if self.verbose:
            print(f"[LLM/PaddleNLP] loaded {model_id} device={self.paddlenlp_device} dtype={dtype or 'default'}")

    def _resolve_paddlenlp_device(self, paddle: Any) -> str:
        configured = str(self.paddlenlp_device or "auto").lower().strip()
        if configured and configured != "auto":
            if configured == "cuda":
                return "gpu"
            if configured.startswith("cuda:"):
                return "gpu:" + configured.split(":", 1)[1]
            return configured
        try:
            if bool(paddle.is_compiled_with_cuda()):
                return "gpu"
        except Exception:
            pass
        return "cpu"

    def _resolve_paddlenlp_dtype(self, device: str) -> Optional[str]:
        dtype = str(self.paddlenlp_dtype or "auto").lower().strip()
        if dtype in {"", "none", "default"}:
            return None
        if dtype == "auto":
            return "float16" if str(device).startswith("gpu") else "float32"
        if dtype in {"fp16", "half"}:
            return "float16"
        if dtype in {"bf16", "bfloat16"}:
            return "bfloat16"
        if dtype in {"fp32", "float"}:
            return "float32"
        return dtype

    @property
    def model_loaded(self) -> bool:
        if self.backend == "none":
            return False
        if self.backend == "paddlenlp":
            return self.llm is not None
        if self.backend in {"openai_compatible", "http"}:
            return bool(self.endpoint_url)
        return False

    def should_run(self, result: Mapping[str, Any]) -> bool:
        if not self.enabled:
            return False
        if self.force_run:
            return True
        template_name = str(((result.get("template") or {}).get("template_name") or ""))
        if template_name in self.run_on_templates:
            return True
        quality_status = str(((result.get("quality") or {}).get("status") or ""))
        if self.run_on_warning_quality and quality_status not in {"", "ok"}:
            return True
        ocr_conf = _mean_ocr_conf(result)
        if ocr_conf and ocr_conf < self.min_ocr_confidence:
            return True
        # run when baseline parser already marked ambiguous/suspicious
        if _detect_vertical_price_stack(result).get("detected"):
            return True
        return False

    def _result_common_kwargs(self) -> Dict[str, Any]:
        return {
            "model_path": str(self.model_path) if self.model_path else "",
            "load_errors": list(self.load_errors),
            "load_debug": dict(self.load_debug),
        }

    def correct_result(self, result: Mapping[str, Any]) -> Optional[LLMCorrectionResult]:
        if not self.enabled:
            return None
        if not self.should_run(result):
            return LLMCorrectionResult(
                enabled=True,
                status="skipped_by_policy",
                model_name=self.model_name,
                model_loaded=self.model_loaded,
                backend=self.backend,
                latency_ms=0.0,
                **self._result_common_kwargs(),
                final=_baseline_final(result),
                evidence_summary={"reason": "policy"},
                catalog_candidates=[],
            )

        start = time.perf_counter()
        evidence = _build_evidence(result)
        observed_prices = _collect_observed_prices(result)
        catalog_query = _catalog_query_text(result, evidence)
        candidates = self.catalog.find_candidates(catalog_query, prices=observed_prices, top_k=self.catalog_top_k) if self.catalog.enabled else []
        prompt = self._build_prompt(result, evidence, candidates)

        status = "fallback"
        raw = ""
        structured: Dict[str, Any] = {}
        warnings: List[str] = []
        usage_prompt = 0
        usage_completion = 0

        if self.model_loaded:
            try:
                raw, usage_prompt, usage_completion = self._call_model(prompt)
                structured = self._extract_json(raw)
                status = "ok" if structured else "invalid_json"
                if not structured:
                    warnings.append("llm_invalid_json")
            except Exception as e:
                status = "error"
                warnings.append(f"llm_error:{type(e).__name__}:{e}")
        else:
            status = "fallback_model_unavailable"
            warnings.extend(["model_unavailable", *self.load_errors])
            if not self.fallback_when_unavailable:
                return LLMCorrectionResult(
                    enabled=True,
                    status=status,
                    model_name=self.model_name,
                    model_loaded=False,
                    backend=self.backend,
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                    **self._result_common_kwargs(),
                    structured={},
                    final=_baseline_final(result),
                    catalog_candidates=[c.to_dict() for c in candidates],
                    evidence_summary={**evidence, "llm_load_debug": self.load_debug},
                    warnings=["model_unavailable", *self.load_errors],
                )

        if not structured:
            structured = self._rule_based_structured(result, evidence, candidates)

        final, post_warnings = self._postprocess_structured(result, structured, candidates)
        warnings.extend(post_warnings)
        return LLMCorrectionResult(
            enabled=True,
            status=status,
            model_name=self.model_name,
            model_loaded=self.model_loaded,
            backend=self.backend,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            **self._result_common_kwargs(),
            prompt_tokens=usage_prompt,
            completion_tokens=usage_completion,
            raw_response=raw if self.debug else "",
            structured=structured,
            final=final,
            catalog_candidates=[c.to_dict() for c in candidates],
            evidence_summary={**evidence, "llm_load_debug": self.load_debug},
            warnings=warnings,
        )

    def _build_prompt(self, result: Mapping[str, Any], evidence: Mapping[str, Any], candidates: Sequence[ProductCandidate]) -> str:
        template_name = str(((result.get("template") or {}).get("template_name") or "unknown"))
        baseline = {
            "parsed": result.get("parsed", {}),
            "prices_main": (result.get("prices") or {}).get("main"),
            "spatial_fields": (result.get("spatial") or {}).get("fields", {}),
        }
        return self.PROMPT_TEMPLATE.format(
            template_name=template_name,
            baseline_json=json.dumps(baseline, ensure_ascii=False, indent=2),
            evidence_json=json.dumps(evidence, ensure_ascii=False, indent=2),
            catalog_json=json.dumps([c.to_prompt_dict() for c in candidates], ensure_ascii=False, indent=2),
        )

    def _call_model(self, prompt: str) -> Tuple[str, int, int]:
        if self.backend == "paddlenlp":
            return self._call_paddlenlp(prompt)
        if self.backend in {"openai_compatible", "http"}:
            payload = {
                "model": self.model_name or "local-model",
                "messages": [
                    {"role": "system", "content": self.SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
            }
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                self.endpoint_url.rstrip("/") + "/chat/completions" if self.endpoint_url and not self.endpoint_url.endswith("/chat/completions") else self.endpoint_url,
                data=data,
                headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # nosec - local runtime expected
                obj = json.loads(resp.read().decode("utf-8"))
            text = str(obj.get("choices", [{}])[0].get("message", {}).get("content", ""))
            usage = obj.get("usage", {}) or {}
            return text, int(usage.get("prompt_tokens", 0) or 0), int(usage.get("completion_tokens", 0) or 0)
        return "", 0, 0

    def _call_paddlenlp(self, prompt: str) -> Tuple[str, int, int]:
        if self.llm is None or self.tokenizer is None or self._paddle is None:
            return "", 0, 0
        paddle = self._paddle
        tokenizer = self.tokenizer
        messages = [
            {"role": "system", "content": self.SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ]

        prompt_text = self.SYSTEM_INSTRUCTION + "\n\n" + prompt
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if isinstance(rendered, str) and rendered.strip():
                    prompt_text = rendered
            except Exception:
                pass

        try:
            inputs = tokenizer(
                prompt_text,
                return_tensors="pd",
                truncation=True,
                max_length=max(512, self.n_ctx - self.max_tokens),
            )
        except TypeError:
            inputs = tokenizer(prompt_text, return_tensors="pd")

        input_len = int(inputs["input_ids"].shape[-1]) if isinstance(inputs, Mapping) and "input_ids" in inputs else 0
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_tokens,
        }
        if self.temperature > 0.0:
            gen_kwargs.update({"decode_strategy": "sampling", "temperature": self.temperature, "top_p": self.top_p})
        else:
            gen_kwargs.update({"decode_strategy": "greedy_search"})

        with paddle.no_grad():
            try:
                output = self.llm.generate(**inputs, **gen_kwargs)
            except TypeError:
                # Compatibility with older PaddleNLP generate signatures.
                gen_kwargs.pop("decode_strategy", None)
                gen_kwargs.pop("temperature", None)
                gen_kwargs.pop("top_p", None)
                try:
                    output = self.llm.generate(**inputs, **gen_kwargs)
                except TypeError:
                    gen_kwargs.pop("max_new_tokens", None)
                    gen_kwargs["max_length"] = input_len + self.max_tokens
                    output = self.llm.generate(**inputs, **gen_kwargs)

        ids = output[0] if isinstance(output, (list, tuple)) else output
        text_full = _decode_paddlenlp_ids(tokenizer, ids)
        text_new = _decode_paddlenlp_new_tokens(tokenizer, ids, input_len)
        text = text_new if "{" in text_new else text_full
        if prompt_text and text.startswith(prompt_text):
            text = text[len(prompt_text):].strip()
        return text.strip(), input_len, _generated_token_count(ids, input_len)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        if not text:
            return {}
        # Prefer fenced or first balanced object.
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
        if m:
            text = m.group(1)
        else:
            text = _first_json_object(text)
        if not text:
            return {}
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _rule_based_structured(
        self,
        result: Mapping[str, Any],
        evidence: Mapping[str, Any],
        candidates: Sequence[ProductCandidate],
    ) -> Dict[str, Any]:
        baseline = _baseline_final(result)
        stack = _detect_vertical_price_stack(result)
        product_name = _clean_product_name(str(baseline.get("product_name") or ""))
        best = candidates[0] if candidates else None
        match_status = "not_used"
        if best is not None:
            if best.score >= self.min_catalog_match_score:
                match_status = "matched"
                product_name = best.name or product_name
            else:
                match_status = "not_found"
        elif self.catalog.enabled:
            match_status = "not_found"

        price_candidates = _baseline_price_candidates(result)
        main_price = baseline.get("main_price")
        card_price = baseline.get("card_price")
        no_card_price = baseline.get("no_card_price")
        price_status = "ok" if main_price else "not_found"
        review_reasons: List[str] = []
        needs_review = False

        if stack.get("detected"):
            # Do not trust pairs like 89.80 formed from two separate vertical rows.
            price_candidates = stack.get("price_candidates", price_candidates)
            if stack.get("baseline_price_suspicious"):
                main_price = None
                card_price = None
                price_status = "ambiguous"
                needs_review = True
                review_reasons.append("vertical_price_stack_do_not_merge_rows")

        ret = {
            "product_name": product_name or None,
            "product_match": {
                "status": match_status,
                "item_id": best.item_id if best and match_status == "matched" else None,
                "catalog_name": best.name if best and match_status == "matched" else None,
                "score": round(float(best.score), 4) if best else 0.0,
            },
            "main_price": _normalize_price_value(main_price),
            "card_price": _normalize_price_value(card_price),
            "no_card_price": _normalize_price_value(no_card_price),
            "old_price": _normalize_price_value(baseline.get("old_price")),
            "unit": baseline.get("unit"),
            "volume": _extract_volume(product_name + " " + str(evidence.get("all_text", ""))),
            "scale_number": baseline.get("scale_number"),
            "barcode_text_raw": baseline.get("barcode_text_raw"),
            "discount_percent_raw": baseline.get("discount_percent_raw"),
            "promo_condition": baseline.get("promo_condition"),
            "price_candidates": price_candidates,
            "price_selection_status": price_status,
            "needs_review": needs_review,
            "review_reasons": review_reasons,
            "confidence": 0.72 if not needs_review else 0.52,
        }
        return enrich_final_fields(result, ret)

    def _postprocess_structured(
        self,
        result: Mapping[str, Any],
        structured: Mapping[str, Any],
        candidates: Sequence[ProductCandidate],
    ) -> Tuple[Dict[str, Any], List[str]]:
        warnings: List[str] = []
        baseline = _baseline_final(result)
        final = dict(structured)
        if not isinstance(final.get("product_match"), dict):
            final["product_match"] = {"status": "not_used", "item_id": None, "catalog_name": None, "score": 0.0}

        # Normalize product/catalog decision.  LLM may hallucinate item ids; validate.
        match = dict(final.get("product_match") or {})
        best = candidates[0] if candidates else None
        if best is None:
            match = {"status": "not_found" if self.catalog.enabled else "not_used", "item_id": None, "catalog_name": None, "score": 0.0}
        else:
            chosen_id = str(match.get("item_id") or "")
            chosen = next((c for c in candidates if c.item_id and c.item_id == chosen_id), None) if chosen_id else None
            if chosen is None and str(match.get("status") or "") == "matched":
                chosen = best
            if chosen is not None and chosen.score >= self.min_catalog_match_score:
                match = {"status": "matched", "item_id": chosen.item_id or None, "catalog_name": chosen.name or None, "score": round(float(chosen.score), 4)}
                # Catalog name is allowed to replace noisy OCR only when match is accepted.
                final["product_name"] = chosen.name
            elif self.catalog.enabled:
                match = {"status": "not_found", "item_id": None, "catalog_name": None, "score": round(float(best.score), 4)}
        final["product_match"] = match

        # Normalize prices and protect against vertical-stack false pairs.
        for key in ("main_price", "card_price", "no_card_price", "old_price"):
            final[key] = _normalize_price_value(final.get(key))

        stack = _detect_vertical_price_stack(result)
        review_reasons = list(final.get("review_reasons") or []) if isinstance(final.get("review_reasons"), list) else []
        if stack.get("detected"):
            existing = final.get("price_candidates")
            if not isinstance(existing, list) or not existing:
                final["price_candidates"] = stack.get("price_candidates", [])
            if stack.get("baseline_price_suspicious"):
                suspicious_price = _normalize_price_value(((result.get("prices") or {}).get("main") or {}).get("value"))
                if final.get("main_price") == suspicious_price or not final.get("main_price"):
                    final["main_price"] = None
                    final["card_price"] = None
                    final["price_selection_status"] = "ambiguous"
                    if "vertical_price_stack_do_not_merge_rows" not in review_reasons:
                        review_reasons.append("vertical_price_stack_do_not_merge_rows")
                    warnings.append("suppressed_suspicious_vertical_stack_price")

        if not final.get("product_name"):
            final["product_name"] = _clean_product_name(str(baseline.get("product_name") or "")) or None
        if "price_candidates" not in final or not isinstance(final.get("price_candidates"), list):
            final["price_candidates"] = _baseline_price_candidates(result)
        if not final.get("price_selection_status"):
            final["price_selection_status"] = "ok" if final.get("main_price") else "not_found"
        final["needs_review"] = bool(final.get("needs_review")) or final.get("price_selection_status") == "ambiguous" or match.get("status") in {"not_found", "ambiguous"}
        final["review_reasons"] = review_reasons
        final["confidence"] = _clip01(_to_float_or_none(final.get("confidence")) or (0.82 if not final["needs_review"] else 0.55))
        final = enrich_final_fields(result, final)
        return final, warnings

    def debug_info(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "model_name": self.model_name,
            "model_loaded": self.model_loaded,
            "model_path": str(self.model_path) if self.model_path else "",
            "paddlenlp_device": self.paddlenlp_device,
            "paddlenlp_dtype": self.paddlenlp_dtype,
            "paddlenlp_local_files_only": self.paddlenlp_local_files_only,
            "load_errors": list(self.load_errors),
            "load_debug": dict(self.load_debug),
            "catalog": self.catalog.debug_info(),
        }


# ---------------------------------------------------------------------------
# Evidence and deterministic guards
# ---------------------------------------------------------------------------


def _build_evidence(result: Mapping[str, Any]) -> Dict[str, Any]:
    ocr = result.get("ocr") or {}
    spatial = result.get("spatial") or {}
    lines = spatial.get("ocr_lines") or []
    full_tag = ocr.get("full_tag") or []
    if not lines and full_tag:
        lines = [
            {"text": x.get("text", ""), "conf": x.get("conf", 0.0), "bbox": _poly_to_xyxy(x.get("box"))}
            for x in full_tag
            if isinstance(x, Mapping)
        ]
    evidence = {
        "all_text": ocr.get("all_text_joined", ""),
        "zone_texts": ocr.get("zone_texts", {}),
        "ocr_lines": lines[:80],
        "spatial_price_candidates": spatial.get("price_candidates", [])[:20],
        "geometry": spatial.get("geometry", {}),
        "vertical_price_stack": _detect_vertical_price_stack(result),
    }
    return evidence


def _baseline_final(result: Mapping[str, Any]) -> Dict[str, Any]:
    return baseline_final_from_result(result)


def _baseline_price_candidates(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for p in ((result.get("spatial") or {}).get("price_candidates") or []):
        if not isinstance(p, Mapping):
            continue
        val = _normalize_price_value(p.get("value"))
        if not val or val in seen:
            continue
        seen.add(val)
        out.append({"value": val, "role": p.get("kind", "unknown"), "source": "spatial", "confidence": 0.70})
    main = ((result.get("prices") or {}).get("main") or {}).get("value")
    mp = _normalize_price_value(main)
    if mp and mp not in seen:
        out.append({"value": mp, "role": "baseline_main", "source": "prices.main", "confidence": 0.75})
    return out


def _detect_vertical_price_stack(result: Mapping[str, Any]) -> Dict[str, Any]:
    spatial = result.get("spatial") or {}
    lines = spatial.get("ocr_lines") or []
    nums: List[Dict[str, Any]] = []
    for l in lines:
        if not isinstance(l, Mapping):
            continue
        text = str(l.get("text", "") or "").strip()
        if not re.fullmatch(r"\d{2,3}", _digits_like(text)):
            continue
        bbox = l.get("bbox") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        nums.append({"text": _digits_like(text), "x1": x1, "y1": y1, "x2": x2, "y2": y2, "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2})
    nums.sort(key=lambda x: x["cy"])
    best_group: List[Dict[str, Any]] = []
    for n in nums:
        group = [m for m in nums if abs(m["cx"] - n["cx"]) <= max(24.0, 0.35 * max(1.0, m["x2"] - m["x1"]))]
        group = sorted(group, key=lambda x: x["cy"])
        if len(group) > len(best_group):
            best_group = group
    detected = len(best_group) >= 3
    baseline_price = _normalize_price_value(((result.get("prices") or {}).get("main") or {}).get("value"))
    suspicious = False
    if detected and baseline_price:
        # Price such as 89.80 is suspicious if rubles/kopeks are two full rows in the same stack.
        parts = baseline_price.split(".")
        if len(parts) == 2:
            rub, kop = parts
            texts = [g["text"] for g in best_group]
            if rub in texts and kop in texts:
                suspicious = True
    price_candidates = []
    for i, g in enumerate(best_group):
        price_candidates.append({
            "value": f"{int(g['text'])}.00",
            "role": "progressive_tier" if detected else "unknown",
            "source": "vertical_price_stack",
            "confidence": 0.62,
            "order": i,
            "bbox": [int(g["x1"]), int(g["y1"]), int(g["x2"]), int(g["y2"])],
        })
    return {
        "detected": detected,
        "baseline_price_suspicious": suspicious,
        "numbers": [{"text": g["text"], "bbox": [int(g["x1"]), int(g["y1"]), int(g["x2"]), int(g["y2"])]} for g in best_group],
        "price_candidates": price_candidates,
    }


def _collect_observed_prices(result: Mapping[str, Any]) -> List[float]:
    vals: List[float] = []
    main = ((result.get("prices") or {}).get("main") or {}).get("value")
    if _to_float_or_none(main) is not None:
        vals.append(float(main))
    for p in ((result.get("spatial") or {}).get("price_candidates") or []):
        if isinstance(p, Mapping) and _to_float_or_none(p.get("value")) is not None:
            vals.append(float(p.get("value")))
    # Include integer rows from vertical stacks, but not fake 1.xx regex candidates.
    stack = _detect_vertical_price_stack(result)
    for c in stack.get("price_candidates", []) if isinstance(stack, Mapping) else []:
        v = _to_float_or_none(c.get("value"))
        if v is not None:
            vals.append(v)
    return list(dict.fromkeys(vals))[:20]


def _catalog_query_text(result: Mapping[str, Any], evidence: Mapping[str, Any]) -> str:
    parts: List[str] = []
    for src in (result.get("parsed") or {}, (result.get("spatial") or {}).get("fields") or {}, evidence.get("zone_texts") or {}):
        if isinstance(src, Mapping):
            for k in ("product_name", "name"):
                if src.get(k):
                    parts.append(str(src.get(k)))
    lines = evidence.get("ocr_lines") or []
    for l in lines[:12]:
        if isinstance(l, Mapping):
            text = str(l.get("text", "") or "")
            if not re.fullmatch(r"[\d\W]+", text):
                parts.append(text)
    return " | ".join(parts)


def _mean_ocr_conf(result: Mapping[str, Any]) -> float:
    vals: List[float] = []
    for item in ((result.get("ocr") or {}).get("full_tag") or []):
        if isinstance(item, Mapping):
            v = _to_float_or_none(item.get("conf"))
            if v is not None:
                vals.append(v)
    return float(sum(vals) / len(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# String/price utilities
# ---------------------------------------------------------------------------


def _normalize_text_for_match(s: str) -> str:
    s = str(s or "").replace("ё", "е").replace("Ё", "Е")
    s = _latin_visual_to_cyrillic(s)
    s = s.lower()
    s = re.sub(r"[^0-9a-zа-я]+", " ", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()


def _latin_visual_to_cyrillic(s: str) -> str:
    table = str.maketrans({
        "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
        "a": "а", "b": "в", "c": "с", "e": "е", "h": "н", "k": "к", "m": "м", "o": "о", "p": "р", "t": "т", "x": "х", "y": "у",
    })
    return s.translate(table)


def _content_tokens(s: str) -> List[str]:
    stop = {"и", "в", "с", "на", "по", "для", "без", "или", "шт", "кг", "г", "л", "мл", "руб", "россия"}
    return [t for t in re.findall(r"[0-9a-zа-я]+", s, flags=re.I) if len(t) >= 2 and t not in stop]


def _text_similarity(q_norm: str, q_tokens: Sequence[str], name_norm: str) -> float:
    n_tokens = set(_content_tokens(name_norm))
    q_set = set(q_tokens)
    if not n_tokens or not q_set:
        return 0.0
    overlap = len(q_set & n_tokens) / max(1, min(len(q_set), len(n_tokens)))
    # Also reward partial token containment, important for cropped/truncated names.
    partial = 0.0
    for qt in q_set:
        if any(qt in nt or nt in qt for nt in n_tokens if len(qt) >= 4 and len(nt) >= 4):
            partial += 1.0
    partial = partial / max(1, len(q_set))
    ratio = SequenceMatcher(None, q_norm, name_norm).ratio()
    return max(0.58 * overlap + 0.27 * partial + 0.15 * ratio, ratio * 0.45)


def _clean_product_name(s: str) -> str:
    return retail_clean_product_name(s)


def _extract_volume(s: str) -> Optional[str]:
    return retail_extract_volume(s)


def _digits_like(s: str) -> str:
    repl = {"О": "0", "O": "0", "o": "0", "о": "0", "I": "1", "l": "1", "|": "1", "!": "1", "S": "5", "s": "5", "З": "3", "з": "3"}
    out = str(s or "")
    for k, v in repl.items():
        out = out.replace(k, v)
    return re.sub(r"\D+", "", out)


def _normalize_price_value(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if not math.isfinite(float(v)):
            return None
        return f"{float(v):.2f}"
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
    if "." not in s and len(s) >= 3:
        # Only compact values like 5499 -> 54.99, but keep 139 -> 139.00.
        if len(s) >= 4:
            s = s[:-2] + "." + s[-2:]
    try:
        return f"{float(s):.2f}"
    except Exception:
        return None


def _to_float_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        try:
            f = float(v)
            return f if math.isfinite(f) else None
        except Exception:
            return None
    s = str(v).strip().lower().replace(",", ".")
    if s in _EMPTY_PRICE_KEYS:
        return None
    s = re.sub(r"[^0-9.+-]", "", s)
    if not s:
        return None
    try:
        f = float(s)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def _to_bool_or_none(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "да"}:
        return True
    if s in {"false", "0", "no", "n", "нет"}:
        return False
    return None


def _decode_paddlenlp_ids(tokenizer: Any, ids: Any) -> str:
    try:
        if hasattr(tokenizer, "batch_decode"):
            return str(tokenizer.batch_decode(ids, skip_special_tokens=True)[0]).strip()
    except Exception:
        pass
    try:
        seq = ids[0] if hasattr(ids, "shape") and len(ids.shape) > 1 else ids
        if hasattr(seq, "numpy"):
            seq = seq.numpy().tolist()
        return str(tokenizer.decode(seq, skip_special_tokens=True)).strip()
    except Exception:
        return ""


def _decode_paddlenlp_new_tokens(tokenizer: Any, ids: Any, input_len: int) -> str:
    try:
        seq = ids[0] if hasattr(ids, "shape") and len(ids.shape) > 1 else ids
        if hasattr(seq, "numpy"):
            seq = seq.numpy().tolist()
        if isinstance(seq, tuple):
            seq = list(seq)
        if isinstance(seq, list) and input_len > 0 and len(seq) > input_len:
            return str(tokenizer.decode(seq[input_len:], skip_special_tokens=True)).strip()
    except Exception:
        pass
    return ""


def _generated_token_count(ids: Any, input_len: int) -> int:
    try:
        shape = getattr(ids, "shape", None)
        if shape is not None:
            total = int(shape[-1])
        else:
            seq = ids[0] if isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)) else ids
            total = len(seq) if hasattr(seq, "__len__") else 0
        return max(0, total - int(input_len)) if input_len and total > input_len else max(0, total)
    except Exception:
        return 0


def _fmt_float(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    return round(float(v), 2)


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _poly_to_xyxy(poly: Any) -> List[int]:
    try:
        pts = poly or []
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
    except Exception:
        return [0, 0, 0, 0]


def _first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return ""
