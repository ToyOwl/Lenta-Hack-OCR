"""
Debug plots for detected-track OCR aggregation.

The plots are optional and are written only by the detected-track dataset runner.
They intentionally depend only on matplotlib + compact JSON structures so that
plots can be generated after OCR without retaining heavy per-frame artifacts.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence
import textwrap


def write_track_debug_plots(track: Mapping[str, Any], out_dir: Path, *, name_prefix: str) -> dict[str, str]:
    """Write per-track aggregation plots and return paths."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except Exception:
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    observations = [o for o in (track.get("observations") or []) if isinstance(o, Mapping)]
    votes = track.get("votes") if isinstance(track.get("votes"), Mapping) else {}
    title_name = str(track.get("track_key") or track.get("source_track_id") or name_prefix)
    status = str(track.get("status") or "")
    warnings = [str(x) for x in (track.get("warnings") or [])]

    if observations:
        frames = [int(o.get("frame_index") or 0) for o in observations]
        scores = [_as_float(o.get("score"), 0.0) for o in observations]
        prices = [str(o.get("main_price") or "") for o in observations]
        price_vals = [_as_float(p, None) for p in prices]
        products = [str(o.get("product_name") or "") for o in observations]
        templates = [str(o.get("template_name") or "") for o in observations]
        ocr_texts = [str(o.get("ocr_all_text_joined") or "") for o in observations]
        discounts = [_discount_label(o) for o in observations]
        stockouts = [str(o.get("stock_status") or "") == "out_of_stock" for o in observations]
        consensus_selected = [1.0 if _as_bool(o.get("visual_consensus_selected")) else 0.0 for o in observations]
        consensus_cluster_ids = [str(o.get("visual_consensus_cluster_id") or "") for o in observations]
        ocr_knn_support = [_as_float(o.get("ocr_consensus_knn_support"), 0.0) or 0.0 for o in observations]
        ocr_sim_ref = [_as_float(o.get("ocr_consensus_sim_to_reference"), 0.0) or 0.0 for o in observations]
        db_statuses = [str(o.get("db_match_status") or "") for o in observations]
        db_items = [str(o.get("db_match_item_id") or "") for o in observations]
        has_ocr = [1.0 if _clean_ocr_for_label(t, p, pr) else 0.0 for t, p, pr in zip(ocr_texts, prices, products)]
        has_price = [1.0 if p else 0.0 for p in prices]
        has_discount = [1.0 if d else 0.0 for d in discounts]
        has_stockout = [1.0 if f else 0.0 for f in stockouts]
        has_db = [1.0 if (st in {"matched", "structured_csv", "weak_match", "ambiguous", "price_only_candidates"} and item) else 0.0 for st, item in zip(db_statuses, db_items)]

        fig_h = 6.8 if len(frames) > 1 else 6.0
        fig_w = max(9.2, min(16.0, 0.70 * max(8, len(frames))))
        fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
        gs = fig.add_gridspec(2, 1, height_ratios=[2.55, 1.35])
        ax_score = fig.add_subplot(gs[0, 0])
        ax_evidence = fig.add_subplot(gs[1, 0], sharex=ax_score)

        cmap = plt.get_cmap("tab20")
        template_to_color = {t: cmap(i % 20) for i, t in enumerate(dict.fromkeys(t for t in templates if t))}
        for seg in _contiguous_template_segments(frames, templates):
            tmpl = seg["template"]
            color = template_to_color.get(tmpl)
            if color is None:
                continue
            ax_score.axvspan(seg["left"], seg["right"], color=color, alpha=0.105, linewidth=0)
            label = _compact_template(tmpl)
            product = _dominant_text(products[seg["start_idx"]:seg["end_idx"] + 1])
            if product:
                label += " | " + _short(product, 30)
            ax_score.text(
                (seg["left"] + seg["right"]) * 0.5,
                1.055,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
                transform=ax_score.get_xaxis_transform(),
            )

        ax_score.plot(frames, scores, marker="o", linewidth=2.0, markersize=5.5, label="score")
        ax_score.fill_between(frames, scores, [0.0] * len(scores), alpha=0.08)
        ax_score.set_ylabel("frame score")
        ax_score.set_ylim(0, 1.08)
        ax_score.grid(True, alpha=0.22)
        ax_score.xaxis.set_major_locator(MaxNLocator(integer=True))
        best = track.get("best_observation") if isinstance(track.get("best_observation"), Mapping) else {}
        best_frame = _safe_int(best.get("frame_index"), None)
        if best_frame is not None:
            ax_score.axvline(best_frame, linestyle="--", linewidth=1.0, alpha=0.45)
            ax_score.text(best_frame, 1.02, "best", ha="center", va="bottom", fontsize=8)

        ax_price = None
        if any(v is not None for v in price_vals):
            xs = [x for x, v in zip(frames, price_vals) if v is not None]
            ys = [float(v) for v in price_vals if v is not None]
            ax_price = ax_score.twinx()
            ax_price.plot(xs, ys, marker="s", linewidth=1.45, markersize=4.8, alpha=0.78, label="price")
            ax_price.set_ylabel("price")
            if ys and len(set(round(y, 2) for y in ys)) == 1:
                y = ys[0]
                pad = max(1.0, abs(y) * 0.08)
                ax_price.set_ylim(y - pad, y + pad)
            for x, y, p, raw_text, prod, disc, stock in zip(frames, price_vals, prices, ocr_texts, products, discounts, stockouts):
                if y is None:
                    continue
                label_parts = [p]
                if disc:
                    label_parts.append(disc)
                if stock:
                    label_parts.append("товар закончился")
                extra = _clean_ocr_for_label(raw_text, p, prod)
                if extra:
                    label_parts.append("ocr: " + _short(extra, 22))
                ax_price.annotate("\n".join(label_parts[:4]), (x, float(y)), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)

        for i, (x, score_value, p, prod, raw_text, disc, stock) in enumerate(zip(frames, scores, prices, products, ocr_texts, discounts, stockouts)):
            if i % max(1, len(frames) // 18 + 1) != 0 and len(frames) > 18:
                continue
            label_parts = []
            if stock:
                label_parts.append("товар закончился")
            if p and ax_price is None:
                label_parts.append(p)
            if disc:
                label_parts.append(disc)
            if prod:
                label_parts.append("prod: " + _short(prod, 24))
            raw_clean = _clean_ocr_for_label(raw_text, p, prod)
            if raw_clean:
                label_parts.append("ocr: " + _short(raw_clean, 24))
            if label_parts:
                dy = 12 if i % 2 == 0 else -27
                va = "bottom" if dy > 0 else "top"
                ax_score.annotate("\n".join(label_parts[:4]), (x, score_value), textcoords="offset points", xytext=(0, dy), ha="center", va=va, fontsize=8)

        rows = [has_ocr, has_price, has_discount, has_stockout, consensus_selected, ocr_knn_support, has_db]
        labels = ["ocr", "price", "discount", "stockout", "cluster", "ocr_knn", "db"]
        ax_evidence.imshow(rows, aspect="auto", vmin=0, vmax=1, interpolation="nearest", cmap="YlGn")
        ax_evidence.set_yticks(range(len(labels)))
        ax_evidence.set_yticklabels(labels)
        ax_evidence.set_xlabel("frame index")
        ax_evidence.set_xticks(range(len(frames)))
        ax_evidence.set_xticklabels([str(x) for x in frames], rotation=0 if len(frames) <= 14 else 45, ha="right" if len(frames) > 14 else "center")
        for col, (p, raw, prod, disc, stock, db_st, db_item) in enumerate(zip(prices, ocr_texts, products, discounts, stockouts, db_statuses, db_items)):
            ocr_short = _short(_clean_ocr_for_label(raw, p, prod), 15)
            if ocr_short:
                ax_evidence.text(col, 0, ocr_short, ha="center", va="center", fontsize=6.5)
            if p:
                ax_evidence.text(col, 1, _short(p, 10), ha="center", va="center", fontsize=7)
            if disc:
                ax_evidence.text(col, 2, _short(disc, 12), ha="center", va="center", fontsize=7)
            if stock:
                ax_evidence.text(col, 3, "out", ha="center", va="center", fontsize=7)
            if consensus_selected[col] > 0:
                cid = consensus_cluster_ids[col] or "✓"
                ax_evidence.text(col, 4, _short(str(cid), 8), ha="center", va="center", fontsize=7)
            if ocr_knn_support[col] > 0:
                ax_evidence.text(col, 5, f"{ocr_sim_ref[col]:.2f}", ha="center", va="center", fontsize=7)
            if db_item or db_st in {"matched", "structured_csv", "rejected_by_track_evidence", "weak_match", "ambiguous"}:
                db_label = db_item if db_item else db_st.replace("rejected_by_track_evidence", "rejected")
                ax_evidence.text(col, 6, _short(db_label, 12), ha="center", va="center", fontsize=7)

        warn_short = _short(",".join(warnings), 150)
        title = f"Track evidence: {title_name} | status={status}" + (f" | {warn_short}" if warn_short else "")
        fig.suptitle("\n".join(textwrap.wrap(title, width=115)), fontsize=13, fontweight="bold")
        path = out_dir / f"{name_prefix}__timeline.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths["timeline"] = str(path)

    vote_items = []
    for field_name in ("main_price", "product_name", "stock_status"):
        v = votes.get(field_name) if isinstance(votes.get(field_name), Mapping) else {}
        total = sum(_as_float(c.get("weight"), 0.0) for c in (v.get("candidates") or []) if isinstance(c, Mapping))
        for c in v.get("candidates") or []:
            if isinstance(c, Mapping):
                value = str(c.get("value") or "")
                if not value:
                    continue
                aliases = c.get("aliases") if isinstance(c.get("aliases"), list) else []
                alias_txt = ""
                if aliases:
                    alias_vals = [str(a.get("value")) for a in aliases if isinstance(a, Mapping) and a.get("value") and str(a.get("value")) != value]
                    if alias_vals:
                        alias_txt = "  ← " + ", ".join(alias_vals[:3])
                weight = _as_float(c.get("weight"), 0.0)
                pct = 100.0 * weight / total if total > 1e-9 else 0.0
                vote_items.append((f"{field_name}: {_short(value, 30)}{alias_txt}", weight, pct, field_name))
    if vote_items:
        vote_items = sorted(vote_items, key=lambda x: x[1], reverse=True)[:12]
        labels = [x[0] for x in vote_items][::-1]
        weights = [x[1] for x in vote_items][::-1]
        pcts = [x[2] for x in vote_items][::-1]
        fig = plt.figure(figsize=(10.2, max(3.4, 0.50 * len(labels) + 1.0)), constrained_layout=True)
        ax = fig.add_subplot(111)
        bars = ax.barh(labels, weights)
        ax.set_xlabel("weighted vote")
        ax.set_title(f"Aggregation votes: {title_name} | status={status}", fontsize=13, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.22)
        max_w = max(weights) if weights else 1.0
        ax.set_xlim(0, max_w * 1.22 + 1e-6)
        for bar, w, pct in zip(bars, weights, pcts):
            ax.text(bar.get_width() + max_w * 0.025, bar.get_y() + bar.get_height() / 2, f"{w:.2f} ({pct:.0f}%)", va="center", fontsize=9)
        path = out_dir / f"{name_prefix}__votes.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths["votes"] = str(path)

    return paths


def write_global_debug_plots(tracks: Sequence[Mapping[str, Any]], out_dir: Path) -> dict[str, str]:
    """Write dataset-level aggregation plots and return paths."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    tracks = [t for t in tracks if isinstance(t, Mapping)]

    statuses = Counter(str(t.get("status") or "unknown") for t in tracks)
    if statuses:
        labels = list(statuses.keys())
        values = [statuses[k] for k in labels]
        fig = plt.figure(figsize=(9.4, max(3.4, 0.44 * len(labels) + 1.0)), constrained_layout=True)
        ax = fig.add_subplot(111)
        bars = ax.barh(labels[::-1], values[::-1])
        ax.set_xlabel("track count")
        ax.set_title("Dataset status distribution", fontsize=13, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.22)
        for b, v in zip(bars, values[::-1]):
            ax.text(b.get_width() + 0.05, b.get_y() + b.get_height() / 2, str(v), va="center")
        path = out_dir / "global__status_counts.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths["status_counts"] = str(path)

    price_ratios = []
    false_scores = []
    obs_counts = []
    unique_prices = []
    stockout_ratios = []
    for t in tracks:
        diag = t.get("diagnostics") if isinstance(t.get("diagnostics"), Mapping) else {}
        price = diag.get("price") if isinstance(diag.get("price"), Mapping) else {}
        stockout = diag.get("stock_status") if isinstance(diag.get("stock_status"), Mapping) else {}
        price_ratios.append(_as_float(price.get("winner_ratio"), 1.0))
        unique_prices.append(_as_float(price.get("unique_count"), 0.0))
        stockout_ratios.append(_as_float(stockout.get("frame_ratio"), 0.0))
        false_scores.append(_as_float((t.get("validation") or {}).get("score") if isinstance(t.get("validation"), Mapping) else 0.0, 0.0))
        obs_counts.append(_as_float(t.get("num_observations"), 0.0))

    _write_hist(plt, price_ratios, out_dir / "global__price_consistency_hist.png", "Price vote consistency", "winner weight ratio", paths, "price_consistency_hist")
    _write_hist(plt, false_scores, out_dir / "global__validation_score_hist.png", "False-candidate / review score", "validation score", paths, "validation_score_hist")
    _write_hist(plt, obs_counts, out_dir / "global__track_length_hist.png", "Track length distribution", "observations per output track", paths, "track_length_hist")
    _write_hist(plt, unique_prices, out_dir / "global__unique_price_count_hist.png", "Unique price hypotheses", "unique price values per output track", paths, "unique_price_count_hist")
    _write_hist(plt, stockout_ratios, out_dir / "global__stockout_ratio_hist.png", "Stockout-service tag support", "stockout frames / track frames", paths, "stockout_ratio_hist")

    return paths


def _write_hist(plt: Any, values: Sequence[float], path: Path, title: str, xlabel: str, paths: dict[str, str], key: str) -> None:
    values = [float(v) for v in values if v is not None]
    if not values:
        return
    fig = plt.figure(figsize=(8.2, 4.5), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.hist(values, bins=min(14, max(3, len(values))))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("track count")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.22)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths[key] = str(path)


def _as_float(v: Any, default: float | None = 0.0) -> float | None:
    try:
        f = float(v)
        return f
    except Exception:
        return default


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "selected"}


def _safe_int(v: Any, default: int | None = 0) -> int | None:
    try:
        return int(v)
    except Exception:
        return default


def _short(s: Any, n: int = 32) -> str:
    s = str(s or "")
    return s if len(s) <= n else s[: max(1, n - 1)] + "…"


def _clean_ocr_for_label(raw_text: Any, price: Any = "", product: Any = "") -> str:
    s = " ".join(str(raw_text or "").replace("\n", " ").split())
    if not s:
        return ""
    p = str(price or "").strip()
    prod = str(product or "").strip()
    if p and s.replace(".", "").replace(",", "").isdigit() and p.replace(".", "").replace(",", "").startswith(s.replace(".", "").replace(",", "")):
        return ""
    if prod and s.lower() == prod.lower():
        return ""
    return s


def _compact_template(t: str) -> str:
    t = str(t or "")
    t = t.replace("shelf_", "").replace("hanging_yellow_promo_large", "yellow_large")
    t = t.replace("progressive_yellow", "progress_yellow")
    return t


def _discount_label(o: Mapping[str, Any]) -> str:
    disc = str(o.get("discount_percent_raw") or "").strip()
    old = str(o.get("old_price") or "").strip()
    promo = str(o.get("promo_condition") or "").strip()
    if disc:
        return disc
    if old:
        return "old=" + old
    if promo:
        return _short(promo, 18)
    return ""


def _dominant_text(values: Sequence[str]) -> str:
    vals = [str(v or "").strip() for v in values if str(v or "").strip()]
    if not vals:
        return ""
    counts = Counter(vals)
    return sorted(counts.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)[0][0]


def _contiguous_template_segments(frames: Sequence[int], templates: Sequence[str]) -> list[dict[str, Any]]:
    if not frames:
        return []
    segments: list[dict[str, Any]] = []
    start = 0
    cur = templates[0] if templates else ""
    for i in range(1, len(frames)):
        if (templates[i] if i < len(templates) else "") != cur:
            segments.append(_template_segment(frames, start, i - 1, cur))
            start = i
            cur = templates[i] if i < len(templates) else ""
    segments.append(_template_segment(frames, start, len(frames) - 1, cur))
    return segments


def _template_segment(frames: Sequence[int], start: int, end: int, template: str) -> dict[str, Any]:
    if len(frames) == 1:
        left, right = frames[0] - 0.5, frames[0] + 0.5
    else:
        left = frames[start] - 0.5 if start == 0 else (frames[start - 1] + frames[start]) * 0.5
        right = frames[end] + 0.5 if end == len(frames) - 1 else (frames[end] + frames[end + 1]) * 0.5
    return {"start_idx": start, "end_idx": end, "template": str(template or ""), "left": left, "right": right}
