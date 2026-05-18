"""
Debug plots for detected-track OCR aggregation.

The plotting goal is manual review, not scientific visualization.  Keep plots
compact: a reviewer must quickly see price stability, product evidence, QR/DB
support and why a track was accepted or sent to review.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence
import textwrap


def write_track_debug_plots(track: Mapping[str, Any], out_dir: Path, *, name_prefix: str) -> dict[str, str]:
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
        _write_timeline_plot(plt, track, observations, out_dir, name_prefix, title_name, status, warnings, paths)
    _write_votes_plot(plt, votes, out_dir, name_prefix, title_name, status, paths)
    return paths


def _write_timeline_plot(plt: Any, track: Mapping[str, Any], observations: Sequence[Mapping[str, Any]], out_dir: Path, name_prefix: str, title_name: str, status: str, warnings: Sequence[str], paths: dict[str, str]) -> None:
    from matplotlib.ticker import MaxNLocator

    frames = [int(o.get("frame_index") or 0) for o in observations]
    scores = [_as_float(o.get("score"), 0.0) or 0.0 for o in observations]
    prices = [str(o.get("main_price") or "") for o in observations]
    price_vals = [_as_float(p, None) for p in prices]
    products = [str(o.get("product_name") or "") for o in observations]
    ocr_texts = [str(o.get("ocr_all_text_joined") or "") for o in observations]
    templates = [str(o.get("template_name") or "") for o in observations]
    discounts = [_discount_label(o) for o in observations]
    stockouts = [str(o.get("stock_status") or "") == "out_of_stock" for o in observations]
    selected = [1.0 if _as_bool(o.get("visual_consensus_selected")) else 0.0 for o in observations]
    ocr_knn = [_as_float(o.get("ocr_consensus_sim_to_reference"), 0.0) or 0.0 for o in observations]
    db_items = [str(o.get("db_match_item_id") or "") for o in observations]
    db_status = [str(o.get("db_match_status") or "") for o in observations]

    fig_w = max(10.5, min(15.5, 0.58 * max(10, len(frames))))
    fig_h = 7.2
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.35, 1.05, 1.65])
    ax_score = fig.add_subplot(gs[0, 0])
    ax_evidence = fig.add_subplot(gs[1, 0], sharex=ax_score)
    ax_text = fig.add_subplot(gs[2, 0], sharex=ax_score)

    # Background template spans.
    cmap = plt.get_cmap("tab20")
    tmpl_colors = {t: cmap(i % 20) for i, t in enumerate(dict.fromkeys(t for t in templates if t))}
    for seg in _contiguous_template_segments(frames, templates):
        color = tmpl_colors.get(seg["template"])
        if color is None:
            continue
        ax_score.axvspan(seg["left"], seg["right"], color=color, alpha=0.09, linewidth=0)
        label = _compact_template(seg["template"])
        ax_score.text((seg["left"] + seg["right"]) * 0.5, 1.035, label, ha="center", va="bottom", fontsize=8, color=color, transform=ax_score.get_xaxis_transform())

    ax_score.plot(frames, scores, marker="o", linewidth=2.2, markersize=5.6, label="frame score")
    ax_score.fill_between(frames, scores, [0.0] * len(scores), alpha=0.07)
    ax_score.set_ylabel("frame score")
    ax_score.set_ylim(0, 1.10)
    ax_score.grid(True, alpha=0.22)
    ax_score.xaxis.set_major_locator(MaxNLocator(integer=True))
    best = track.get("best_observation") if isinstance(track.get("best_observation"), Mapping) else {}
    best_frame = _safe_int(best.get("frame_index"), None)
    if best_frame is not None:
        ax_score.axvline(best_frame, linestyle="--", linewidth=1.0, alpha=0.55)
        ax_score.text(best_frame, 1.00, "best", ha="center", va="top", fontsize=8)

    if any(v is not None for v in price_vals):
        xs = [x for x, v in zip(frames, price_vals) if v is not None]
        ys = [float(v) for v in price_vals if v is not None]
        ax_price = ax_score.twinx()
        ax_price.plot(xs, ys, marker="s", linewidth=1.6, markersize=5.0, alpha=0.82, label="price")
        ax_price.set_ylabel("price")
        if ys and len(set(round(y, 2) for y in ys)) == 1:
            y = ys[0]
            pad = max(1.0, abs(y) * 0.08)
            ax_price.set_ylim(y - pad, y + pad)
        for x, y, p in zip(frames, price_vals, prices):
            if y is None or not p:
                continue
            ax_price.annotate(_short(p, 12), (x, float(y)), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)

    has_ocr = [1.0 if _clean_ocr_for_label(t, p, pr) else 0.0 for t, p, pr in zip(ocr_texts, prices, products)]
    has_price = [1.0 if p else 0.0 for p in prices]
    has_product = [1.0 if p else 0.0 for p in products]
    has_discount = [1.0 if d else 0.0 for d in discounts]
    has_stockout = [1.0 if f else 0.0 for f in stockouts]
    has_db = [1.0 if (item or st in {"matched", "structured_csv", "weak_match", "ambiguous"}) else 0.0 for st, item in zip(db_status, db_items)]
    rows = [has_ocr, has_price, has_product, has_discount, has_stockout, selected, ocr_knn, has_db]
    labels = ["ocr", "price", "product", "discount", "stockout", "cluster", "ocr_knn", "db"]
    ax_evidence.imshow(rows, aspect="auto", vmin=0, vmax=1, interpolation="nearest", cmap="YlGn")
    ax_evidence.set_yticks(range(len(labels)))
    ax_evidence.set_yticklabels(labels)
    ax_evidence.set_xticks(range(len(frames)))
    ax_evidence.set_xticklabels([str(x) for x in frames], rotation=0 if len(frames) <= 16 else 45, ha="right" if len(frames) > 16 else "center")
    for col, (p, prod, disc, db, knn) in enumerate(zip(prices, products, discounts, db_items, ocr_knn)):
        if p:
            ax_evidence.text(col, 1, _short(p, 9), ha="center", va="center", fontsize=7)
        if prod:
            ax_evidence.text(col, 2, "✓", ha="center", va="center", fontsize=8)
        if disc:
            ax_evidence.text(col, 3, _short(disc, 8), ha="center", va="center", fontsize=7)
        if db:
            ax_evidence.text(col, 7, _short(db, 9), ha="center", va="center", fontsize=7)
        if knn > 0:
            ax_evidence.text(col, 6, f"{knn:.2f}", ha="center", va="center", fontsize=7)

    ax_text.axis("off")
    lines = []
    for idx, (fr, p, prod, raw, disc) in enumerate(zip(frames, prices, products, ocr_texts, discounts)):
        raw_clean = _clean_ocr_for_label(raw, p, prod)
        bits = [f"f{fr}"]
        if p:
            bits.append(f"price={p}")
        if disc:
            bits.append(f"disc={disc}")
        if prod:
            bits.append(f"prod={_short(prod, 34)}")
        if raw_clean:
            bits.append(f"ocr={_short(raw_clean, 48)}")
        if len(bits) > 1:
            lines.append(" | ".join(bits))
    if not lines:
        lines = ["Нет компактных OCR/price/product наблюдений для отображения."]
    max_lines = 7
    text = "\n".join(lines[:max_lines] + ([f"... +{len(lines) - max_lines} rows"] if len(lines) > max_lines else []))
    ax_text.text(0.01, 0.98, text, transform=ax_text.transAxes, ha="left", va="top", fontsize=8.4, family="monospace")

    warn_short = _short(",".join(warnings), 130)
    title = f"Track evidence: {title_name} | status={status}" + (f" | {warn_short}" if warn_short else "")
    fig.suptitle("\n".join(textwrap.wrap(title, width=105)), fontsize=13, fontweight="bold")
    path = out_dir / f"{name_prefix}__timeline.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths["timeline"] = str(path)


def _write_votes_plot(plt: Any, votes: Mapping[str, Any], out_dir: Path, name_prefix: str, title_name: str, status: str, paths: dict[str, str]) -> None:
    vote_items = []
    for field_name in ("main_price", "product_name", "discount_percent_raw", "stock_status"):
        v = votes.get(field_name) if isinstance(votes.get(field_name), Mapping) else {}
        total = sum(_as_float(c.get("weight"), 0.0) or 0.0 for c in (v.get("candidates") or []) if isinstance(c, Mapping))
        for c in v.get("candidates") or []:
            if not isinstance(c, Mapping):
                continue
            value = str(c.get("value") or "")
            if not value:
                continue
            weight = _as_float(c.get("weight"), 0.0) or 0.0
            pct = 100.0 * weight / total if total > 1e-9 else 0.0
            label = f"{field_name}: {_short(value, 36)}"
            vote_items.append((label, weight, pct, field_name))
    if not vote_items:
        return
    vote_items = sorted(vote_items, key=lambda x: x[1], reverse=True)[:14]
    labels = [x[0] for x in vote_items][::-1]
    weights = [x[1] for x in vote_items][::-1]
    pcts = [x[2] for x in vote_items][::-1]
    fig = plt.figure(figsize=(10.6, max(3.7, 0.46 * len(labels) + 1.1)), constrained_layout=True)
    ax = fig.add_subplot(111)
    bars = ax.barh(labels, weights)
    ax.set_xlabel("weighted vote")
    ax.set_title(f"Aggregation votes: {title_name} | status={status}", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.22)
    max_w = max(weights) if weights else 1.0
    ax.set_xlim(0, max_w * 1.30 + 1e-6)
    for bar, w, pct in zip(bars, weights, pcts):
        ax.text(bar.get_width() + max_w * 0.025, bar.get_y() + bar.get_height() / 2, f"{w:.2f} ({pct:.0f}%)", va="center", fontsize=9)
    path = out_dir / f"{name_prefix}__votes.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths["votes"] = str(path)


def write_global_debug_plots(tracks: Sequence[Mapping[str, Any]], out_dir: Path) -> dict[str, str]:
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
        price_ratios.append(_as_float(price.get("winner_ratio"), 1.0) or 0.0)
        unique_prices.append(_as_float(price.get("unique_count"), 0.0) or 0.0)
        stockout_ratios.append(_as_float(stockout.get("frame_ratio"), 0.0) or 0.0)
        false_scores.append(_as_float((t.get("validation") or {}).get("score") if isinstance(t.get("validation"), Mapping) else 0.0, 0.0) or 0.0)
        obs_counts.append(_as_float(t.get("num_observations"), 0.0) or 0.0)

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
        return float(v)
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
    for val in [price, product]:
        val = str(val or "").strip()
        if val:
            s = s.replace(val, " ")
    s = " ".join(s.split())
    return s


def _discount_label(obs: Mapping[str, Any]) -> str:
    for k in ("discount_percent_raw", "discount_amount", "discount"):
        v = str(obs.get(k) or "").strip()
        if v:
            return v
    return ""


def _compact_template(t: str) -> str:
    t = str(t or "")
    mapping = {
        "shelf_red_promo": "red_promo",
        "progressive_yellow": "progressive",
        "hanging_yellow_promo_large": "yellow_promo",
    }
    return mapping.get(t, t[:18])


def _contiguous_template_segments(frames: Sequence[int], templates: Sequence[str]) -> list[dict[str, Any]]:
    if not frames:
        return []
    out: list[dict[str, Any]] = []
    start = 0
    for i in range(1, len(frames) + 1):
        if i == len(frames) or templates[i] != templates[start]:
            left = frames[start] - 0.5 if start == 0 else (frames[start - 1] + frames[start]) * 0.5
            right = frames[i - 1] + 0.5 if i == len(frames) else (frames[i - 1] + frames[i]) * 0.5
            out.append({"start_idx": start, "end_idx": i - 1, "template": templates[start], "left": left, "right": right})
            start = i
    return out
