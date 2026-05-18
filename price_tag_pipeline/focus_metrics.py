# -*- coding: utf-8 -*-
"""Focus and blur metrics for price-tag track frame selection.

The module is intentionally OpenCV/Numpy-only.  It implements a compact subset
of focus measures that are useful for shelf-label videos: Laplacian variance,
Tenengrad, Brenner, spatial frequency and entropy.  The final score is computed
on price-tag specific regions instead of the whole crop so that a sharp QR code
or a sharp shelf rail does not dominate selection of the OCR reference frame.

The names of several metrics follow the focus-measure terminology used by
Pertuz-style focus metric collections: LAPV, TENG, BREN, SFRQ and HISE.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class FocusQuality:
    score: float
    global_score: float
    price_score: float
    header_score: float
    qr_score: float
    lapv: float
    teng: float
    bren: float
    sfrq: float
    hise: float
    contrast: float
    dark_ratio: float
    overexposed_ratio: float
    underexposed_ratio: float
    width: int
    height: int

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, float):
                d[k] = round(float(v), 6)
        return d


def compute_focus_quality(image: np.ndarray, *, roi_policy: str = "price_tag") -> FocusQuality:
    """Compute a price-tag oriented focus score.

    ``roi_policy='price_tag'`` uses three semantic regions:
    - lower-left/central area for the large price;
    - upper-left area for the product text;
    - upper-right area for QR diagnostics.

    The returned ``score`` deliberately gives most weight to price and header
    text, not to the QR block.  QR sharpness is returned separately.
    """
    if image is None or image.size == 0:
        return _empty_quality()
    if image.ndim == 2:
        gray = image.copy()
    else:
        gray = cv2.cvtColor(_ensure_bgr(image), cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h < 8 or w < 8:
        return _empty_quality(width=w, height=h)

    gray = _clip_border(gray, border_ratio=0.015)
    h2, w2 = gray.shape[:2]
    regions = _price_tag_regions(gray)

    g_score, g_metrics = _region_focus_score(gray)
    p_score, p_metrics = _region_focus_score(regions["price"])
    h_score, h_metrics = _region_focus_score(regions["header"])
    q_score, _ = _region_focus_score(regions["qr"])

    policy = str(roi_policy or "price_tag").lower().strip()
    if policy in {"global", "full"}:
        score = g_score
    elif policy in {"qr", "code"}:
        score = 0.58 * q_score + 0.24 * g_score + 0.18 * max(p_score, h_score)
    else:
        # Price and product text should dominate.  QR is useful as an auxiliary
        # sharpness cue only, because a sharp QR can coexist with unreadable
        # product/price text after motion blur.
        score = 0.46 * p_score + 0.34 * h_score + 0.14 * g_score + 0.06 * q_score

    sat_penalty = _saturation_penalty(gray)
    score = max(0.0, min(1.0, score * sat_penalty))

    return FocusQuality(
        score=float(score),
        global_score=float(g_score),
        price_score=float(p_score),
        header_score=float(h_score),
        qr_score=float(q_score),
        lapv=float(g_metrics.get("lapv", 0.0)),
        teng=float(g_metrics.get("teng", 0.0)),
        bren=float(g_metrics.get("bren", 0.0)),
        sfrq=float(g_metrics.get("sfrq", 0.0)),
        hise=float(g_metrics.get("hise", 0.0)),
        contrast=float(g_metrics.get("contrast", 0.0)),
        dark_ratio=float(g_metrics.get("dark_ratio", 0.0)),
        overexposed_ratio=float(np.mean(gray >= 245)),
        underexposed_ratio=float(np.mean(gray <= 8)),
        width=int(w2),
        height=int(h2),
    )


def normalize_focus_scores(metrics: Sequence[Mapping[str, Any]], *, key: str = "score") -> list[float]:
    """Robustly normalize per-track focus values to [0, 1]."""
    vals = np.asarray([_to_float(m.get(key), 0.0) for m in metrics], dtype=np.float32)
    if vals.size == 0:
        return []
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return [0.0 for _ in vals]
    lo = float(np.percentile(finite, 10))
    hi = float(np.percentile(finite, 90))
    if hi <= lo + 1e-6:
        hi = float(np.max(finite))
        lo = float(np.min(finite))
    if hi <= lo + 1e-6:
        return [0.65 if float(v) > 0 else 0.0 for v in vals]
    out = np.clip((vals - lo) / max(1e-6, hi - lo), 0.0, 1.0)
    # Keep the absolute score as a weak floor.  This avoids destroying all
    # weights in uniformly blurred tracks.
    abs_floor = np.clip(vals, 0.0, 1.0) * 0.35
    out = np.maximum(out * 0.85, abs_floor)
    return [float(x) for x in out]


def summarize_focus(metrics: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not metrics:
        return {"enabled": True, "count": 0}
    scores = np.asarray([_to_float(m.get("score"), 0.0) for m in metrics], dtype=np.float32)
    qrs = np.asarray([_to_float(m.get("qr_score"), 0.0) for m in metrics], dtype=np.float32)
    return {
        "enabled": True,
        "count": int(len(metrics)),
        "score_min": round(float(np.min(scores)), 6),
        "score_mean": round(float(np.mean(scores)), 6),
        "score_max": round(float(np.max(scores)), 6),
        "qr_score_max": round(float(np.max(qrs)), 6),
    }


def _price_tag_regions(gray: np.ndarray) -> Dict[str, np.ndarray]:
    h, w = gray.shape[:2]
    y_mid = int(round(h * 0.46))
    y_low = int(round(h * 0.36))
    x_qr = int(round(w * 0.58))
    x_text = int(round(w * 0.72))
    return {
        "price": gray[y_low:h, 0:max(8, x_text)],
        "header": gray[0:max(8, int(round(h * 0.58))), 0:max(8, x_qr)],
        "qr": gray[0:max(8, int(round(h * 0.62))), max(0, x_qr):w],
        "lower": gray[y_mid:h, :],
    }


def _region_focus_score(region: np.ndarray) -> Tuple[float, Dict[str, float]]:
    if region is None or region.size == 0:
        return 0.0, {}
    g = region
    if g.ndim == 3:
        g = cv2.cvtColor(_ensure_bgr(g), cv2.COLOR_BGR2GRAY)
    if min(g.shape[:2]) < 8:
        return 0.0, {}
    g = g.astype(np.uint8, copy=False)

    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    lapv = float(np.var(lap))
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag2 = gx * gx + gy * gy
    teng = float(np.mean(mag2))
    bren = _brenner(g)
    sfrq = _spatial_frequency(g)
    hise = _entropy(g)
    contrast = float(np.std(g.astype(np.float32))) / 64.0
    dark_ratio = float(np.mean(g < max(48, int(np.percentile(g, 20)))))

    # Saturating transforms keep the metric stable across resolutions and
    # illumination regimes.  Constants are empirical but deliberately broad.
    lap_n = _sat(lapv, 850.0)
    ten_n = _sat(teng, 18000.0)
    bren_n = _sat(bren, 260.0)
    sf_n = _sat(sfrq, 42.0)
    ent_n = max(0.0, min(1.0, (hise - 4.2) / 3.0))
    contrast_n = max(0.0, min(1.0, contrast))

    score = 0.28 * lap_n + 0.24 * ten_n + 0.18 * bren_n + 0.14 * sf_n + 0.08 * ent_n + 0.08 * contrast_n
    if dark_ratio < 0.015:
        score *= 0.82
    return float(max(0.0, min(1.0, score))), {
        "lapv": lapv,
        "teng": teng,
        "bren": bren,
        "sfrq": sfrq,
        "hise": hise,
        "contrast": contrast,
        "dark_ratio": dark_ratio,
    }


def _brenner(g: np.ndarray, d: int = 2) -> float:
    if g.shape[1] <= d:
        return 0.0
    diff_x = g[:, d:].astype(np.float32) - g[:, :-d].astype(np.float32)
    if g.shape[0] > d:
        diff_y = g[d:, :].astype(np.float32) - g[:-d, :].astype(np.float32)
        return float(0.5 * np.mean(diff_x * diff_x) + 0.5 * np.mean(diff_y * diff_y))
    return float(np.mean(diff_x * diff_x))


def _spatial_frequency(g: np.ndarray) -> float:
    gf = g.astype(np.float32)
    if gf.shape[1] > 1:
        rf = float(np.sqrt(np.mean(np.diff(gf, axis=1) ** 2)))
    else:
        rf = 0.0
    if gf.shape[0] > 1:
        cf = float(np.sqrt(np.mean(np.diff(gf, axis=0) ** 2)))
    else:
        cf = 0.0
    return float(np.sqrt(rf * rf + cf * cf))


def _entropy(g: np.ndarray) -> float:
    hist = cv2.calcHist([g], [0], None, [256], [0, 256]).reshape(-1)
    p = hist / max(1.0, float(hist.sum()))
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p))) if p.size else 0.0


def _saturation_penalty(gray: np.ndarray) -> float:
    over = float(np.mean(gray >= 245))
    under = float(np.mean(gray <= 8))
    penalty = 1.0 - 0.65 * max(0.0, over - 0.08) - 0.50 * max(0.0, under - 0.04)
    return max(0.55, min(1.0, penalty))


def _sat(x: float, scale: float) -> float:
    x = max(0.0, float(x))
    return float(x / (x + max(1e-6, float(scale))))


def _clip_border(gray: np.ndarray, *, border_ratio: float) -> np.ndarray:
    h, w = gray.shape[:2]
    bx = int(round(w * border_ratio))
    by = int(round(h * border_ratio))
    if bx <= 0 and by <= 0:
        return gray
    if h - 2 * by < 8 or w - 2 * bx < 8:
        return gray
    return gray[by:h - by, bx:w - bx]


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _empty_quality(*, width: int = 0, height: int = 0) -> FocusQuality:
    return FocusQuality(
        score=0.0,
        global_score=0.0,
        price_score=0.0,
        header_score=0.0,
        qr_score=0.0,
        lapv=0.0,
        teng=0.0,
        bren=0.0,
        sfrq=0.0,
        hise=0.0,
        contrast=0.0,
        dark_ratio=0.0,
        overexposed_ratio=0.0,
        underexposed_ratio=0.0,
        width=int(width),
        height=int(height),
    )


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except Exception:
        return default
