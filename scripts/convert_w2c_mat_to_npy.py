"""
convert_w2c_mat_to_npy.py

Convert classic W2C / Color Names LUT from MATLAB .mat to NumPy .npy/.npz.

Expected classic W2C LUT shape:
  (32768, 11)

Color channel order used by the pipeline:
  black, blue, brown, gray, green, orange, pink, purple, red, white, yellow

Examples:
  python scripts/convert_w2c_mat_to_npy.py --src models/color_names/w2c.mat --dst models/color_names/w2c.npy
  python scripts/convert_w2c_mat_to_npy.py --src ColorNaming/w2c.mat --dst models/color_names/w2c.npz --npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np


EXPECTED_SHAPE = (32768, 11)
W2C_KEYS = ("w2c", "W2C", "colorNames", "ColorNames")


def load_mat(path: Path) -> Dict[str, Any]:
    try:
        import scipy.io  # type: ignore
    except Exception as exc:
        raise RuntimeError("scipy is required to read .mat files. Install it with: pip install scipy") from exc
    return scipy.io.loadmat(str(path))


def find_w2c_array(mat: Dict[str, Any], key: str = "") -> np.ndarray:
    if key:
        if key not in mat:
            raise KeyError(f"Key '{key}' not found. Available keys: {sorted(k for k in mat.keys() if not k.startswith('__'))}")
        arr = np.asarray(mat[key])
    else:
        arr = None
        for k in W2C_KEYS:
            if k in mat:
                arr = np.asarray(mat[k])
                break
        if arr is None:
            candidates = []
            for k, v in mat.items():
                if k.startswith("__"):
                    continue
                a = np.asarray(v)
                if a.ndim == 2 and (a.shape == EXPECTED_SHAPE or a.shape == EXPECTED_SHAPE[::-1]):
                    candidates.append((k, a))
            if not candidates:
                available = {k: np.asarray(v).shape for k, v in mat.items() if not k.startswith("__")}
                raise KeyError(f"Could not find W2C LUT. Available arrays: {available}")
            arr = candidates[0][1]

    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape == EXPECTED_SHAPE[::-1]:
        arr = arr.T
    if arr.shape != EXPECTED_SHAPE:
        raise ValueError(f"Unexpected W2C shape: {arr.shape}. Expected {EXPECTED_SHAPE} or {EXPECTED_SHAPE[::-1]}.")
    return arr


def normalize_rows(w2c: np.ndarray) -> np.ndarray:
    w2c = np.asarray(w2c, dtype=np.float32)
    w2c = np.maximum(w2c, 0.0)
    row_sum = w2c.sum(axis=1, keepdims=True)
    return w2c / np.maximum(row_sum, 1e-12)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert W2C Color Names LUT .mat to .npy/.npz")
    p.add_argument("--src", required=True, type=str, help="Input w2c.mat path")
    p.add_argument("--dst", required=True, type=str, help="Output .npy or .npz path")
    p.add_argument("--key", default="", type=str, help="Optional MATLAB variable key. Default: auto-detect")
    p.add_argument("--npz", action="store_true", help="Save as .npz with key 'w2c' instead of .npy")
    p.add_argument("--no_normalize", action="store_true", help="Do not normalize rows to sum=1")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")

    mat = load_mat(src)
    w2c = find_w2c_array(mat, key=args.key)
    if not args.no_normalize:
        w2c = normalize_rows(w2c)

    dst.parent.mkdir(parents=True, exist_ok=True)
    if args.npz or dst.suffix.lower() == ".npz":
        np.savez_compressed(dst, w2c=w2c)
    else:
        np.save(dst, w2c)

    row_sums = w2c.sum(axis=1)
    print(f"[OK] saved: {dst}")
    print(f"     shape={w2c.shape} dtype={w2c.dtype}")
    print(f"     row_sum_min={row_sums.min():.6f} row_sum_max={row_sums.max():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
