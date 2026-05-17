"""
Unicode-safe image IO and filesystem helpers.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def imread_unicode(path: Path) -> Optional[np.ndarray]:
    data = np.fromfile(str(path), dtype=np.uint8)

    if data.size == 0:
        return None

    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(str(path))
    return True

def iter_images(input_dir: Path, recursive: bool = True) -> List[Path]:
    globber = input_dir.rglob if recursive else input_dir.glob
    return sorted([p for p in globber("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])

def safe_stem(path: Path) -> str:
    return re.sub(r"[^\w\-.а-яА-ЯёЁ]+", "_", path.stem, flags=re.UNICODE)
