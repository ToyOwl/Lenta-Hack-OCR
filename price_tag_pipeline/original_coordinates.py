"""Original full-frame bbox support for detected-track crop datasets.

The detected-track runner usually receives already-cropped tag images.  Their
local crop bbox is therefore ``[0, 0, crop_w, crop_h]``.  For final task output
we often need coordinates in the original video frame.  This module reads an
optional detector/tracker CSV with rows like::

    frame_idx,tr_id,xyxy
    6085,id_1,"100,200,420,310"
    5895,1,"98 198 419 312"

If the CSV is absent or a row is not found, callers should fall back to the crop
bbox.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

BBox = Tuple[int, int, int, int]


@dataclass
class OriginalCoordinateLookupResult:
    bbox: BBox
    source: str
    row: Dict[str, Any] = field(default_factory=dict)
    key: str = ""


class OriginalCoordinateMap:
    """Lookup full-frame tag bbox by ``(frame_idx, tr_id)``."""

    def __init__(self, cfg: Mapping[str, Any], *, root_dir: Path | None = None) -> None:
        self.cfg = dict(cfg or {})
        self.root_dir = Path(root_dir).resolve() if root_dir else None
        self.enabled = bool(self.cfg.get("enabled", False))
        self.csv_path_raw = str(self.cfg.get("csv_path") or self.cfg.get("path") or "").strip()
        self.csv_path: Optional[Path] = None
        self.loaded = False
        self.load_error = ""
        self.rows_total = 0
        self.rows_loaded = 0
        self.duplicates = 0
        self._map: Dict[Tuple[int, str], OriginalCoordinateLookupResult] = {}
        self._warnings: List[str] = []

        if self.enabled:
            self._load_if_possible()

    @property
    def warnings(self) -> List[str]:
        return list(self._warnings)

    def describe(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "csv_path_raw": self.csv_path_raw,
            "csv_path": str(self.csv_path) if self.csv_path else "",
            "loaded": self.loaded,
            "load_error": self.load_error,
            "rows_total": self.rows_total,
            "rows_loaded": self.rows_loaded,
            "duplicates": self.duplicates,
            "warnings": self.warnings,
        }

    def lookup(
        self,
        *,
        frame_idx: int,
        tr_id: Any,
        sequence_name: str = "",
        fallback_bbox: Sequence[int] | None = None,
    ) -> OriginalCoordinateLookupResult:
        fallback = _safe_bbox_tuple(fallback_bbox or [0, 0, 1, 1])
        if not self.enabled:
            return OriginalCoordinateLookupResult(bbox=fallback, source="crop_bbox_disabled")
        if not self.loaded:
            return OriginalCoordinateLookupResult(bbox=fallback, source="crop_bbox_no_csv")

        candidates = _track_id_candidates(tr_id, sequence_name=sequence_name)
        for cand in candidates:
            key = (int(frame_idx), cand)
            if key in self._map:
                res = self._map[key]
                return OriginalCoordinateLookupResult(bbox=res.bbox, source="csv", row=dict(res.row), key=f"{frame_idx}:{cand}")
        return OriginalCoordinateLookupResult(bbox=fallback, source="crop_bbox_not_found")

    def _load_if_possible(self) -> None:
        path = _resolve_csv_path(self.csv_path_raw, self.root_dir, self.cfg)
        if path is None or not path.exists():
            self.csv_path = path
            self.loaded = False
            self.load_error = "csv_not_found" if self.csv_path_raw else "csv_path_empty"
            if self.csv_path_raw:
                self._warnings.append(f"original_coordinates_csv_not_found:{self.csv_path_raw}")
            return
        self.csv_path = path

        encoding = str(self.cfg.get("encoding") or "utf-8-sig")
        delimiter = str(self.cfg.get("delimiter") or "auto")
        with path.open("r", encoding=encoding, newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            if delimiter.lower() == "auto":
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
                except Exception:
                    dialect = csv.excel
            else:
                dialect = csv.excel
                dialect.delimiter = delimiter
            reader = csv.DictReader(f, dialect=dialect)
            if not reader.fieldnames:
                self.load_error = "empty_or_headerless_csv"
                return
            for row in reader:
                self.rows_total += 1
                parsed = self._parse_row(row)
                if parsed is None:
                    continue
                frame_idx, tr_id, bbox = parsed
                for cand in _track_id_candidates(tr_id):
                    key = (frame_idx, cand)
                    if key in self._map:
                        self.duplicates += 1
                        if bool(self.cfg.get("keep_first_duplicate", True)):
                            continue
                    self._map[key] = OriginalCoordinateLookupResult(bbox=bbox, source="csv", row=dict(row), key=f"{frame_idx}:{cand}")
                self.rows_loaded += 1
        self.loaded = bool(self._map)
        if not self.loaded:
            self.load_error = "no_valid_rows"

    def _parse_row(self, row: Mapping[str, Any]) -> Optional[Tuple[int, str, BBox]]:
        frame_col = _first_existing_col(row, self.cfg.get("frame_col") or self.cfg.get("frame_idx_col") or "frame_idx", aliases=("frame_idx", "frame_index", "frame", "idx"))
        track_col = _first_existing_col(row, self.cfg.get("track_col") or self.cfg.get("tr_id_col") or "tr_id", aliases=("tr_id", "track_id", "track", "id", "object_id"))
        xyxy_col = _first_existing_col(row, self.cfg.get("xyxy_col") or self.cfg.get("bbox_col") or "xyxy", aliases=("xyxy", "bbox", "box"))
        if not frame_col or not track_col:
            if bool(self.cfg.get("strict", False)):
                raise ValueError("original-coordinates CSV must contain frame_idx and tr_id columns")
            return None
        frame = _parse_int(row.get(frame_col))
        tr_id = str(row.get(track_col) or "").strip()
        if frame is None or not tr_id:
            return None

        bbox: Optional[BBox] = None
        if xyxy_col:
            xyxy_raw = row.get(xyxy_col)
            # Common malformed CSV case: header is frame_idx,tr_id,xyxy, but
            # rows are not quoted: 6085,id_1,10,20,110,220.  DictReader keeps
            # the extra fields under key None.  Reassemble them into xyxy.
            extra = row.get(None) if isinstance(row, dict) else None
            if isinstance(extra, list) and extra:
                xyxy_raw = ",".join([str(xyxy_raw or "")] + [str(x) for x in extra])
            bbox = parse_xyxy(xyxy_raw)
        if bbox is None:
            bbox = _parse_bbox_from_split_columns(row, self.cfg)
        if bbox is None:
            return None
        return int(frame), tr_id, bbox


def build_original_coordinate_map(dt_cfg: Mapping[str, Any], *, root_dir: Path | None = None) -> OriginalCoordinateMap:
    raw = dt_cfg.get("original_coordinates", {}) if isinstance(dt_cfg.get("original_coordinates"), Mapping) else {}
    cfg = {
        "enabled": False,
        "csv_path": "",
        "frame_col": "frame_idx",
        "track_col": "tr_id",
        "xyxy_col": "xyxy",
        "delimiter": "auto",
        "encoding": "utf-8-sig",
        "strict": False,
        "fallback_to_crop_bbox": True,
        "keep_first_duplicate": True,
        "auto_find_names": ["original_coordinates.csv", "detections.csv", "tracks.csv", "bboxes.csv"],
    }
    cfg.update(dict(raw))
    # Backward-compatible flat aliases.
    for old_key, new_key in (
        ("original_coords_csv", "csv_path"),
        ("coordinates_csv", "csv_path"),
        ("detections_csv", "csv_path"),
    ):
        if old_key in dt_cfg:
            cfg[new_key] = dt_cfg.get(old_key)
            cfg["enabled"] = True
    if str(cfg.get("csv_path") or "").strip():
        cfg["enabled"] = bool(cfg.get("enabled", True))
    return OriginalCoordinateMap(cfg, root_dir=root_dir)


def parse_xyxy(value: Any) -> Optional[BBox]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            return _safe_bbox_tuple(value[:4])
        except Exception:
            return None
    raw = str(value).strip()
    if not raw:
        return None
    # Accept JSON/list strings and arbitrary separators: "1,2,3,4", "[1 2 3 4]".
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 4:
            return _safe_bbox_tuple(parsed[:4])
    except Exception:
        pass
    nums = re.findall(r"-?\d+(?:\.\d+)?", raw)
    if len(nums) < 4:
        return None
    try:
        return _safe_bbox_tuple([float(x) for x in nums[:4]])
    except Exception:
        return None


def _parse_bbox_from_split_columns(row: Mapping[str, Any], cfg: Mapping[str, Any]) -> Optional[BBox]:
    aliases = [
        ("x_min", "y_min", "x_max", "y_max"),
        ("xmin", "ymin", "xmax", "ymax"),
        ("x1", "y1", "x2", "y2"),
        ("left", "top", "right", "bottom"),
    ]
    custom = cfg.get("split_bbox_cols")
    if isinstance(custom, (list, tuple)) and len(custom) >= 4:
        aliases.insert(0, tuple(str(x) for x in custom[:4]))  # type: ignore[arg-type]
    lower = {str(k).lower(): k for k in row.keys()}
    for names in aliases:
        actual = [lower.get(n.lower()) for n in names]
        if all(actual):
            vals = [row.get(k) for k in actual if k]
            try:
                return _safe_bbox_tuple(vals)
            except Exception:
                continue
    return None


def _resolve_csv_path(raw: str, root_dir: Path | None, cfg: Mapping[str, Any]) -> Optional[Path]:
    raw = str(raw or "").strip()
    candidates: List[Path] = []
    if raw:
        p = Path(raw).expanduser()
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.extend([Path.cwd() / p])
            if root_dir is not None:
                candidates.extend([root_dir / p, root_dir.parent / p])
    elif bool(cfg.get("auto_find", False)) and root_dir is not None:
        for name in cfg.get("auto_find_names") or []:
            candidates.extend([root_dir / str(name), root_dir.parent / str(name)])
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    return candidates[0].resolve() if candidates else None


def _first_existing_col(row: Mapping[str, Any], preferred: Any, *, aliases: Sequence[str]) -> str:
    lower = {str(k).lower(): str(k) for k in row.keys()}
    names = [str(preferred)] if preferred else []
    names.extend(str(a) for a in aliases)
    for name in names:
        if name in row:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return ""


def _parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return int(float(raw))
    except Exception:
        m = re.search(r"-?\d+", raw)
        if m:
            try:
                return int(m.group(0))
            except Exception:
                return None
    return None


def _safe_bbox_tuple(values: Sequence[Any]) -> BBox:
    vals = [int(round(float(x))) for x in list(values)[:4]]
    if len(vals) < 4:
        raise ValueError("bbox must contain 4 numbers")
    x1, y1, x2, y2 = vals
    # Keep coordinates valid even if detector exports reversed corners.
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return int(x1), int(y1), int(x2), int(y2)


def _track_id_candidates(tr_id: Any, *, sequence_name: str = "") -> List[str]:
    raw = str(tr_id or "").strip()
    parts = [raw]
    if "/" in raw or "\\" in raw:
        parts.append(re.split(r"[/\\]", raw)[-1])
    if sequence_name:
        parts.extend([f"{sequence_name}/{raw}", f"{sequence_name}\\{raw}"])

    out: List[str] = []
    for item in parts:
        item = str(item or "").strip()
        if not item:
            continue
        norm = item.lower()
        variants = [norm]
        if norm.startswith("id_"):
            variants.append(norm[3:])
        if norm.startswith("track_"):
            variants.append(norm[6:])
        m = re.fullmatch(r"0*(\d+)", norm)
        if m:
            variants.extend([m.group(1), f"id_{m.group(1)}", f"track_{m.group(1)}"])
        else:
            m2 = re.search(r"(\d+)", norm)
            if m2:
                variants.extend([m2.group(1), f"id_{m2.group(1)}", f"track_{m2.group(1)}"])
        for v in variants:
            v = v.strip().lower()
            if v and v not in out:
                out.append(v)
    return out
