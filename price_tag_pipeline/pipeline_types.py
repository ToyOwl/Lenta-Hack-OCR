"""
Shared data structures for price tag OCR pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Box:
    cls: str
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float = 1.0
    source: str = "unknown"

    def clip(self, w: int, h: int) -> "Box":
        return Box(
            cls=self.cls,
            x1=max(0, min(int(self.x1), max(0, w - 1))),
            y1=max(0, min(int(self.y1), max(0, h - 1))),
            x2=max(0, min(int(self.x2), w)),
            y2=max(0, min(int(self.y2), h)),
            conf=float(self.conf),
            source=self.source,
        )

    @property
    def width(self) -> int:
        return max(0, int(self.x2 - self.x1))

    @property
    def height(self) -> int:
        return max(0, int(self.y2 - self.y1))

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)

    def expand(self, w: int, h: int, px: int = 4) -> "Box":
        return Box(
            cls=self.cls,
            x1=self.x1 - px,
            y1=self.y1 - px,
            x2=self.x2 + px,
            y2=self.y2 + px,
            conf=self.conf,
            source=self.source,
        ).clip(w, h)

    def to_xyxy(self) -> List[int]:
        return [int(self.x1), int(self.y1), int(self.x2), int(self.y2)]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OCRItem:
    text: str
    conf: float
    box: Optional[List[List[float]]] = None
    zone: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecodedCode:
    kind: str
    decoded: bool
    payload: str = ""
    fmt: str = ""
    conf: float = 0.0
    bbox: Optional[List[int]] = None
    decoder: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TemplateResult:
    template_name: str
    confidence: float
    scores: Dict[str, float]
    color_features: Dict[str, Any]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
