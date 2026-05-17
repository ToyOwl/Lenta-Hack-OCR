"""Price tag OCR pipeline package."""

from .csv_corrector import StructuredCSVOCRCorrector, StructuredCatalogMatcher
from .pipeline import PriceTagPipeline
from .price_rail_splitter import PriceRailSplitter
from .template_classifier import ColorNameTemplateClassifier
from .tilt_corrector import TiltCorrector
from .preprocess_glare import GlareSuppressor
from .track_aggregator import PriceTagTrackAggregator
from .detected_tracks_dataset import DetectionTrackFolder, process_detected_tracks_dataset

__all__ = [
    "PriceTagPipeline",
    "ColorNameTemplateClassifier",
    "TiltCorrector",
    "GlareSuppressor",
    "PriceRailSplitter",
    "StructuredCSVOCRCorrector",
    "StructuredCatalogMatcher",
    "PriceTagTrackAggregator",
    "DetectionTrackFolder",
    "process_detected_tracks_dataset",
]
