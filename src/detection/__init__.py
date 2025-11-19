"""Text detection models and utilities."""

from .base_detector import BaseDetector, BoundingBox, DetectionResult
from .faster_rcnn import FasterRCNNDetector
from .detr_detector import DETRDetector
from .detector_factory import DetectorFactory

__all__ = [
    "BaseDetector",
    "BoundingBox",
    "DetectionResult",
    "FasterRCNNDetector",
    "DETRDetector",
    "DetectorFactory",
]
