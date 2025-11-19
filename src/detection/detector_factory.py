"""
Factory for creating detector instances.
Provides a clean interface for instantiating different detector types.
"""

from typing import Literal, Optional
from pathlib import Path
from .base_detector import BaseDetector
from .faster_rcnn import FasterRCNNDetector
from .detr_detector import DETRDetector
from ..utils.logger import get_logger
from ..utils.config import config

logger = get_logger(__name__)

DetectorType = Literal["faster_rcnn", "detr"]


class DetectorFactory:
    """Factory for creating text detectors."""
    
    @staticmethod
    def create_detector(
        detector_type: DetectorType = "faster_rcnn",
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        **kwargs
    ) -> BaseDetector:
        """
        Create a detector instance.
        
        Args:
            detector_type: Type of detector ("faster_rcnn" or "detr")
            model_path: Path to model checkpoint (optional, uses pretrained if None)
            device: Device to run on
            confidence_threshold: Minimum confidence threshold
            **kwargs: Additional detector-specific arguments
            
        Returns:
            Initialized detector instance
        """
        # Use config defaults if not provided
        device = device or config.project.device
        confidence_threshold = confidence_threshold or config.models.detection_threshold
        
        logger.info(f"Creating detector: {detector_type}")
        
        if detector_type == "faster_rcnn":
            # For Faster-RCNN, only pass model_path if it's an actual file
            custom_model_path = None
            if model_path:
                path = Path(model_path)
                if path.exists() and path.is_file():
                    custom_model_path = model_path
            
            return FasterRCNNDetector(
                model_path=custom_model_path,
                device=device,
                confidence_threshold=confidence_threshold,
                **kwargs
            )
        
        elif detector_type == "detr":
            return DETRDetector(
                model_path=model_path or "facebook/detr-resnet-50",
                device=device,
                confidence_threshold=confidence_threshold,
            )
        
        else:
            raise ValueError(
                f"Unknown detector type: {detector_type}. "
                f"Available types: faster_rcnn, detr"
            )
    
    @staticmethod
    def list_available_detectors() -> list:
        """List all available detector types."""
        return ["faster_rcnn", "detr"]
