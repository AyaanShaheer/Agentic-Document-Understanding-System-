"""
Abstract base class for all text detection models.
Defines the interface that all detectors must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Tuple
from pathlib import Path
import numpy as np
import torch
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BoundingBox:
    """Represents a detected bounding box."""
    
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str = "text"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.confidence,
            "label": self.label,
        }
    
    def area(self) -> float:
        """Calculate bounding box area."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Return coordinates in xyxy format."""
        return self.x1, self.y1, self.x2, self.y2


@dataclass
class DetectionResult:
    """Container for detection results."""
    
    boxes: List[BoundingBox]
    image_shape: Tuple[int, int]  # (height, width)
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "boxes": [box.to_dict() for box in self.boxes],
            "image_shape": self.image_shape,
            "processing_time": self.processing_time,
            "num_detections": len(self.boxes),
        }


class BaseDetector(ABC):
    """Abstract base class for text detection models."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize detector.
        
        Args:
            model_path: Path to model checkpoint or HuggingFace model ID
            device: Device to run inference on ('cuda' or 'cpu')
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        logger.info(f"Initializing {self.__class__.__name__} on {self.device}")
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the detection model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def detect(
        self,
        image: Union[np.ndarray, torch.Tensor],
    ) -> DetectionResult:
        """
        Detect text regions in an image.
        
        Args:
            image: Input image (numpy array or tensor)
            
        Returns:
            DetectionResult containing bounding boxes
        """
        pass
    
    def batch_detect(
        self,
        images: List[Union[np.ndarray, torch.Tensor]],
    ) -> List[DetectionResult]:
        """
        Detect text regions in multiple images.
        
        Args:
            images: List of input images
            
        Returns:
            List of DetectionResult objects
        """
        results = []
        for image in images:
            result = self.detect(image)
            results.append(result)
        
        return results
    
    def _filter_boxes(
        self,
        boxes: List[BoundingBox],
    ) -> List[BoundingBox]:
        """
        Filter bounding boxes based on confidence threshold.
        
        Args:
            boxes: List of bounding boxes
            
        Returns:
            Filtered list of bounding boxes
        """
        filtered = [
            box for box in boxes
            if box.confidence >= self.confidence_threshold
        ]
        
        logger.debug(f"Filtered {len(boxes)} boxes to {len(filtered)}")
        return filtered
    
    def _apply_nms(
        self,
        boxes: List[BoundingBox],
        iou_threshold: float = 0.5,
    ) -> List[BoundingBox]:
        """
        Apply Non-Maximum Suppression to remove overlapping boxes.
        
        Args:
            boxes: List of bounding boxes
            iou_threshold: IoU threshold for NMS
            
        Returns:
            List of boxes after NMS
        """
        if len(boxes) == 0:
            return boxes
        
        # Convert to tensor format
        box_coords = torch.tensor([[b.x1, b.y1, b.x2, b.y2] for b in boxes])
        scores = torch.tensor([b.confidence for b in boxes])
        
        # Apply NMS
        keep_indices = torch.ops.torchvision.nms(box_coords, scores, iou_threshold)
        
        # Filter boxes
        kept_boxes = [boxes[i] for i in keep_indices]
        
        logger.debug(f"NMS: {len(boxes)} boxes -> {len(kept_boxes)} boxes")
        return kept_boxes
    
    def save_model(self, save_path: Union[str, Path]) -> None:
        """
        Save model checkpoint.
        
        Args:
            save_path: Path to save the model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_path='{self.model_path}', "
            f"device='{self.device}', "
            f"confidence_threshold={self.confidence_threshold})"
        )
