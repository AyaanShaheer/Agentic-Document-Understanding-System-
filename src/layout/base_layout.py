"""
Abstract base class for document layout understanding models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch
from PIL import Image

from ..utils.logger import get_logger
from ..detection.base_detector import BoundingBox
from ..recognition.base_recognizer import RecognitionResult

logger = get_logger(__name__)


@dataclass
class LayoutEntity:
    """Represents a layout entity with text, position, and semantic label."""
    
    text: str
    bbox: BoundingBox
    label: str  # e.g., 'title', 'paragraph', 'table', 'figure'
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "bbox": self.bbox.to_dict(),
            "label": self.label,
            "confidence": self.confidence,
        }


@dataclass
class DocumentLayout:
    """Container for document layout analysis results."""
    
    entities: List[LayoutEntity]
    image_shape: Tuple[int, int]
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entities": [entity.to_dict() for entity in self.entities],
            "image_shape": self.image_shape,
            "processing_time": self.processing_time,
            "num_entities": len(self.entities),
        }
    
    def get_entities_by_label(self, label: str) -> List[LayoutEntity]:
        """Get all entities with a specific label."""
        return [e for e in self.entities if e.label == label]


class BaseLayoutAnalyzer(ABC):
    """Abstract base class for document layout analysis."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
    ):
        """
        Initialize layout analyzer.
        
        Args:
            model_path: Path to model checkpoint or HuggingFace model ID
            device: Device to run inference on
        """
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        
        logger.info(f"Initializing {self.__class__.__name__} on {self.device}")
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the layout analysis model."""
        pass
    
    @abstractmethod
    def analyze(
        self,
        image: Union[np.ndarray, Image.Image],
        ocr_results: Optional[List[RecognitionResult]] = None,
    ) -> DocumentLayout:
        """
        Analyze document layout.
        
        Args:
            image: Input image
            ocr_results: Optional pre-computed OCR results
            
        Returns:
            DocumentLayout object with layout entities
        """
        pass
