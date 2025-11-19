"""
Abstract base class for all text recognition models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
from pathlib import Path
import numpy as np
import torch
from dataclasses import dataclass

from ..utils.logger import get_logger
from ..detection.base_detector import BoundingBox

logger = get_logger(__name__)


@dataclass
class RecognitionResult:
    """Container for recognition results."""
    
    text: str
    confidence: float
    bbox: BoundingBox
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": self.bbox.to_dict(),
        }


class BaseRecognizer(ABC):
    """Abstract base class for text recognition models."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
    ):
        """Initialize recognizer."""
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        
        logger.info(f"Initializing {self.__class__.__name__} on {self.device}")
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the recognition model."""
        pass
    
    @abstractmethod
    def recognize(
        self,
        image: np.ndarray,
        bboxes: List[BoundingBox],
    ) -> List[RecognitionResult]:
        """Recognize text in detected regions."""
        pass
