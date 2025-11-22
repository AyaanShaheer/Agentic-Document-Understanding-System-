"""
Abstract base class for Vision-Language Models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from PIL import Image
import torch

from ..utils.logger import get_logger
from ..layout.base_layout import DocumentLayout

logger = get_logger(__name__)


@dataclass
class VLMResponse:
    """Container for VLM response."""
    
    text: str
    confidence: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class BaseVLM(ABC):
    """Abstract base class for Vision-Language Models."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
    ):
        """
        Initialize VLM.
        
        Args:
            model_path: Path to model or API key
            device: Device to run on
        """
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        
        logger.info(f"Initializing {self.__class__.__name__}")
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the VLM model."""
        pass
    
    @abstractmethod
    def query(
        self,
        image: Union[np.ndarray, Image.Image],
        prompt: str,
        document_layout: Optional[DocumentLayout] = None,
    ) -> VLMResponse:
        """
        Query the VLM with an image and prompt.
        
        Args:
            image: Input image
            prompt: Text prompt/question
            document_layout: Optional layout information
            
        Returns:
            VLMResponse with answer
        """
        pass
    
    def extract_entities(
        self,
        image: Union[np.ndarray, Image.Image],
        document_layout: DocumentLayout,
    ) -> Dict[str, Any]:
        """
        Extract structured entities from document.
        
        Args:
            image: Input image
            document_layout: Layout analysis results
            
        Returns:
            Dictionary of extracted entities
        """
        # Default implementation using query
        prompt = "Extract all key information from this document as structured JSON."
        response = self.query(image, prompt, document_layout)
        
        return {"extracted_text": response.text}
