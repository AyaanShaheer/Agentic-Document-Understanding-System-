"""
Factory for creating recognizer instances.
"""

from typing import Literal, Optional
from .base_recognizer import BaseRecognizer
from .trocr_recognizer import TrOCRRecognizer, TrOCRHandwrittenRecognizer
from .crnn_recognizer import CRNNRecognizer
from ..utils.logger import get_logger
from ..utils.config import config

logger = get_logger(__name__)

RecognizerType = Literal["trocr", "trocr_handwritten", "crnn"]


class RecognizerFactory:
    """Factory for creating text recognizers."""
    
    @staticmethod
    def create_recognizer(
        recognizer_type: RecognizerType = "trocr",
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> BaseRecognizer:
        """
        Create a recognizer instance.
        
        Args:
            recognizer_type: Type of recognizer
            model_path: Path to model checkpoint
            device: Device to run on
            **kwargs: Additional recognizer-specific arguments
            
        Returns:
            Initialized recognizer instance
        """
        device = device or config.project.device
        
        logger.info(f"Creating recognizer: {recognizer_type}")
        
        if recognizer_type == "trocr":
            return TrOCRRecognizer(
                model_path=model_path or config.models.recognition_model_path,
                device=device,
                **kwargs
            )
        
        elif recognizer_type == "trocr_handwritten":
            return TrOCRHandwrittenRecognizer(
                model_path=model_path or "microsoft/trocr-base-handwritten",
                device=device,
                **kwargs
            )
        
        elif recognizer_type == "crnn":
            return CRNNRecognizer(
                model_path=model_path,
                device=device,
                **kwargs
            )
        
        else:
            raise ValueError(
                f"Unknown recognizer type: {recognizer_type}. "
                f"Available: trocr, trocr_handwritten, crnn"
            )
    
    @staticmethod
    def list_available_recognizers() -> list:
        """List all available recognizer types."""
        return ["trocr", "trocr_handwritten", "crnn"]
