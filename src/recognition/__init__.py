"""Text recognition models and utilities."""

from .base_recognizer import BaseRecognizer, RecognitionResult
from .trocr_recognizer import TrOCRRecognizer, TrOCRHandwrittenRecognizer
from .crnn_recognizer import CRNNRecognizer, CRNN
from .recognizer_factory import RecognizerFactory

__all__ = [
    "BaseRecognizer",
    "RecognitionResult",
    "TrOCRRecognizer",
    "TrOCRHandwrittenRecognizer",
    "CRNNRecognizer",
    "CRNN",
    "RecognizerFactory",
]
