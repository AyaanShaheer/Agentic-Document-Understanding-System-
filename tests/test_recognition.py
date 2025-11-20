"""
Unit tests for text recognition models.
"""

import pytest
import numpy as np
from PIL import Image

from src.recognition import RecognizerFactory, TrOCRRecognizer, CRNNRecognizer
from src.detection.base_detector import BoundingBox


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    image = np.ones((100, 400, 3), dtype=np.uint8) * 255
    return Image.fromarray(image)


@pytest.fixture
def sample_bbox():
    """Create a sample bounding box."""
    return BoundingBox(x1=10, y1=10, x2=390, y2=90, confidence=0.9)


class TestTrOCRRecognizer:
    """Test TrOCR recognizer."""
    
    @pytest.mark.slow
    def test_initialization(self):
        """Test recognizer initialization."""
        recognizer = TrOCRRecognizer(device="cpu")
        assert recognizer.model is not None
        assert recognizer.processor is not None
    
    @pytest.mark.slow
    def test_recognize_single(self, sample_image):
        """Test single image recognition."""
        recognizer = TrOCRRecognizer(device="cpu")
        text = recognizer.recognize_single(sample_image)
        assert isinstance(text, str)


class TestCRNNRecognizer:
    """Test CRNN recognizer."""
    
    def test_initialization(self):
        """Test CRNN initialization."""
        recognizer = CRNNRecognizer(device="cpu")
        assert recognizer.model is not None
        assert recognizer.num_classes > 0
    
    def test_recognize_single(self, sample_image):
        """Test single image recognition."""
        recognizer = CRNNRecognizer(device="cpu")
        text = recognizer.recognize_single(sample_image)
        assert isinstance(text, str)


class TestRecognizerFactory:
    """Test recognizer factory."""
    
    @pytest.mark.slow
    def test_create_trocr(self):
        """Test creating TrOCR recognizer."""
        recognizer = RecognizerFactory.create_recognizer(
            recognizer_type="trocr",
            device="cpu"
        )
        assert isinstance(recognizer, TrOCRRecognizer)
    
    def test_create_crnn(self):
        """Test creating CRNN recognizer."""
        recognizer = RecognizerFactory.create_recognizer(
            recognizer_type="crnn",
            device="cpu"
        )
        assert isinstance(recognizer, CRNNRecognizer)
    
    def test_list_available(self):
        """Test listing available recognizers."""
        recognizers = RecognizerFactory.list_available_recognizers()
        assert "trocr" in recognizers
        assert "crnn" in recognizers
