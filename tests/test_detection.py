"""
Unit tests for text detection models.
"""

import pytest
import numpy as np
from PIL import Image
import torch

from src.detection import DetectorFactory, FasterRCNNDetector, DETRDetector


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # Create a simple 800x600 RGB image
    image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    return image


@pytest.fixture
def sample_pil_image(sample_image):
    """Create a sample PIL image."""
    return Image.fromarray(sample_image)


class TestFasterRCNNDetector:
    """Test Faster-RCNN detector."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = FasterRCNNDetector(device="cpu")
        assert detector.model is not None
        assert detector.device == torch.device("cpu")
    
    def test_detect_numpy(self, sample_image):
        """Test detection on numpy array."""
        detector = FasterRCNNDetector(device="cpu", confidence_threshold=0.5)
        result = detector.detect(sample_image)
        
        assert result is not None
        assert hasattr(result, 'boxes')
        assert hasattr(result, 'processing_time')
        assert isinstance(result.boxes, list)
    
    def test_detect_pil(self, sample_pil_image):
        """Test detection on PIL image."""
        detector = FasterRCNNDetector(device="cpu")
        result = detector.detect(sample_pil_image)
        
        assert result is not None
        assert result.image_shape == (600, 800)
    
    def test_batch_detect(self, sample_image):
        """Test batch detection."""
        detector = FasterRCNNDetector(device="cpu")
        images = [sample_image, sample_image, sample_image]
        
        results = detector.batch_detect(images)
        
        assert len(results) == 3
        assert all(hasattr(r, 'boxes') for r in results)


class TestDETRDetector:
    """Test DETR detector."""
    
    @pytest.mark.slow
    def test_initialization(self):
        """Test DETR initialization."""
        detector = DETRDetector(device="cpu")
        assert detector.model is not None
        assert detector.processor is not None
    
    @pytest.mark.slow
    def test_detect(self, sample_pil_image):
        """Test DETR detection."""
        detector = DETRDetector(device="cpu", confidence_threshold=0.5)
        result = detector.detect(sample_pil_image)
        
        assert result is not None
        assert isinstance(result.boxes, list)


class TestDetectorFactory:
    """Test detector factory."""
    
    def test_create_faster_rcnn(self):
        """Test creating Faster-RCNN detector."""
        detector = DetectorFactory.create_detector(
            detector_type="faster_rcnn",
            device="cpu"
        )
        assert isinstance(detector, FasterRCNNDetector)
    
    @pytest.mark.slow
    def test_create_detr(self):
        """Test creating DETR detector."""
        detector = DetectorFactory.create_detector(
            detector_type="detr",
            device="cpu"
        )
        assert isinstance(detector, DETRDetector)
    
    def test_invalid_detector_type(self):
        """Test invalid detector type."""
        with pytest.raises(ValueError):
            DetectorFactory.create_detector(detector_type="invalid")
    
    def test_list_available(self):
        """Test listing available detectors."""
        detectors = DetectorFactory.list_available_detectors()
        assert "faster_rcnn" in detectors
        assert "detr" in detectors
