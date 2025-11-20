"""
Test script for text recognition models.
Run: python examples/test_recognition.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from PIL import Image

from src.detection import FasterRCNNDetector
from src.recognition import RecognizerFactory
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_test_image():
    """Create a test image with text."""
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Add text
    cv2.putText(image, "Sample Document", (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.putText(image, "Text Recognition Test", (100, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    cv2.putText(image, "Hello World 123", (100, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return image


def draw_results(image, recognition_results):
    """Draw recognition results on image."""
    image_copy = image.copy()
    
    for result in recognition_results:
        bbox = result.bbox
        x1, y1, x2, y2 = map(int, [bbox.x1, bbox.y1, bbox.x2, bbox.y2])
        
        # Draw box
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw text
        label = f"{result.text} ({result.confidence:.2f})"
        cv2.putText(
            image_copy,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    
    return image_copy


def main():
    """Main test function."""
    
    logger.info("Creating test image...")
    image = create_test_image()
    
    # Step 1: Detect text regions
    logger.info("Step 1: Detecting text regions...")
    detector = FasterRCNNDetector(
        model_path=None,
        device="cpu",
        confidence_threshold=0.7,  # Higher threshold to reduce false positives
        pretrained=True,
    )
    
    detection_result = detector.detect(image)
    logger.info(f"Detected {len(detection_result.boxes)} regions")
    
    # Step 2: Recognize text
    logger.info("Step 2: Recognizing text with TrOCR...")
    recognizer = RecognizerFactory.create_recognizer(
        recognizer_type="trocr",
        device="cpu",
    )
    
    recognition_results = recognizer.recognize(image, detection_result.boxes)
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("RECOGNITION RESULTS:")
    logger.info("="*50)
    
    for i, result in enumerate(recognition_results, 1):
        logger.info(
            f"{i}. Text: '{result.text}' | "
            f"Confidence: {result.confidence:.3f} | "
            f"BBox: ({result.bbox.x1:.0f}, {result.bbox.y1:.0f}, "
            f"{result.bbox.x2:.0f}, {result.bbox.y2:.0f})"
        )
    
    # Draw and save
    output_image = draw_results(image, recognition_results)
    output_path = Path("outputs/recognition_result.jpg")
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), output_image)
    
    logger.info(f"\n✓ Result saved to {output_path}")


if __name__ == "__main__":
    main()
