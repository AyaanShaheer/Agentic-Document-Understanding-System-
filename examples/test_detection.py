"""
Example script to test text detection models.
Run: python examples/test_detection.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
from PIL import Image

from src.detection import DetectorFactory
from src.utils.logger import get_logger

logger = get_logger(__name__)


def draw_boxes(image: np.ndarray, detection_result) -> np.ndarray:
    """Draw bounding boxes on image."""
    image_copy = image.copy()
    
    for box in detection_result.boxes:
        x1, y1, x2, y2 = map(int, [box.x1, box.y1, box.x2, box.y2])
        
        # Draw rectangle
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw confidence
        label = f"{box.confidence:.2f}"
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
    """Main testing function."""
    
    # Create sample image or load your own
    # For testing, create a simple image
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Add some text (simulated document)
    cv2.putText(image, "Sample Document", (100, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.putText(image, "Text Detection Test", (100, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    
    logger.info("Testing Faster-RCNN detector...")
    
    # Create detector
    detector = DetectorFactory.create_detector(
        detector_type="faster_rcnn",
        device="cpu",  # Change to "cuda" if available
        confidence_threshold=0.5
    )
    
    # Run detection
    result = detector.detect(image)
    
    logger.info(f"Detected {len(result.boxes)} text regions")
    logger.info(f"Processing time: {result.processing_time:.3f}s")
    
    # Draw results
    output_image = draw_boxes(image, result)
    
    # Save result
    output_path = Path("outputs/detection_result.jpg")
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), output_image)
    
    logger.info(f"Result saved to {output_path}")
    
    # Print detection details
    for i, box in enumerate(result.boxes):
        logger.info(
            f"Box {i+1}: "
            f"coords=({box.x1:.0f}, {box.y1:.0f}, {box.x2:.0f}, {box.y2:.0f}), "
            f"confidence={box.confidence:.3f}"
        )


if __name__ == "__main__":
    main()
