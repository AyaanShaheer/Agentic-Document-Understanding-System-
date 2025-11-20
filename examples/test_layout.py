"""
Test script for layout analysis.
Run: python examples/test_layout.py
"""

import sys
from pathlib import Path
from typing import Tuple, List, Optional, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from PIL import Image

from src.detection import FasterRCNNDetector
from src.recognition import TrOCRRecognizer
from src.layout import LayoutLMv3Analyzer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_test_document():
    """Create a test document image."""
    image = np.ones((800, 1000, 3), dtype=np.uint8) * 255
    
    # Title
    cv2.putText(image, "INVOICE", (400, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 4)
    
    # Header info
    cv2.putText(image, "Invoice #: INV-2025-001", (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(image, "Date: Nov 20, 2025", (50, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Questions/Answers
    cv2.putText(image, "Bill To:", (50, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(image, "John Doe", (50, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
    
    # Items
    cv2.putText(image, "Description", (50, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(image, "Product A", (50, 500),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 2)
    cv2.putText(image, "Product B", (50, 550),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 2)
    
    # Total
    cv2.putText(image, "Total: $500.00", (600, 700),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    return image


def draw_layout_results(image, layout_result):
    """Draw layout analysis results."""
    image_copy = image.copy()
    
    # Color map for different labels
    color_map = {
        "B-TITLE": (255, 0, 0),    # Red
        "I-TITLE": (255, 100, 100),
        "B-HEADER": (0, 255, 0),   # Green
        "I-HEADER": (100, 255, 100),
        "B-QUESTION": (0, 0, 255), # Blue
        "I-QUESTION": (100, 100, 255),
        "B-ANSWER": (255, 255, 0), # Yellow
        "I-ANSWER": (255, 255, 100),
        "O": (128, 128, 128),      # Gray
    }
    
    for entity in layout_result.entities:
        bbox = entity.bbox
        x1, y1, x2, y2 = map(int, [bbox.x1, bbox.y1, bbox.x2, bbox.y2])
        
        color = color_map.get(entity.label, (128, 128, 128))
        
        # Draw box
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label_text = f"{entity.label}: {entity.text[:20]}"
        cv2.putText(
            image_copy,
            label_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1
        )
    
    return image_copy


def main():
    """Main test function."""
    
    logger.info("Creating test document...")
    image = create_test_document()
    
    # Step 1: Detect
    logger.info("Step 1: Detecting text...")
    detector = FasterRCNNDetector(device="cpu", confidence_threshold=0.7)
    detection_result = detector.detect(image)
    logger.info(f"Detected {len(detection_result.boxes)} regions")
    
    # Step 2: Recognize
    logger.info("Step 2: Recognizing text...")
    recognizer = TrOCRRecognizer(device="cpu")
    recognition_results = recognizer.recognize(image, detection_result.boxes)
    logger.info(f"Recognized {len(recognition_results)} texts")
    
    # Step 3: Layout Analysis
    logger.info("Step 3: Analyzing layout with LayoutLMv3...")
    layout_analyzer = LayoutLMv3Analyzer(device="cpu", apply_ocr=False)
    layout_result = layout_analyzer.analyze(image, recognition_results)
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("LAYOUT ANALYSIS RESULTS:")
    logger.info("="*60)
    
    for i, entity in enumerate(layout_result.entities, 1):
        logger.info(
            f"{i}. Label: {entity.label:15} | "
            f"Text: '{entity.text[:30]:30}' | "
            f"Conf: {entity.confidence:.3f}"
        )
    
    # Group by label
    logger.info("\n" + "="*60)
    logger.info("GROUPED BY LABEL:")
    logger.info("="*60)
    
    for label in set(e.label for e in layout_result.entities):
        entities_with_label = layout_result.get_entities_by_label(label)
        logger.info(f"\n{label} ({len(entities_with_label)} items):")
        for entity in entities_with_label[:3]:  # Show first 3
            logger.info(f"  - {entity.text[:50]}")
    
    # Save visualization
    output_image = draw_layout_results(image, layout_result)
    output_path = Path("outputs/layout_result.jpg")
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), output_image)
    
    logger.info(f"\n✓ Result saved to {output_path}")


if __name__ == "__main__":
    main()
