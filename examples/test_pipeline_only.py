"""Test pipeline without LLM reasoning."""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from src.agents import DocumentUnderstandingTools
from src.utils.logger import get_logger

logger = get_logger(__name__)

def create_test_image():
    """Create a clear test image with better text visibility."""
    image = np.ones((800, 1000, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Title
    cv2.putText(image, "INVOICE", (350, 100), font, 3, (0, 0, 0), 5)
    
    # Invoice details
    cv2.putText(image, "Invoice Number: INV-2025-001", (50, 200), font, 1.2, (0, 0, 0), 3)
    cv2.putText(image, "Date: November 20, 2025", (50, 250), font, 1.2, (0, 0, 0), 3)
    
    # Bill to
    cv2.putText(image, "Bill To: John Doe Inc.", (50, 350), font, 1.2, (0, 0, 0), 3)
    cv2.putText(image, "123 Main Street", (50, 400), font, 1, (50, 50, 50), 2)
    
    # Total
    cv2.rectangle(image, (550, 650), (950, 750), (0, 0, 0), 3)
    cv2.putText(image, "Total: $500.00", (600, 710), font, 1.5, (0, 0, 0), 3)
    
    return image


def filter_boxes(boxes, min_area=100):
    """Filter out tiny boxes that are likely false positives."""
    filtered = []
    for box in boxes:
        width = box['x2'] - box['x1']
        height = box['y2'] - box['y1']
        area = width * height
        
        if area >= min_area and width > 10 and height > 10:
            filtered.append(box)
    
    logger.info(f"Filtered {len(boxes)} -> {len(filtered)} boxes (removed tiny detections)")
    return filtered


def main():
    """Main test function."""
    logger.info("="*70)
    logger.info("PIPELINE TEST - Document Understanding without LLM")
    logger.info("="*70)
    
    # Create test image
    logger.info("\nCreating test image...")
    image = create_test_image()
    
    # Save test image
    output_path = Path("outputs/test_pipeline.jpg")
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), image)
    logger.info(f"Test image saved to: {output_path}")
    
    # Initialize tools
    logger.info("\nInitializing tools...")
    tools = DocumentUnderstandingTools("cpu")
    
    # Step 1: Detection
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Text Detection")
    logger.info("="*70)
    
    detect_result = tools.detect_text(image)
    
    if detect_result.get('success'):
        logger.info(f"✓ Detected: {detect_result['num_regions']} regions")
        
        # Filter tiny boxes
        filtered_boxes = filter_boxes(detect_result['boxes'], min_area=500)
        logger.info(f"✓ After filtering: {len(filtered_boxes)} valid regions")
        
        # Step 2: Recognition
        logger.info("\n" + "="*70)
        logger.info("STEP 2: Text Recognition")
        logger.info("="*70)
        
        if len(filtered_boxes) > 0:
            recog_result = tools.recognize_text(image, filtered_boxes)
            
            if recog_result.get('success'):
                logger.info(f"✓ Recognized: {recog_result['num_texts']} texts")
                
                # Display recognized texts
                logger.info("\nRecognized Texts:")
                for i, text_obj in enumerate(recog_result['texts'][:10], 1):
                    logger.info(f"  {i}. '{text_obj['text']}' (confidence: {text_obj['confidence']:.2f})")
                
                # Step 3: Layout Analysis
                logger.info("\n" + "="*70)
                logger.info("STEP 3: Layout Analysis")
                logger.info("="*70)
                
                layout_result = tools.analyze_layout(image, recog_result['texts'])
                
                if layout_result.get('success'):
                    logger.info(f"✓ Analyzed: {layout_result['num_entities']} layout entities")
                    
                    # Group by label
                    labels = {}
                    for entity in layout_result['entities']:
                        label = entity['label']
                        labels[label] = labels.get(label, 0) + 1
                    
                    logger.info("\nLayout Structure:")
                    for label, count in labels.items():
                        logger.info(f"  {label}: {count} entities")
                    
                    logger.info("\nSample Entities:")
                    for i, entity in enumerate(layout_result['entities'][:5], 1):
                        logger.info(f"  {i}. [{entity['label']}] '{entity['text']}'")
                else:
                    logger.error(f"✗ Layout analysis failed: {layout_result.get('error')}")
            else:
                logger.error(f"✗ Recognition failed: {recog_result.get('error')}")
        else:
            logger.warning("No valid regions to recognize after filtering")
    else:
        logger.error(f"✗ Detection failed: {detect_result.get('error')}")
    
    logger.info("\n" + "="*70)
    logger.info("✓ PIPELINE TEST COMPLETED")
    logger.info("="*70)


if __name__ == "__main__":
    main()
