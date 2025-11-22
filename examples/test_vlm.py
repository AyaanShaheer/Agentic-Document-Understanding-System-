"""
Test script for VLM integration.
Run: python examples/test_vlm.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from PIL import Image

from src.vlm import GeminiVLM
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_test_document():
    """Create a test invoice document with more visible text."""
    # Create white background
    image = np.ones((800, 1000, 3), dtype=np.uint8) * 255
    
    # Draw text with more visibility
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Title
    cv2.putText(image, "INVOICE", (350, 100), font, 3, (0, 0, 0), 5)
    
    # Invoice details
    cv2.putText(image, "Invoice Number: INV-2025-001", (50, 200), font, 1.2, (0, 0, 0), 3)
    cv2.putText(image, "Date: November 20, 2025", (50, 250), font, 1.2, (0, 0, 0), 3)
    
    # Bill to
    cv2.putText(image, "Bill To:", (50, 350), font, 1.5, (0, 0, 0), 3)
    cv2.putText(image, "John Doe Inc.", (50, 410), font, 1.2, (50, 50, 50), 2)
    cv2.putText(image, "123 Main Street", (50, 460), font, 1, (50, 50, 50), 2)
    cv2.putText(image, "New York, NY 10001", (50, 500), font, 1, (50, 50, 50), 2)
    
    # Items header
    cv2.putText(image, "Items:", (50, 590), font, 1.3, (0, 0, 0), 3)
    
    # Items
    cv2.putText(image, "1. Product A - $250.00", (80, 640), font, 1, (50, 50, 50), 2)
    cv2.putText(image, "2. Product B - $250.00", (80, 680), font, 1, (50, 50, 50), 2)
    
    # Total
    cv2.rectangle(image, (550, 720), (950, 780), (0, 0, 0), 3)
    cv2.putText(image, "Total: $500.00", (600, 760), font, 1.5, (0, 0, 0), 3)
    
    return image


def main():
    """Main test function - Direct VLM testing without OCR pipeline."""
    
    logger.info("="*70)
    logger.info("VLM INTEGRATION TEST - Gemini Vision Document Understanding")
    logger.info("="*70)
    
    # Create test document
    logger.info("\nCreating test invoice document...")
    image = create_test_document()
    
    # Save test image
    output_path = Path("outputs/test_invoice.jpg")
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), image)
    logger.info(f"Test document saved to: {output_path}")
    
    # Initialize Gemini VLM
    logger.info("\nInitializing Gemini VLM...")
    
    try:
        vlm = GeminiVLM()
        
        # Test 1: Document Summarization
        logger.info("\n" + "="*70)
        logger.info("TEST 1: Document Summarization")
        logger.info("="*70)
        
        summary = vlm.summarize(image)
        logger.info(f"\nSummary:\n{summary}")
        
        # Test 2: Question Answering
        logger.info("\n" + "="*70)
        logger.info("TEST 2: Question Answering")
        logger.info("="*70)
        
        questions = [
            "What is the invoice number?",
            "Who is this invoice for?",
            "What is the total amount?",
            "What is the invoice date?",
            "How many items are listed?",
        ]
        
        for question in questions:
            answer = vlm.answer_question(image, question)
            logger.info(f"\nQ: {question}")
            logger.info(f"A: {answer}")
        
        # Test 3: Entity Extraction
        logger.info("\n" + "="*70)
        logger.info("TEST 3: Structured Entity Extraction")
        logger.info("="*70)
        
        # Create a minimal document layout (empty)
        from src.layout.base_layout import DocumentLayout
        empty_layout = DocumentLayout(
            entities=[],
            image_shape=(image.shape[0], image.shape[1]),
            processing_time=0.0
        )
        
        entities = vlm.extract_entities(image, empty_layout)
        logger.info(f"\nExtracted Entities:")
        
        import json
        logger.info(json.dumps(entities, indent=2))
        
        # Test 4: Document Classification
        logger.info("\n" + "="*70)
        logger.info("TEST 4: Document Classification")
        logger.info("="*70)
        
        categories = ["invoice", "receipt", "contract", "letter", "form"]
        classification = vlm.classify_document(image, categories)
        
        logger.info("\nClassification Results:")
        for category, score in classification.items():
            logger.info(f"  {category}: {score:.2%}")
        
        logger.info("\n" + "="*70)
        logger.info("✓ VLM INTEGRATION TEST COMPLETED SUCCESSFULLY")
        logger.info("="*70)
    
    except Exception as e:
        logger.error(f"\n❌ Error during VLM testing: {e}")
        logger.info("\nMake sure you have:")
        logger.info("1. Set GOOGLE_API_KEY in your .env file")
        logger.info("2. Get your API key from: https://makersuite.google.com/app/apikey")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
