"""
Test script for LangGraph document understanding agent.
Run: python examples/test_agent.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from PIL import Image
import json

from src.agents import DocumentUnderstandingAgent
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_test_invoice():
    """Create a test invoice."""
    image = np.ones((800, 1000, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(image, "INVOICE", (350, 100), font, 3, (0, 0, 0), 5)
    cv2.putText(image, "Invoice Number: INV-2025-001", (50, 200), font, 1.2, (0, 0, 0), 3)
    cv2.putText(image, "Date: November 20, 2025", (50, 250), font, 1.2, (0, 0, 0), 3)
    cv2.putText(image, "Bill To: John Doe Inc.", (50, 350), font, 1.2, (0, 0, 0), 3)
    cv2.putText(image, "Total: $500.00", (600, 700), font, 1.5, (0, 0, 0), 3)
    
    output_path = Path("outputs/test_invoice_agent.jpg")
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), image)
    
    return str(output_path)


def main():
    """Main test function."""
    
    logger.info("="*70)
    logger.info("LANGGRAPH AGENT TEST - Complete Document Understanding")
    logger.info("="*70)
    
    # Create test document
    logger.info("\nCreating test invoice...")
    image_path = create_test_invoice()
    logger.info(f"Test invoice saved to: {image_path}")
    
    # Initialize agent
    logger.info("\nInitializing Document Understanding Agent...")
    agent = DocumentUnderstandingAgent(device="cpu")
    
    # Test queries
    queries = [
        "What is the invoice number?",
        "What is the total amount?",
        "Who is this invoice for?",
        "What is the date of this invoice?",
    ]
    
    for i, query in enumerate(queries, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"Query {i}: {query}")
        logger.info("="*70)
        
        result = agent.process_document(image_path, query)
        
        logger.info(f"\nFinal Answer: {result.get('final_answer', 'N/A')}")
        logger.info(f"Confidence: {result.get('confidence_score', 0):.2%}")
        logger.info(f"Processing Time: {result.get('processing_time', 0):.2f}s")
        
        if result.get('errors'):
            logger.warning(f"Errors: {result['errors']}")
    
    logger.info(f"\n{'='*70}")
    logger.info("✓ AGENT TEST COMPLETED")
    logger.info("="*70)


if __name__ == "__main__":
    main()
