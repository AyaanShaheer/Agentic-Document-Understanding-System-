"""
TrOCR-based text recognition using HuggingFace Transformers.
State-of-the-art transformer-based OCR for both printed and handwritten text.
"""

import time
from typing import List, Union, Optional
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from .base_recognizer import BaseRecognizer, RecognitionResult
from ..detection.base_detector import BoundingBox
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TrOCRRecognizer(BaseRecognizer):
    """
    TrOCR recognizer for text recognition.
    Uses Vision Transformer encoder + Text Transformer decoder.
    """
    
    def __init__(
        self,
        model_path: str = "microsoft/trocr-base-printed",
        device: str = "cuda",
        batch_size: int = 8,
    ):
        """
        Initialize TrOCR recognizer.
        
        Args:
            model_path: HuggingFace model ID or local path
            device: Device to run on
            batch_size: Batch size for recognition
        """
        super().__init__(model_path=model_path, device=device)
        self.batch_size = batch_size
        self.processor = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load TrOCR model and processor."""
        logger.info(f"Loading TrOCR model: {self.model_path}")
        
        # Load processor and model
        self.processor = TrOCRProcessor.from_pretrained(self.model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("✓ TrOCR model loaded successfully")
    
    def recognize(
        self,
        image: Union[np.ndarray, Image.Image],
        bboxes: List[BoundingBox],
    ) -> List[RecognitionResult]:
        """
        Recognize text in detected regions.
        
        Args:
            image: Input image (numpy array or PIL Image)
            bboxes: List of bounding boxes to recognize
            
        Returns:
            List of RecognitionResult objects
        """
        if len(bboxes) == 0:
            return []
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Crop regions
        cropped_images = self._crop_regions(image, bboxes)
        
        # Recognize in batches
        results = []
        for i in range(0, len(cropped_images), self.batch_size):
            batch_images = cropped_images[i:i + self.batch_size]
            batch_bboxes = bboxes[i:i + self.batch_size]
            
            batch_results = self._recognize_batch(batch_images, batch_bboxes)
            results.extend(batch_results)
        
        logger.debug(f"Recognized {len(results)} text regions")
        return results
    
    def _crop_regions(
        self,
        image: Image.Image,
        bboxes: List[BoundingBox],
    ) -> List[Image.Image]:
        """
        Crop image regions based on bounding boxes.
        
        Args:
            image: PIL Image
            bboxes: List of bounding boxes
            
        Returns:
            List of cropped PIL Images
        """
        cropped_images = []
        
        for bbox in bboxes:
            # Ensure coordinates are within image bounds
            x1 = max(0, int(bbox.x1))
            y1 = max(0, int(bbox.y1))
            x2 = min(image.width, int(bbox.x2))
            y2 = min(image.height, int(bbox.y2))
            
            # Crop region
            cropped = image.crop((x1, y1, x2, y2))
            cropped_images.append(cropped)
        
        return cropped_images
    
    def _recognize_batch(
        self,
        images: List[Image.Image],
        bboxes: List[BoundingBox],
    ) -> List[RecognitionResult]:
        """
        Recognize text in a batch of images.
        
        Args:
            images: List of cropped images
            bboxes: Corresponding bounding boxes
            
        Returns:
            List of RecognitionResult objects
        """
        # Preprocess images
        pixel_values = self.processor(
            images=images,
            return_tensors="pt",
        ).pixel_values.to(self.device)
        
        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
        
        # Decode predictions
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        
        # Create results
        results = []
        for text, bbox in zip(generated_texts, bboxes):
            result = RecognitionResult(
                text=text,
                confidence=bbox.confidence,  # Use detection confidence as proxy
                bbox=bbox,
            )
            results.append(result)
        
        return results
    
    def recognize_single(
        self,
        image: Union[np.ndarray, Image.Image],
    ) -> str:
        """
        Recognize text in a single image (no bounding box).
        
        Args:
            image: Input image
            
        Returns:
            Recognized text string
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Preprocess
        pixel_values = self.processor(
            images=image,
            return_tensors="pt",
        ).pixel_values.to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
        
        # Decode
        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
        
        return text


class TrOCRHandwrittenRecognizer(TrOCRRecognizer):
    """TrOCR recognizer specifically for handwritten text."""
    
    def __init__(
        self,
        model_path: str = "microsoft/trocr-base-handwritten",
        device: str = "cuda",
        batch_size: int = 8,
    ):
        """Initialize handwritten text recognizer."""
        super().__init__(
            model_path=model_path,
            device=device,
            batch_size=batch_size,
        )
