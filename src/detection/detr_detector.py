"""
DETR (Detection Transformer) based text detection.
Uses transformer architecture for end-to-end object detection.
"""

import time
from typing import Union, List, Optional
import numpy as np
import torch
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection

from .base_detector import BaseDetector, BoundingBox, DetectionResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DETRDetector(BaseDetector):
    """
    DETR detector for text region detection.
    Uses transformer-based architecture for end-to-end detection.
    """
    
    def __init__(
        self,
        model_path: str = "facebook/detr-resnet-50",
        device: str = "cuda",
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize DETR detector.
        
        Args:
            model_path: HuggingFace model ID or path to checkpoint
            device: Device to run on
            confidence_threshold: Minimum confidence for detections
        """
        super().__init__(
            model_path=model_path,
            device=device,
            confidence_threshold=confidence_threshold,
        )
        
        self.processor = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load DETR model from HuggingFace."""
        logger.info(f"Loading DETR model: {self.model_path}")
        
        # Load processor and model
        self.processor = DetrImageProcessor.from_pretrained(self.model_path)
        self.model = DetrForObjectDetection.from_pretrained(self.model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("✓ DETR model loaded successfully")
    
    def detect(
        self,
        image: Union[np.ndarray, Image.Image],
    ) -> DetectionResult:
        """
        Detect text regions using DETR.
        
        Args:
            image: Input image (numpy array or PIL Image)
            
        Returns:
            DetectionResult with bounding boxes
        """
        start_time = time.time()
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        original_shape = (image.height, image.width)
        
        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([original_shape]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes,
            threshold=self.confidence_threshold,
        )[0]
        
        # Parse to BoundingBox objects
        boxes = self._parse_results(results)
        
        processing_time = time.time() - start_time
        
        logger.debug(f"DETR detected {len(boxes)} regions in {processing_time:.3f}s")
        
        return DetectionResult(
            boxes=boxes,
            image_shape=original_shape,
            processing_time=processing_time,
        )
    
    def _parse_results(self, results: dict) -> List[BoundingBox]:
        """
        Parse DETR results into BoundingBox objects.
        
        Args:
            results: DETR output dictionary
            
        Returns:
            List of BoundingBox objects
        """
        boxes = []
        
        for score, label, box in zip(
            results["scores"],
            results["labels"],
            results["boxes"]
        ):
            x1, y1, x2, y2 = box.cpu().numpy()
            
            bbox = BoundingBox(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                confidence=float(score),
                label="text",
            )
            boxes.append(bbox)
        
        return boxes
    
    def batch_detect(
        self,
        images: List[Union[np.ndarray, Image.Image]],
    ) -> List[DetectionResult]:
        """
        Batch detection with DETR.
        
        Args:
            images: List of input images
            
        Returns:
            List of DetectionResult objects
        """
        start_time = time.time()
        
        # Convert all to PIL
        pil_images = []
        original_shapes = []
        
        for image in images:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            pil_images.append(image)
            original_shapes.append((image.height, image.width))
        
        # Batch preprocess
        inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Batch inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process each result
        target_sizes = torch.tensor(original_shapes).to(self.device)
        results_batch = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.confidence_threshold,
        )
        
        # Parse all results
        detection_results = []
        for results, original_shape in zip(results_batch, original_shapes):
            boxes = self._parse_results(results)
            
            detection_results.append(DetectionResult(
                boxes=boxes,
                image_shape=original_shape,
                processing_time=(time.time() - start_time) / len(images),
            ))
        
        logger.info(f"DETR batch detection of {len(images)} images completed")
        return detection_results
