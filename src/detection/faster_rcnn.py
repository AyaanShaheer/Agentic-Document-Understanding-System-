"""
Faster-RCNN based text detection using TorchVision.
Pre-trained on COCO dataset and fine-tunable for document text detection.
"""

import time
from typing import Union, List, Optional
import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image

from .base_detector import BaseDetector, BoundingBox, DetectionResult
from ..utils.logger import get_logger
from ..utils.config import config

logger = get_logger(__name__)


class FasterRCNNDetector(BaseDetector):
    """
    Faster-RCNN detector for text region detection.
    Uses ResNet50-FPN-V2 backbone with improved performance.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.7,
        num_classes: int = 2,  # background + text
        pretrained: bool = True,
    ):
        """
        Initialize Faster-RCNN detector.
        
        Args:
            model_path: Path to fine-tuned checkpoint (optional)
            device: Device to run on
            confidence_threshold: Minimum confidence for detections
            num_classes: Number of classes (2 for text detection)
            pretrained: Whether to load pretrained COCO weights
        """
        super().__init__(
            model_path=model_path or "torchvision://fasterrcnn_resnet50_fpn_v2",
            device=device,
            confidence_threshold=confidence_threshold,
        )
        
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.load_model()
    
    def load_model(self) -> None:
        """Load Faster-RCNN model with ResNet50-FPN-V2 backbone."""
        from pathlib import Path
    
        logger.info("Loading Faster-RCNN ResNet50-FPN-V2 model...")
    
        # Load pretrained model
        if self.pretrained:
            weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        else:
            self.model = fasterrcnn_resnet50_fpn_v2(weights=None)
    
        # Replace the classifier head for custom number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, 
            self.num_classes
    )
    
        # Load custom weights if provided and file exists
        if self.model_path:
            model_path = Path(self.model_path)
            # Only try to load if it's an actual file path (not a HF model ID) and exists
            if model_path.exists() and model_path.is_file():
                logger.info(f"Loading custom weights from {self.model_path}")
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint)
            elif not str(self.model_path).startswith("pretrained"):
                logger.warning(
                    f"Model path {self.model_path} does not exist. "
                    f"Using pretrained COCO weights."
            )
    
        self.model.to(self.device)
        self.model.eval()
    
        logger.info("✓ Faster-RCNN model loaded successfully")


    def detect(
        self,
        image: Union[np.ndarray, torch.Tensor, Image.Image],
    ) -> DetectionResult:
        """
        Detect text regions in an image.
        
        Args:
            image: Input image (numpy array, tensor, or PIL Image)
            
        Returns:
            DetectionResult with bounding boxes
        """
        start_time = time.time()
        
        # Preprocess image
        image_tensor, original_shape = self._preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model([image_tensor])[0]
        
        # Parse predictions
        boxes = self._parse_predictions(predictions, original_shape)
        
        # Filter and apply NMS
        boxes = self._filter_boxes(boxes)
        boxes = self._apply_nms(boxes, iou_threshold=0.5)
        
        processing_time = time.time() - start_time
        
        logger.debug(f"Detected {len(boxes)} text regions in {processing_time:.3f}s")
        
        return DetectionResult(
            boxes=boxes,
            image_shape=original_shape,
            processing_time=processing_time,
        )
    
    def _preprocess_image(
        self,
        image: Union[np.ndarray, torch.Tensor, Image.Image],
    ) -> tuple:
        """
        Preprocess image for Faster-RCNN.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (preprocessed tensor, original shape)
        """
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            original_shape = (image.height, image.width)
        elif isinstance(image, Image.Image):
            original_shape = (image.height, image.width)
        elif isinstance(image, torch.Tensor):
            original_shape = (image.shape[-2], image.shape[-1])
            # Assume already preprocessed
            return image.to(self.device), original_shape
        
        # Convert to tensor and normalize
        image_tensor = torchvision.transforms.functional.to_tensor(image)
        image_tensor = image_tensor.to(self.device)
        
        return image_tensor, original_shape
    
    def _parse_predictions(
        self,
        predictions: dict,
        original_shape: tuple,
    ) -> List[BoundingBox]:
        """
        Parse Faster-RCNN predictions into BoundingBox objects.
        
        Args:
            predictions: Model predictions
            original_shape: Original image shape (height, width)
            
        Returns:
            List of BoundingBox objects
        """
        boxes = []
        
        pred_boxes = predictions['boxes'].cpu().numpy()
        pred_scores = predictions['scores'].cpu().numpy()
        pred_labels = predictions['labels'].cpu().numpy()
        
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            # Skip background class (label 0)
            if label == 0:
                continue
            
            x1, y1, x2, y2 = box
            
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
        images: List[Union[np.ndarray, torch.Tensor]],
    ) -> List[DetectionResult]:
        """
        Detect text regions in multiple images (batched inference).
        
        Args:
            images: List of input images
            
        Returns:
            List of DetectionResult objects
        """
        start_time = time.time()
        
        # Preprocess all images
        image_tensors = []
        original_shapes = []
        
        for image in images:
            tensor, shape = self._preprocess_image(image)
            image_tensors.append(tensor)
            original_shapes.append(shape)
        
        # Run batched inference
        with torch.no_grad():
            predictions_batch = self.model(image_tensors)
        
        # Parse each prediction
        results = []
        for predictions, original_shape in zip(predictions_batch, original_shapes):
            boxes = self._parse_predictions(predictions, original_shape)
            boxes = self._filter_boxes(boxes)
            boxes = self._apply_nms(boxes, iou_threshold=0.5)
            
            results.append(DetectionResult(
                boxes=boxes,
                image_shape=original_shape,
                processing_time=(time.time() - start_time) / len(images),
            ))
        
        logger.info(f"Batch detection of {len(images)} images completed")
        return results
    
    def train_mode(self):
        """Set model to training mode."""
        self.model.train()
    
    def eval_mode(self):
        """Set model to evaluation mode."""
        self.model.eval()
