"""
Custom Vision Transformer for document layout analysis.
Lightweight alternative to LayoutLMv3 for simple layout tasks.
"""

import time
from typing import List, Union, Optional
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from transformers import ViTModel, ViTConfig

from .base_layout import BaseLayoutAnalyzer, DocumentLayout, LayoutEntity
from ..detection.base_detector import BoundingBox
from ..recognition.base_recognizer import RecognitionResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ViTLayoutClassifier(nn.Module):
    """Vision Transformer with classification head for layout analysis."""
    
    def __init__(
        self,
        num_labels: int = 9,
        pretrained_model: str = "google/vit-base-patch16-224",
    ):
        """
        Initialize ViT layout classifier.
        
        Args:
            num_labels: Number of layout labels
            pretrained_model: Pretrained ViT model name
        """
        super().__init__()
        
        self.vit = ViTModel.from_pretrained(pretrained_model)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, pixel_values):
        """Forward pass."""
        outputs = self.vit(pixel_values=pixel_values)
        sequence_output = outputs.last_hidden_state
        
        # Use [CLS] token
        cls_output = sequence_output[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        
        return logits


class ViTLayoutAnalyzer(BaseLayoutAnalyzer):
    """
    Custom ViT-based layout analyzer.
    Simpler and faster than LayoutLMv3 for basic layout tasks.
    """
    
    LABEL_MAP = {
        0: "text",
        1: "title",
        2: "list",
        3: "table",
        4: "figure",
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        num_labels: int = 5,
    ):
        """Initialize ViT layout analyzer."""
        super().__init__(
            model_path=model_path or "google/vit-base-patch16-224",
            device=device,
        )
        self.num_labels = num_labels
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        
        self.load_model()
    
    def load_model(self) -> None:
        """Load ViT model."""
        logger.info("Loading ViT layout classifier...")
        
        self.model = ViTLayoutClassifier(
            num_labels=self.num_labels,
            pretrained_model=self.model_path,
        )
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("✓ ViT layout model loaded successfully")
    
    def analyze(
        self,
        image: Union[np.ndarray, Image.Image],
        ocr_results: Optional[List[RecognitionResult]] = None,
    ) -> DocumentLayout:
        """
        Analyze document layout.
        
        Args:
            image: Input image
            ocr_results: Pre-computed OCR results
            
        Returns:
            DocumentLayout
        """
        start_time = time.time()
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        original_shape = (image.height, image.width)
        
        # If OCR results provided, classify each region
        if ocr_results:
            entities = self._classify_regions(image, ocr_results)
        else:
            # Classify entire image
            entities = [self._classify_full_image(image)]
        
        processing_time = time.time() - start_time
        
        return DocumentLayout(
            entities=entities,
            image_shape=original_shape,
            processing_time=processing_time,
        )
    
    def _classify_regions(
        self,
        image: Image.Image,
        ocr_results: List[RecognitionResult],
    ) -> List[LayoutEntity]:
        """Classify each OCR region."""
        entities = []
        
        for result in ocr_results:
            # Crop region
            x1, y1, x2, y2 = int(result.bbox.x1), int(result.bbox.y1), \
                             int(result.bbox.x2), int(result.bbox.y2)
            
            cropped = image.crop((x1, y1, x2, y2))
            
            # Classify
            label_id = self._classify_image(cropped)
            label = self.LABEL_MAP.get(label_id, "text")
            
            entity = LayoutEntity(
                text=result.text,
                bbox=result.bbox,
                label=label,
                confidence=result.confidence,
            )
            entities.append(entity)
        
        return entities
    
    def _classify_image(self, image: Image.Image) -> int:
        """Classify a single image region."""
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(img_tensor)
            pred = logits.argmax(-1).item()
        
        return pred
    
    def _classify_full_image(self, image: Image.Image) -> LayoutEntity:
        """Classify entire image."""
        label_id = self._classify_image(image)
        label = self.LABEL_MAP.get(label_id, "text")
        
        bbox = BoundingBox(
            x1=0,
            y1=0,
            x2=image.width,
            y2=image.height,
            confidence=0.9,
        )
        
        return LayoutEntity(
            text="",
            bbox=bbox,
            label=label,
            confidence=0.9,
        )
