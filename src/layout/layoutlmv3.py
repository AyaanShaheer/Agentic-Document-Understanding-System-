"""
LayoutLMv3-based document layout understanding.
Multimodal transformer for Document AI with unified text and image masking.
"""

import time
from typing import List, Union, Optional, Dict, Any, Tuple
import numpy as np
import torch
from PIL import Image
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3ImageProcessor,
)

from .base_layout import BaseLayoutAnalyzer, DocumentLayout, LayoutEntity
from ..detection.base_detector import BoundingBox
from ..recognition.base_recognizer import RecognitionResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


class LayoutLMv3Analyzer(BaseLayoutAnalyzer):
    """
    LayoutLMv3 analyzer for document layout understanding.
    Processes text, layout, and visual features jointly.
    """
    
    # Standard layout labels for document understanding
    LABEL_MAP = {
        0: "O",          # Other
        1: "B-HEADER",   # Beginning of header
        2: "I-HEADER",   # Inside header
        3: "B-QUESTION", # Beginning of question
        4: "I-QUESTION", # Inside question
        5: "B-ANSWER",   # Beginning of answer
        6: "I-ANSWER",   # Inside answer
        7: "B-TITLE",    # Beginning of title
        8: "I-TITLE",    # Inside title
    }
    
    def __init__(
        self,
        model_path: str = "microsoft/layoutlmv3-base",
        device: str = "cuda",
        apply_ocr: bool = True,
    ):
        """
        Initialize LayoutLMv3 analyzer.
        
        Args:
            model_path: HuggingFace model ID or local path
            device: Device to run on
            apply_ocr: Whether to apply OCR (uses Tesseract internally)
        """
        super().__init__(model_path=model_path, device=device)
        self.apply_ocr = apply_ocr
        self.processor = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load LayoutLMv3 model and processor."""
        logger.info(f"Loading LayoutLMv3 model: {self.model_path}")
        
        # Load processor and model
        self.processor = LayoutLMv3Processor.from_pretrained(
            self.model_path,
            apply_ocr=self.apply_ocr,
        )
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            self.model_path
        )
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("✓ LayoutLMv3 model loaded successfully")
    
    def analyze(
        self,
        image: Union[np.ndarray, Image.Image],
        ocr_results: Optional[List[RecognitionResult]] = None,
    ) -> DocumentLayout:
        """
        Analyze document layout using LayoutLMv3.
        
        Args:
            image: Input image
            ocr_results: Pre-computed OCR results (optional)
            
        Returns:
            DocumentLayout with classified entities
        """
        start_time = time.time()
        
        # Convert to PIL
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        original_shape = (image.height, image.width)
        
        # Prepare input
        if ocr_results and len(ocr_results) > 0:
            # Use pre-computed OCR
            entities = self._analyze_with_ocr(image, ocr_results)
        else:
            # Let processor apply OCR (only if apply_ocr is True)
            if self.apply_ocr:
                entities = self._analyze_with_processor_ocr(image)
            else:
                logger.warning("No OCR results provided and apply_ocr=False. Returning empty layout.")
                entities = []
        
        processing_time = time.time() - start_time
        
        logger.debug(f"LayoutLMv3 analyzed {len(entities)} entities in {processing_time:.3f}s")
        
        return DocumentLayout(
            entities=entities,
            image_shape=original_shape,
            processing_time=processing_time,
        )

    
    def _analyze_with_processor_ocr(
        self,
        image: Image.Image,
    ) -> List[LayoutEntity]:
        """
        Analyze using processor's built-in OCR.
        
        Args:
            image: PIL Image
            
        Returns:
            List of LayoutEntity objects
        """
        try:
            # Process image (applies OCR internally)
            encoding = self.processor(
                image,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=512,
            )
            
            # Move to device
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**encoding)
            
            # Get predictions
            predictions = outputs.logits.argmax(-1).squeeze().tolist()
            
            # Extract entities
            entities = self._extract_entities(
                encoding,
                predictions,
                image.size,
            )
            
            return entities
        
        except Exception as e:
            logger.error(f"Error in processor OCR: {e}")
            logger.warning("Falling back to empty layout")
            return []

    
    def _analyze_with_ocr(
        self,
        image: Image.Image,
        ocr_results: List[RecognitionResult],
    ) -> List[LayoutEntity]:
        """
        Analyze using pre-computed OCR results.
        
        Args:
            image: PIL Image
            ocr_results: Pre-computed OCR results
            
        Returns:
            List of LayoutEntity objects
        """
        # Prepare words and boxes
        words = [result.text for result in ocr_results]
        boxes = [
            [
                int(result.bbox.x1),
                int(result.bbox.y1),
                int(result.bbox.x2),
                int(result.bbox.y2),
            ]
            for result in ocr_results
        ]
        
        # Process with LayoutLMv3
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        
        # Create entities with predictions
        entities = []
        for result, pred_id in zip(ocr_results, predictions):
            if isinstance(pred_id, list):
                pred_id = pred_id[0]
            
            label = self.LABEL_MAP.get(pred_id, "O")
            
            entity = LayoutEntity(
                text=result.text,
                bbox=result.bbox,
                label=label,
                confidence=result.confidence,
            )
            entities.append(entity)
        
        return entities
    
    def _extract_entities(
        self,
        encoding: Dict[str, torch.Tensor],
        predictions: List[int],
        image_size: Tuple[int, int],
    ) -> List[LayoutEntity]:
        """
        Extract entities from model predictions.
        
        Args:
            encoding: Processor encoding
            predictions: Model predictions
            image_size: Original image size (width, height)
            
        Returns:
            List of LayoutEntity objects
        """
        entities = []
        
        # Get tokens and boxes
        tokens = encoding["input_ids"].squeeze().tolist()
        boxes = encoding["bbox"].squeeze().tolist()
        
        # Decode tokens
        if self.processor.tokenizer:
            words = self.processor.tokenizer.convert_ids_to_tokens(tokens)
        else:
            words = [str(t) for t in tokens]
        
        # Process each token
        for word, box, pred_id in zip(words, boxes, predictions):
            # Skip special tokens
            if word in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            
            # Normalize box coordinates
            x1, y1, x2, y2 = box
            x1 = (x1 / 1000) * image_size[0]
            y1 = (y1 / 1000) * image_size[1]
            x2 = (x2 / 1000) * image_size[0]
            y2 = (y2 / 1000) * image_size[1]
            
            bbox = BoundingBox(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                confidence=0.9,
            )
            
            label = self.LABEL_MAP.get(pred_id, "O")
            
            entity = LayoutEntity(
                text=word.replace("##", ""),  # Remove BERT subword marker
                bbox=bbox,
                label=label,
                confidence=0.9,
            )
            entities.append(entity)
        
        return entities
    
    def fine_tune(
        self,
        train_dataset,
        eval_dataset,
        output_dir: str = "./layoutlm_finetuned",
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
    ):
        """
        Fine-tune LayoutLMv3 on custom dataset.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_dir: Directory to save model
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        from transformers import TrainingArguments, Trainer
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        logger.info("Starting LayoutLMv3 fine-tuning...")
        trainer.train()
        
        # Save model
        trainer.save_model(output_dir)
        logger.info(f"Model saved to {output_dir}")
