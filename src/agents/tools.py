"""
Tools for the document understanding agent.
Each tool represents a component of the pipeline.
"""

from typing import Dict, Any, List
from pathlib import Path
import numpy as np
from PIL import Image

from ..detection import DetectorFactory
from ..recognition import RecognizerFactory
from ..layout import LayoutLMv3Analyzer
from ..vlm import GeminiVLM
from ..utils.logger import get_logger
from ..utils.config import config

logger = get_logger(__name__)


class DocumentUnderstandingTools:
    """Collection of tools for document understanding agent."""
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize all tools.
        
        Args:
            device: Device to run models on
        """
        self.device = device
        self._detector = None
        self._recognizer = None
        self._layout_analyzer = None
        self._vlm = None
        
        logger.info("Initializing Document Understanding Tools...")
    
    @property
    def detector(self):
        """Lazy load detector."""
        if self._detector is None:
            logger.info("Loading text detector...")
            self._detector = DetectorFactory.create_detector(
                detector_type="faster_rcnn",
                device=self.device,
                confidence_threshold=0.5,
            )
        return self._detector
    
    @property
    def recognizer(self):
        """Lazy load recognizer."""
        if self._recognizer is None:
            logger.info("Loading text recognizer...")
            self._recognizer = RecognizerFactory.create_recognizer(
                recognizer_type="trocr",
                device=self.device,
            )
        return self._recognizer
    
    @property
    def layout_analyzer(self):
        """Lazy load layout analyzer."""
        if self._layout_analyzer is None:
            logger.info("Loading layout analyzer...")
            self._layout_analyzer = LayoutLMv3Analyzer(
                device=self.device,
                apply_ocr=False,
            )
        return self._layout_analyzer
    
    @property
    def vlm(self):
        """Lazy load VLM."""
        if self._vlm is None:
            try:
                logger.info("Loading VLM...")
                self._vlm = GeminiVLM()
            except Exception as e:
                logger.warning(f"Failed to load VLM: {e}")
                self._vlm = None
        return self._vlm
    
    def detect_text(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect text regions in image.
        
        Args:
            image: Input image
            
        Returns:
            Detection results
        """
        try:
            result = self.detector.detect(image)
            return {
                "success": True,
                "num_regions": len(result.boxes),
                "boxes": [box.to_dict() for box in result.boxes],
                "processing_time": result.processing_time,
            }
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return {"success": False, "error": str(e)}
    
    def recognize_text(self, image: np.ndarray, boxes: List) -> Dict[str, Any]:
        """
        Recognize text in detected regions.
        
        Args:
            image: Input image
            boxes: List of bounding boxes
            
        Returns:
            Recognition results
        """
        try:
            from ..detection.base_detector import BoundingBox
            
            # Convert dict boxes to BoundingBox objects
            bbox_objects = [
                BoundingBox(
                    x1=box["x1"],
                    y1=box["y1"],
                    x2=box["x2"],
                    y2=box["y2"],
                    confidence=box["confidence"],
                    label=box.get("label", "text"),
                )
                for box in boxes
            ]
            
            results = self.recognizer.recognize(image, bbox_objects)
            
            return {
                "success": True,
                "num_texts": len(results),
                "texts": [r.to_dict() for r in results],
            }
        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return {"success": False, "error": str(e)}
    
    def analyze_layout(
        self,
        image: np.ndarray,
        recognition_results: List[Dict],
    ) -> Dict[str, Any]:
        """
        Analyze document layout.
        
        Args:
            image: Input image
            recognition_results: List of recognition results
            
        Returns:
            Layout analysis results
        """
        try:
            from ..recognition.base_recognizer import RecognitionResult
            from ..detection.base_detector import BoundingBox
            
            # Convert dicts to RecognitionResult objects
            rec_objects = [
                RecognitionResult(
                    text=r["text"],
                    confidence=r["confidence"],
                    bbox=BoundingBox(**r["bbox"]),
                )
                for r in recognition_results
            ]
            
            layout = self.layout_analyzer.analyze(image, rec_objects)
            
            return {
                "success": True,
                "num_entities": len(layout.entities),
                "entities": [e.to_dict() for e in layout.entities],
                "processing_time": layout.processing_time,
            }
        except Exception as e:
            logger.error(f"Layout analysis error: {e}")
            return {"success": False, "error": str(e)}
    
    def extract_information(
        self,
        image: np.ndarray,
        query: str,
    ) -> Dict[str, Any]:
        """
        Extract specific information using VLM.
        
        Args:
            image: Input image
            query: What to extract
            
        Returns:
            Extracted information
        """
        try:
            if self.vlm is None:
                return {
                    "success": False,
                    "error": "VLM not available",
                }
            
            response = self.vlm.answer_question(
                Image.fromarray(image),
                query,
            )
            
            return {
                "success": True,
                "answer": response,
            }
        except Exception as e:
            logger.error(f"Information extraction error: {e}")
            return {"success": False, "error": str(e)}
