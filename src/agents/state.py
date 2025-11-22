"""
LangGraph state management for document understanding agent.
"""

from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np
from PIL import Image

from ..detection.base_detector import BoundingBox, DetectionResult
from ..recognition.base_recognizer import RecognitionResult
from ..layout.base_layout import DocumentLayout, LayoutEntity


@dataclass
class DocumentState:
    """State container for document understanding pipeline."""
    
    # Input
    image: Optional[np.ndarray] = None
    image_path: Optional[str] = None
    user_query: str = ""
    
    # Pipeline results
    detection_result: Optional[DetectionResult] = None
    recognition_results: List[RecognitionResult] = field(default_factory=list)
    layout_result: Optional[DocumentLayout] = None
    
    # Extracted information
    entities: Dict[str, Any] = field(default_factory=dict)
    structured_data: Dict[str, Any] = field(default_factory=dict)
    
    # Agent reasoning
    reasoning_steps: List[str] = field(default_factory=list)
    final_answer: str = ""
    confidence_score: float = 0.0
    
    # Metadata
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "user_query": self.user_query,
            "detection": self.detection_result.to_dict() if self.detection_result else None,
            "recognition": [r.to_dict() for r in self.recognition_results],
            "layout": self.layout_result.to_dict() if self.layout_result else None,
            "entities": self.entities,
            "structured_data": self.structured_data,
            "reasoning_steps": self.reasoning_steps,
            "final_answer": self.final_answer,
            "confidence_score": self.confidence_score,
            "processing_time": self.processing_time,
            "errors": self.errors,
        }


class AgentState(TypedDict):
    """TypedDict for LangGraph state."""
    
    document_state: DocumentState
    messages: List[Dict[str, str]]
    next_step: str
