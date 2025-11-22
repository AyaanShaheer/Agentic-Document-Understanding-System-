"""
Pydantic schemas for API request/response models.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class BoundingBoxResponse(BaseModel):
    """Bounding box response model."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str = "text"


class DetectionResponse(BaseModel):
    """Text detection response."""
    num_regions: int
    boxes: List[BoundingBoxResponse]
    processing_time: float


class RecognitionResponse(BaseModel):
    """Text recognition response."""
    text: str
    confidence: float
    bbox: BoundingBoxResponse


class LayoutEntityResponse(BaseModel):
    """Layout entity response."""
    text: str
    label: str
    confidence: float
    bbox: BoundingBoxResponse


class DocumentAnalysisResponse(BaseModel):
    """Complete document analysis response."""
    detection: DetectionResponse
    recognition: List[RecognitionResponse]
    layout_entities: List[LayoutEntityResponse]
    processing_time: float
    status: str = "success"


class QueryRequest(BaseModel):
    """Document query request."""
    query: str = Field(..., description="Question to answer about the document")


class QueryResponse(BaseModel):
    """Document query response."""
    query: str
    answer: str
    confidence: float
    reasoning_steps: List[str]
    processing_time: float
    status: str = "success"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_loaded: Dict[str, bool]
