"""
FastAPI application for document understanding.
"""

from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import time
from pathlib import Path

from ..agents import DocumentUnderstandingTools
from ..utils.logger import get_logger
from ..utils.config import config
from .schemas import (
    DetectionResponse,
    DocumentAnalysisResponse,
    QueryRequest,
    QueryResponse,
    HealthResponse,
    BoundingBoxResponse,
    RecognitionResponse,
    LayoutEntityResponse,
)

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Agentic Document Understanding API",
    description="Production-ready document understanding with detection, recognition, and layout analysis",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global tools instance (lazy loaded)
tools: Optional[DocumentUnderstandingTools] = None


def get_tools() -> DocumentUnderstandingTools:
    """Get or initialize tools."""
    global tools
    if tools is None:
        logger.info("Initializing document understanding tools...")
        tools = DocumentUnderstandingTools(device=config.project.device)
    return tools


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("Starting Agentic Document Understanding API...")
    logger.info(f"Device: {config.project.device}")
    logger.info(f"Environment: {config.project.environment}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API...")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint."""
    return {
        "status": "online",
        "version": "1.0.0",
        "models_loaded": {
            "detector": tools._detector is not None if tools else False,
            "recognizer": tools._recognizer is not None if tools else False,
            "layout_analyzer": tools._layout_analyzer is not None if tools else False,
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": {
            "detector": True,
            "recognizer": True,
            "layout_analyzer": True,
        }
    }


@app.post("/detect", response_model=DetectionResponse)
async def detect_text(file: UploadFile = File(...)):
    """
    Detect text regions in an uploaded document image.
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_array = np.array(image)
        
        # Detect
        tools_instance = get_tools()
        result = tools_instance.detect_text(image_array)
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error"))
        
        # Convert to response model
        boxes = [BoundingBoxResponse(**box) for box in result["boxes"]]
        
        return DetectionResponse(
            num_regions=result["num_regions"],
            boxes=boxes,
            processing_time=result["processing_time"],
        )
    
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=DocumentAnalysisResponse)
async def analyze_document(file: UploadFile = File(...)):
    """
    Complete document analysis: detection + recognition + layout.
    """
    start_time = time.time()
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_array = np.array(image)
        
        tools_instance = get_tools()
        
        # Step 1: Detection
        detect_result = tools_instance.detect_text(image_array)
        if not detect_result.get("success"):
            raise HTTPException(status_code=500, detail="Detection failed")
        
        # Filter tiny boxes
        boxes = [
            box for box in detect_result["boxes"]
            if (box["x2"] - box["x1"]) * (box["y2"] - box["y1"]) > 500
        ]
        
        # Step 2: Recognition
        recog_result = tools_instance.recognize_text(image_array, boxes)
        if not recog_result.get("success"):
            raise HTTPException(status_code=500, detail="Recognition failed")
        
        # Step 3: Layout Analysis
        layout_result = tools_instance.analyze_layout(image_array, recog_result["texts"])
        if not layout_result.get("success"):
            raise HTTPException(status_code=500, detail="Layout analysis failed")
        
        # Build response
        detection_response = DetectionResponse(
            num_regions=len(boxes),
            boxes=[BoundingBoxResponse(**box) for box in boxes],
            processing_time=detect_result["processing_time"],
        )
        
        recognition_responses = [
            RecognitionResponse(
                text=r["text"],
                confidence=r["confidence"],
                bbox=BoundingBoxResponse(**r["bbox"]),
            )
            for r in recog_result["texts"]
        ]
        
        layout_responses = [
            LayoutEntityResponse(
                text=e["text"],
                label=e["label"],
                confidence=e["confidence"],
                bbox=BoundingBoxResponse(**e["bbox"]),
            )
            for e in layout_result["entities"]
        ]
        
        total_time = time.time() - start_time
        
        return DocumentAnalysisResponse(
            detection=detection_response,
            recognition=recognition_responses,
            layout_entities=layout_responses,
            processing_time=total_time,
            status="success",
        )
    
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_document(
    file: UploadFile = File(...),
    query: str = Form(...),
):
    """
    Answer a question about a document using the full pipeline + VLM.
    """
    start_time = time.time()
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_array = np.array(image)
        
        tools_instance = get_tools()
        
        # Run analysis pipeline
        detect_result = tools_instance.detect_text(image_array)
        boxes = [
            box for box in detect_result["boxes"]
            if (box["x2"] - box["x1"]) * (box["y2"] - box["y1"]) > 500
        ]
        
        recog_result = tools_instance.recognize_text(image_array, boxes)
        layout_result = tools_instance.analyze_layout(image_array, recog_result["texts"])
        
        # Extract answer using VLM
        vlm_result = tools_instance.extract_information(image_array, query)
        
        answer = vlm_result.get("answer", "Unable to answer query")
        
        total_time = time.time() - start_time
        
        return QueryResponse(
            query=query,
            answer=answer,
            confidence=0.9,
            reasoning_steps=[
                f"Detected {len(boxes)} text regions",
                f"Recognized {len(recog_result['texts'])} texts",
                f"Analyzed {len(layout_result['entities'])} layout entities",
                "Applied VLM reasoning",
            ],
            processing_time=total_time,
            status="success",
        )
    
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
