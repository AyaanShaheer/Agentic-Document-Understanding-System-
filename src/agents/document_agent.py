"""
LangGraph-based document understanding agent.
Orchestrates the entire pipeline with reasoning capabilities.
"""

import time
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
from PIL import Image

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .state import AgentState, DocumentState
from .tools import DocumentUnderstandingTools
from ..utils.logger import get_logger
from ..utils.config import config

logger = get_logger(__name__)


class DocumentUnderstandingAgent:
    """
    Agentic document understanding system using LangGraph.
    Coordinates detection, recognition, layout analysis, and reasoning.
    """
    
    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize the document understanding agent.
        
        Args:
            groq_api_key: Groq API key (uses config if not provided)
            device: Device for model inference
        """
        self.device = device
        self.tools = DocumentUnderstandingTools(device=device)
        
        # Initialize LLM (Groq)
        api_key = groq_api_key or config.gemini.api_key  # Fallback to any key
        self.llm = ChatGroq(
            api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.2,
        )
        
        # Build graph
        self.graph = self._build_graph()
        
        logger.info("✓ Document Understanding Agent initialized")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Define nodes
        workflow.add_node("detect", self._detect_node)
        workflow.add_node("recognize", self._recognize_node)
        workflow.add_node("analyze_layout", self._analyze_layout_node)
        workflow.add_node("reason", self._reason_node)
        
        # Define edges
        workflow.set_entry_point("detect")
        workflow.add_edge("detect", "recognize")
        workflow.add_edge("recognize", "analyze_layout")
        workflow.add_edge("analyze_layout", "reason")
        workflow.add_edge("reason", END)
        
        return workflow.compile()
    
    def _detect_node(self, state: AgentState) -> AgentState:
        """Detection node."""
        logger.info("Step 1: Text Detection")
        
        doc_state = state["document_state"]
        doc_state.reasoning_steps.append("Detecting text regions in document...")
        
        result = self.tools.detect_text(doc_state.image)
        
        if result["success"]:
            from ..detection.base_detector import DetectionResult, BoundingBox
            
            boxes = [BoundingBox(**box) for box in result["boxes"]]
            doc_state.detection_result = DetectionResult(
                boxes=boxes,
                image_shape=doc_state.image.shape[:2],
                processing_time=result["processing_time"],
            )
            
            logger.info(f"✓ Detected {len(boxes)} text regions")
        else:
            doc_state.errors.append(f"Detection failed: {result.get('error')}")
        
        state["document_state"] = doc_state
        return state
    
    def _recognize_node(self, state: AgentState) -> AgentState:
        """Recognition node."""
        logger.info("Step 2: Text Recognition")
        
        doc_state = state["document_state"]
        doc_state.reasoning_steps.append("Recognizing text in detected regions...")
        
        if doc_state.detection_result:
            boxes = [box.to_dict() for box in doc_state.detection_result.boxes]
            result = self.tools.recognize_text(doc_state.image, boxes)
            
            if result["success"]:
                from ..recognition.base_recognizer import RecognitionResult
                from ..detection.base_detector import BoundingBox
                
                doc_state.recognition_results = [
                    RecognitionResult(
                        text=r["text"],
                        confidence=r["confidence"],
                        bbox=BoundingBox(**r["bbox"]),
                    )
                    for r in result["texts"]
                ]
                
                logger.info(f"✓ Recognized {len(doc_state.recognition_results)} texts")
            else:
                doc_state.errors.append(f"Recognition failed: {result.get('error')}")
        
        state["document_state"] = doc_state
        return state
    
    def _analyze_layout_node(self, state: AgentState) -> AgentState:
        """Layout analysis node."""
        logger.info("Step 3: Layout Analysis")
        
        doc_state = state["document_state"]
        doc_state.reasoning_steps.append("Analyzing document layout structure...")
        
        if doc_state.recognition_results:
            rec_dicts = [r.to_dict() for r in doc_state.recognition_results]
            result = self.tools.analyze_layout(doc_state.image, rec_dicts)
            
            if result["success"]:
                from ..layout.base_layout import DocumentLayout, LayoutEntity
                from ..detection.base_detector import BoundingBox
                
                entities = [
                    LayoutEntity(
                        text=e["text"],
                        bbox=BoundingBox(**e["bbox"]),
                        label=e["label"],
                        confidence=e["confidence"],
                    )
                    for e in result["entities"]
                ]
                
                doc_state.layout_result = DocumentLayout(
                    entities=entities,
                    image_shape=doc_state.image.shape[:2],
                    processing_time=result["processing_time"],
                )
                
                logger.info(f"✓ Analyzed {len(entities)} layout entities")
            else:
                doc_state.errors.append(f"Layout analysis failed: {result.get('error')}")
        
        state["document_state"] = doc_state
        return state
    
    def _reason_node(self, state: AgentState) -> AgentState:
        """Reasoning node using LLM."""
        logger.info("Step 4: LLM Reasoning")
        
        doc_state = state["document_state"]
        doc_state.reasoning_steps.append("Applying LLM reasoning to answer query...")
        
        # Prepare context for LLM
        context = self._prepare_context(doc_state)
        
        # Create prompt
        prompt = f"""You are a document understanding AI assistant. You have analyzed a document and extracted the following information:

{context}

User Query: {doc_state.user_query}

Based on the extracted information, provide a clear and concise answer to the user's query.
If the information is not available in the document, say so explicitly."""
        
        messages = [
            SystemMessage(content="You are an expert document understanding assistant."),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = self.llm.invoke(messages)
            doc_state.final_answer = response.content
            doc_state.confidence_score = 0.9
            
            logger.info("✓ Generated final answer")
        except Exception as e:
            logger.error(f"LLM reasoning error: {e}")
            doc_state.final_answer = "Error generating answer."
            doc_state.errors.append(f"LLM error: {str(e)}")
        
        state["document_state"] = doc_state
        return state
    
    def _prepare_context(self, doc_state: DocumentState) -> str:
        """Prepare context string for LLM."""
        context_parts = []
        
        # Detection info
        if doc_state.detection_result:
            context_parts.append(
                f"Detected {len(doc_state.detection_result.boxes)} text regions."
            )
        
        # Recognition info
        if doc_state.recognition_results:
            texts = [r.text for r in doc_state.recognition_results[:20]]  # Top 20
            context_parts.append(f"Recognized texts: {', '.join(texts)}")
        
        # Layout info
        if doc_state.layout_result:
            labels = {}
            for entity in doc_state.layout_result.entities:
                labels[entity.label] = labels.get(entity.label, 0) + 1
            
            layout_summary = ", ".join([f"{k}: {v}" for k, v in labels.items()])
            context_parts.append(f"Layout structure: {layout_summary}")
        
        return "\n\n".join(context_parts)
    
    def process_document(
        self,
        image_path: str,
        query: str,
    ) -> Dict[str, Any]:
        """
        Process a document and answer a query.
        
        Args:
            image_path: Path to document image
            query: User's question about the document
            
        Returns:
            Processing results
        """
        start_time = time.time()
        
        # Load image
        image = np.array(Image.open(image_path).convert("RGB"))
        
        # Initialize state
        doc_state = DocumentState(
            image=image,
            image_path=image_path,
            user_query=query,
        )
        
        initial_state = {
            "document_state": doc_state,
            "messages": [],
            "next_step": "detect",
        }
        
        # Run graph
        logger.info(f"Processing document: {image_path}")
        logger.info(f"Query: {query}")
        
        try:
            final_state = self.graph.invoke(initial_state)
            doc_state = final_state["document_state"]
            doc_state.processing_time = time.time() - start_time
            
            return doc_state.to_dict()
        
        except Exception as e:
            logger.error(f"Agent processing error: {e}")
            return {
                "error": str(e),
                "query": query,
                "processing_time": time.time() - start_time,
            }
