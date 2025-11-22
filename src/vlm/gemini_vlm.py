"""
Google Gemini Vision API integration for document understanding.
Uses Gemini 2.0 Flash for multimodal document analysis.
"""

import time
from typing import Union, Optional, Dict, Any, List
import numpy as np
from PIL import Image
import json
import google.generativeai as genai

from .base_vlm import BaseVLM, VLMResponse
from ..layout.base_layout import DocumentLayout
from ..utils.logger import get_logger
from ..utils.config import config

logger = get_logger(__name__)


class GeminiVLM(BaseVLM):
    """
    Gemini Vision-Language Model for document understanding.
    Uses Google's Gemini API for multimodal reasoning.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-exp",
        temperature: float = 0.3,
    ):
        """
        Initialize Gemini VLM.

        Args:
            api_key: Google API key (uses config if not provided)
            model_name: Gemini model name
            temperature: Generation temperature
        """
        self.api_key = api_key or config.gemini.api_key
        self.model_name = model_name or config.gemini.model
        self.temperature = temperature or config.gemini.temperature

        # call parent's constructor (keeps your original pattern)
        super().__init__(model_path=self.api_key, device="cpu")
        self.load_model()

    def load_model(self) -> None:
        """Initialize Gemini API."""
        logger.info(f"Initializing Gemini API: {self.model_name}")

        try:
            genai.configure(api_key=self.api_key)

            # Keep generation_config as a dict; your client may expect different parameter names.
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": self.temperature,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                },
            )

            logger.info("✓ Gemini VLM initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {e}")
            logger.error("Make sure GOOGLE_API_KEY is set in .env file")
            raise

    def query(
        self,
        image: Union[np.ndarray, Image.Image],
        prompt: str,
        document_layout: Optional[DocumentLayout] = None,
    ) -> VLMResponse:
        """
        Query Gemini with image and prompt.

        Args:
            image: Input image
            prompt: Question or instruction
            document_layout: Optional layout context

        Returns:
            VLMResponse with Gemini's answer
        """
        start_time = time.time()

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if document_layout:
            layout_context = self._format_layout_context(document_layout)
            enhanced_prompt = f"{prompt}\n\nDocument Layout Context:\n{layout_context}"
        else:
            enhanced_prompt = prompt

        try:
            logger.debug(f"Sending request to Gemini: {self.model_name}")

            # Many genai clients accept either a generate(...) or generate_content(...) call.
            # If your client uses a different call signature adapt accordingly.
            response = self.model.generate_content([enhanced_prompt, image])

            # Some responses use different fields; prefer .text if present, otherwise try str(response)
            resp_text = None
            if hasattr(response, "text") and response.text is not None:
                resp_text = response.text
            elif hasattr(response, "output") and response.output is not None:
                # fallback common name
                resp_text = response.output
            else:
                # As a last resort convert to string representation
                resp_text = str(response)

            # Safety / blocked content check (guarded)
            block_reason = None
            if hasattr(response, "prompt_feedback") and getattr(response, "prompt_feedback") is not None:
                pf = response.prompt_feedback
                # some clients embed block reason differently; guard access
                block_reason = getattr(pf, "block_reason", None) or getattr(pf, "reason", None)

            if block_reason:
                logger.warning(f"Content was blocked: {block_reason}")
                return VLMResponse(
                    text="Content was blocked by safety filters.",
                    confidence=0.0,
                    metadata={
                        "error": "blocked",
                        "reason": str(block_reason),
                    },
                )

            processing_time = time.time() - start_time

            logger.debug(f"Gemini query completed in {processing_time:.3f}s")

            # Ensure resp_text is a string
            resp_text = resp_text if isinstance(resp_text, str) else str(resp_text)

            return VLMResponse(
                text=resp_text,
                confidence=0.9,
                metadata={
                    "model": self.model_name,
                    "processing_time": processing_time,
                    "prompt": prompt,
                    "tokens": len(resp_text.split()) if resp_text else 0,
                },
            )

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return VLMResponse(
                text=f"Error: {str(e)}",
                confidence=0.0,
                metadata={"error": str(e)},
            )

    def _format_layout_context(self, document_layout: DocumentLayout) -> str:
        """Format layout information as context for the prompt."""
        context_parts = []
        labels = set(e.label for e in document_layout.entities)

        for label in sorted(labels):
            entities = document_layout.get_entities_by_label(label)
            if entities:
                texts = list({e.text.strip() for e in entities if e.text and e.text.strip()})[:5]
                if texts:
                    context_parts.append(f"{label}: {', '.join(texts)}")

        return "\n".join(context_parts) if context_parts else "No layout context available."

    def extract_entities(
        self,
        image: Union[np.ndarray, Image.Image],
        document_layout: DocumentLayout,
    ) -> Dict[str, Any]:
        """Extract structured entities using Gemini."""
        prompt = """Analyze this document and extract the following information as JSON:
{
    "document_type": "invoice/receipt/form/letter/other",
    "title": "document title if any",
    "date": "document date if any",
    "entities": [
        {"type": "person/organization/amount/date/other", "value": "extracted value", "label": "field name"}
    ],
    "key_values": {
        "field_name": "field_value"
    },
    "summary": "brief summary of the document"
}

Return ONLY valid JSON, no markdown code blocks or explanations."""
        response = self.query(image, prompt, document_layout)

        try:
            text = response.text.strip() if response.text else ""

            # Strip common fenced code blocks like ```json ... ``` or ```
            if text.startswith("```json"):
                text = text[len("```json"):]
            elif text.startswith("```"):
                text = text[len("```"):]

            if text.endswith("```"):
                text = text[: -len("```")]

            text = text.strip()
            entities = json.loads(text)
            return entities

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from Gemini response: {e}")
            logger.debug(f"Raw response: {response.text}")
            return {
                "error": "json_parse_failed",
                "raw_text": response.text,
                "extracted_partial": self._extract_key_value_pairs(response.text or ""),
            }

    def _extract_key_value_pairs(self, text: str) -> Dict[str, str]:
        """Fallback method to extract key-value pairs from unstructured text."""
        pairs = {}
        lines = text.splitlines()

        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip().strip('"\'')
                    value = parts[1].strip().strip(',').strip('"\'')
                    if key:
                        pairs[key] = value

        return pairs

    def answer_question(
        self,
        image: Union[np.ndarray, Image.Image],
        question: str,
        document_layout: Optional[DocumentLayout] = None,
    ) -> str:
        """Answer a specific question about the document."""
        prompt = f"""Question: {question}

Please provide a clear, concise, and accurate answer based on the document. 
If the information is not present in the document, say "Information not found in document."
Answer directly without repeating the question."""
        response = self.query(image, prompt, document_layout)
        return response.text

    def summarize(
        self,
        image: Union[np.ndarray, Image.Image],
        document_layout: Optional[DocumentLayout] = None,
        max_length: int = 200,
    ) -> str:
        """Generate a summary of the document."""
        prompt = f"""Provide a concise summary of this document in approximately {max_length} words or less.
Focus on:
1. Document type and purpose
2. Key information (names, dates, amounts, etc.)
3. Main action items or conclusions

Be specific and factual."""
        response = self.query(image, prompt, document_layout)
        return response.text

    def classify_document(
        self,
        image: Union[np.ndarray, Image.Image],
        categories: List[str],
    ) -> Dict[str, float]:
        """Classify document into one of the given categories."""
        categories_str = ", ".join(categories)

        prompt = f"""Classify this document into ONE of the following categories: {categories_str}

Return your answer in this exact JSON format:
{{"category": "chosen_category", "confidence": 0.95, "reasoning": "brief explanation"}}"""

        response = self.query(image, prompt)

        try:
            text = response.text.strip() if response.text else ""

            if text.startswith("```json"):
                text = text[len("```json"):]
            elif text.startswith("```"):
                text = text[len("```"):]

            if text.endswith("```"):
                text = text[: -len("```")]

            result = json.loads(text.strip())

            scores = {cat: 0.0 for cat in categories}
            if result.get("category") in categories:
                scores[result["category"]] = float(result.get("confidence", 0.9))

            return scores

        except Exception as e:
            logger.warning(f"Failed to parse classification result: {e}")
            return {cat: 1.0 / len(categories) for cat in categories}

    def validate_document(
        self,
        image: Union[np.ndarray, Image.Image],
        expected_fields: List[str],
    ) -> Dict[str, Any]:
        """Validate if document contains all expected fields."""
        fields_str = ", ".join(expected_fields)

        prompt = f"""Check if this document contains all of the following required fields: {fields_str}

For each field, indicate if it's present and what value it has.

Return as JSON:
{{"fields": [{{"name": "field_name", "present": true, "value": "extracted_value"}}], "is_valid": true, "missing_fields": []}}"""

        response = self.query(image, prompt)

        try:
            text = response.text.strip() if response.text else ""

            if text.startswith("```json"):
                text = text[len("```json"):]
            elif text.startswith("```"):
                text = text[len("```"):]

            if text.endswith("```"):
                text = text[: -len("```")]

            result = json.loads(text.strip())
            return result

        except Exception as e:
            logger.warning(f"Failed to parse validation result: {e}")
            return {
                "is_valid": False,
                "error": str(e),
                "raw_response": response.text,
            }
