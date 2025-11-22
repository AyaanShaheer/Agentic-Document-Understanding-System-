"""
Configuration management for the Agentic Document Understanding System.
Loads and validates all environment variables and model configurations.
"""

import os
from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
import yaml
from loguru import logger
from dataclasses import dataclass

# Load environment variables
load_dotenv()


class ProjectConfig(BaseModel):
    """Main project configuration."""
    
    project_name: str = Field(default="agentic-document-understanding")
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    
    @validator("device")
    def validate_device(cls, v):
        """Validate device availability."""
        import torch
        if v == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return "cpu"
        return v


class ModelConfig(BaseModel):
    """Model paths and configurations."""
    
    # Detection
    detection_model_path: str = "facebook/detr-resnet-50"
    detection_threshold: float = 0.7
    
    # Recognition
    recognition_model_path: str = "microsoft/trocr-base-printed"
    
    # Layout
    layout_model_path: str = "microsoft/layoutlmv3-base"
    
    # Vision-Language Model
    vlm_model_path: str = "Qwen/Qwen-VL-Chat"
    
    # Processing
    batch_size: int = 8
    max_image_size: int = 1024


class GeminiConfig(BaseModel):
    """Gemini API configuration."""
    
    api_key: str = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    model: str = Field(default="gemini-2.0-flash-exp")
    temperature: float = 0.3
    max_retries: int = 2
    max_iterations: int = 5
    
@dataclass
class GroqConfig:
    """Groq API configuration."""
    api_key: str
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.2

    @validator("api_key")
    def validate_api_key(cls, v):
        """Ensure API key is provided."""
        if not v:
            raise ValueError("GOOGLE_API_KEY must be set in environment variables")
        return v


class APIConfig(BaseModel):
    """FastAPI server configuration."""
    
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False


class Config:
    """Main configuration class that aggregates all configs."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration from environment and optional YAML file."""
        
        self.project = ProjectConfig(
            project_name=os.getenv("PROJECT_NAME", "agentic-document-understanding"),
            environment=os.getenv("ENVIRONMENT", "development"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            device=os.getenv("DEVICE", "cuda"),
        )
        
        self.models = ModelConfig(
            detection_model_path=os.getenv("DETECTION_MODEL_PATH", "facebook/detr-resnet-50"),
            recognition_model_path=os.getenv("RECOGNITION_MODEL_PATH", "microsoft/trocr-base-printed"),
            layout_model_path=os.getenv("LAYOUT_MODEL_PATH", "microsoft/layoutlmv3-base"),
            vlm_model_path=os.getenv("VLM_MODEL_PATH", "Qwen/Qwen-VL-Chat"),
            batch_size=int(os.getenv("BATCH_SIZE", "8")),
            max_image_size=int(os.getenv("MAX_IMAGE_SIZE", "1024")),
        )
        
        self.gemini = GeminiConfig(
            api_key=os.getenv("GOOGLE_API_KEY", ""),
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"),
            temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.3")),
            max_iterations=int(os.getenv("MAX_AGENT_ITERATIONS", "5")),
        )
        
        # Groq configuration
        self.groq = GroqConfig(
            api_key=os.getenv("GROQ_API_KEY", ""),
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=float(os.getenv("GROQ_TEMPERATURE", "0.2")),
)


        self.api = APIConfig(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            workers=int(os.getenv("API_WORKERS", "4")),
        )
        
        # Load additional config from YAML if provided
        if config_path and config_path.exists():
            self._load_yaml_config(config_path)
        
        logger.info(f"Configuration loaded: {self.project.environment} environment")
        logger.info(f"Device: {self.project.device}")
        logger.info(f"Gemini Model: {self.gemini.model}")
    
    def _load_yaml_config(self, config_path: Path):
        """Load additional configuration from YAML file."""
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Override with YAML values if present
        if 'models' in yaml_config:
            for key, value in yaml_config['models'].items():
                if hasattr(self.models, key):
                    setattr(self.models, key, value)


# Global config instance
config = Config()
