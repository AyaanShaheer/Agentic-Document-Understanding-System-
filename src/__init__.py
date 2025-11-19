"""Agentic Document Understanding System - Main Package."""

__version__ = "0.1.0"
__author__ = "Your Name"

from .utils.config import config
from .utils.logger import get_logger

__all__ = ["config", "get_logger"]
