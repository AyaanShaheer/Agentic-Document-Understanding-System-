"""
Run the FastAPI application.
Usage: python run_api.py
"""

import uvicorn
from src.utils.config import config

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.project.environment == "development",
        workers=1 if config.project.environment == "development" else config.api.workers,
    )
