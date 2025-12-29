import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env file from the project root
# We assume the project root is the current working directory or 3 levels up from this file
# This file: src/langrag/config/settings.py
SERVER_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ENV_PATH = SERVER_ROOT / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    # Fallback to simple load_dotenv which looks in cwd
    load_dotenv()

class Settings(BaseModel):
    """Global Application Settings"""
    
    # Environment
    ENV: str = Field(default="development", description="Environment: development, production, testing")
    LOG_LEVEL: str = Field(default="INFO", description="Log level")
    
    # Project Paths
    ROOT_DIR: Path = Field(default=SERVER_ROOT, description="Project root directory")
    
    # Storage Paths
    # Defaulting to a 'storage' directory in root if possible, or simple relative paths
    CHROMA_DB_PATH: str = Field(default="storage/chroma_db", description="Path to ChromaDB storage")
    DUCKDB_PATH: str = Field(default="storage/langrag.duckdb", description="Path to DuckDB storage")
    
    # Model API Keys
    QWEN_API_KEY: Optional[str] = Field(default=None, description="Qwen API Key")

    model_config = {
        "frozen": True,
        "arbitrary_types_allowed": True
    }

def load_settings() -> Settings:
    """Load settings from environment variables."""
    return Settings(
        ENV=os.getenv("ENV", "development"),
        LOG_LEVEL=os.getenv("LOG_LEVEL", "INFO"),
        ROOT_DIR=SERVER_ROOT,
        CHROMA_DB_PATH=os.getenv("CHROMA_DB_PATH", str(SERVER_ROOT / "storage/chroma_db")),
        DUCKDB_PATH=os.getenv("DUCKDB_PATH", str(SERVER_ROOT / "storage/langrag.duckdb")),
        QWEN_API_KEY=os.getenv("QWEN_API_KEY"),
    )

# Global settings instance
settings = load_settings()
