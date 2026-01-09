"""
LangRAG Core Configuration Module.

This module provides centralized configuration management using Pydantic Settings.
It demonstrates the recommended pattern for configuration in LangRAG applications.

Architecture Design:
-------------------
The configuration system follows a layered approach:

    SettingsConfigDict (Pydantic Settings)
        ├── Automatic .env file loading
        ├── Environment variable binding (with prefix)
        └── Type validation and coercion

Configuration Sources (in priority order, highest to lowest):
1. Explicit constructor arguments
2. Environment variables (with LANGRAG_ prefix for core settings)
3. .env file in project root
4. Default values in Field definitions

Why Pydantic Settings?
---------------------
1. **Type Safety**: All configuration values are validated at load time.
   Typos or invalid values fail fast with clear error messages.

2. **Environment Variable Binding**: Automatic loading from environment
   variables without manual os.getenv() calls. The prefix prevents
   collision with other applications.

3. **Documentation**: Field descriptions serve as self-documenting
   configuration reference.

4. **Immutability**: Settings are frozen after initialization,
   preventing accidental modifications during runtime.

5. **Testability**: Easy to override settings in tests by passing
   explicit values to the constructor.

Example Usage:
-------------
    # Default usage (loads from environment)
    from langrag.config import settings
    print(settings.LOG_LEVEL)

    # Override for testing
    from langrag.config.settings import Settings
    test_settings = Settings(ENV="testing", LOG_LEVEL="DEBUG")

    # Access nested path configurations
    print(settings.CHROMA_DB_PATH)

Environment Variables:
---------------------
All environment variables use the LANGRAG_ prefix:
- LANGRAG_ENV: Application environment (development/production/testing)
- LANGRAG_LOG_LEVEL: Logging level (DEBUG/INFO/WARNING/ERROR)
- LANGRAG_CHROMA_DB_PATH: ChromaDB storage path
- LANGRAG_DUCKDB_PATH: DuckDB storage path
- LANGRAG_QWEN_API_KEY: Qwen API key for LLM/Reranker services

The prefix can be omitted in .env file for convenience:
    ENV=production
    LOG_LEVEL=INFO
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root directory detection
# This file: src/langrag/config/settings.py
# Project root: 4 levels up
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    """
    Core LangRAG configuration settings.

    This class provides type-safe, validated configuration for the LangRAG
    library. Values are automatically loaded from environment variables
    with the LANGRAG_ prefix, or from a .env file in the project root.

    Attributes:
        ENV: Application environment (development/production/testing).
             Affects logging verbosity and feature toggles.
        LOG_LEVEL: Logging level for the application.
        ROOT_DIR: Project root directory path.
        CHROMA_DB_PATH: Storage path for ChromaDB vector database.
        DUCKDB_PATH: Storage path for DuckDB database.
        QWEN_API_KEY: API key for Qwen services (LLM, Reranker).
    """

    model_config = SettingsConfigDict(
        # Look for .env file in project root
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        # Use LANGRAG_ prefix for environment variables
        # e.g., LANGRAG_LOG_LEVEL, LANGRAG_ENV
        env_prefix="LANGRAG_",
        # Also check for unprefixed env vars (backward compatibility)
        # Priority: prefixed > unprefixed > .env > defaults
        extra="ignore",
        # Make settings immutable after initialization
        frozen=True,
        # Allow Path type
        arbitrary_types_allowed=True,
        # Case insensitive env var names
        case_sensitive=False,
    )

    # ==========================================================================
    # Environment Configuration
    # ==========================================================================

    ENV: str = Field(
        default="development",
        description="Application environment: development, production, or testing. "
        "Affects logging verbosity, debug features, and performance optimizations."
    )

    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level for the application. "
        "Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL."
    )

    # ==========================================================================
    # Path Configuration
    # ==========================================================================

    ROOT_DIR: Path = Field(
        default=PROJECT_ROOT,
        description="Project root directory. All relative paths are resolved from here."
    )

    CHROMA_DB_PATH: str = Field(
        default="storage/chroma_db",
        description="Storage path for ChromaDB vector database. "
        "Can be absolute or relative to ROOT_DIR."
    )

    DUCKDB_PATH: str = Field(
        default="storage/langrag.duckdb",
        description="Storage path for DuckDB database file. "
        "Can be absolute or relative to ROOT_DIR."
    )

    # ==========================================================================
    # API Keys
    # ==========================================================================

    QWEN_API_KEY: str | None = Field(
        default=None,
        description="API key for Qwen services (Dashscope). "
        "Required for QwenReranker and Qwen LLM features. "
        "Get your key at: https://dashscope.aliyun.com/"
    )

    # ==========================================================================
    # Computed Properties
    # ==========================================================================

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENV.lower() == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENV.lower() == "production"

    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.ENV.lower() == "testing"

    def get_absolute_path(self, relative_path: str) -> Path:
        """
        Convert a relative path to absolute path based on ROOT_DIR.

        Args:
            relative_path: Path relative to ROOT_DIR, or absolute path.

        Returns:
            Absolute Path object.
        """
        path = Path(relative_path)
        if path.is_absolute():
            return path
        return self.ROOT_DIR / path


# =============================================================================
# Global Settings Instance
# =============================================================================

# Singleton settings instance, loaded at module import time.
# This provides convenient access: `from langrag.config import settings`
settings = Settings()


def get_settings() -> Settings:
    """
    Get the global settings instance.

    This function is useful for dependency injection patterns,
    particularly with FastAPI's Depends().

    Returns:
        The global Settings instance.

    Example:
        from fastapi import Depends
        from langrag.config.settings import get_settings, Settings

        @app.get("/config")
        def show_config(settings: Settings = Depends(get_settings)):
            return {"env": settings.ENV}
    """
    return settings
