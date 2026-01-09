"""
Web Application Configuration Module.

This module provides configuration management for the LangRAG Web application,
built on top of pydantic-settings for type-safe, environment-aware configuration.

Design Philosophy:
-----------------
The Web configuration extends the core LangRAG settings with web-specific
configuration like database paths, API endpoints, and storage locations.

    WebSettings (pydantic-settings)
        ├── Inherits: Core LangRAG patterns
        ├── Automatic .env loading
        ├── WEB_ prefixed environment variables
        └── Auto-creates data directories on load

Configuration Hierarchy:
-----------------------
For a complete RAG web application, configuration flows like this:

    Environment Variables / .env
            │
            ▼
    ┌─────────────────────┐
    │  langrag.config     │  Core library settings (LANGRAG_ prefix)
    │  Settings           │  - QWEN_API_KEY, LOG_LEVEL, etc.
    └─────────────────────┘
            │
            ▼
    ┌─────────────────────┐
    │  web.config         │  Web app settings (WEB_ prefix)
    │  WebSettings        │  - Database paths, storage dirs
    └─────────────────────┘

Why Separate Settings?
---------------------
1. **Separation of Concerns**: Core library shouldn't know about web paths.
2. **Deployment Flexibility**: Web settings can differ per deployment.
3. **Testing Isolation**: Override web settings without affecting core.

Example Usage:
-------------
    # Import settings singleton (auto-loads from environment)
    from web.config import settings

    # Access configuration
    print(settings.DATA_DIR)
    print(settings.DATABASE_URL)

    # Backward-compatible module-level constants (deprecated)
    from web.config import DATA_DIR, DATABASE_URL

    # Override for testing
    from web.config import WebSettings
    test_settings = WebSettings(DATA_DIR=Path("/tmp/test_data"))

Environment Variables:
---------------------
All web environment variables use the WEB_ prefix:
- WEB_DATA_DIR: Root directory for web data storage
- WEB_DB_DIR: SQLite database directory
- WEB_CHROMA_DIR: ChromaDB storage directory
- WEB_DUCKDB_DIR: DuckDB storage directory
- WEB_SEEKDB_DIR: SeekDB storage directory
"""

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# =============================================================================
# Path Constants
# =============================================================================

# Project root directory detection
# This file: web/config.py -> project root is parent
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default data directory (can be overridden via environment)
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "web" / "data"


class WebSettings(BaseSettings):
    """
    Web application configuration settings.

    This class provides type-safe configuration for the LangRAG Web application.
    Values are automatically loaded from environment variables with the WEB_
    prefix, or from a .env file in the project root.

    The class automatically creates required directories on initialization,
    ensuring the application can start without manual setup.

    Attributes:
        PROJECT_ROOT: Project root directory (computed, not configurable).
        DATA_DIR: Root directory for all web data storage.
        DB_DIR: Directory for SQLite database files.
        CHROMA_DIR: Directory for ChromaDB vector storage.
        DUCKDB_DIR: Directory for DuckDB storage.
        SEEKDB_DIR: Directory for SeekDB vector storage.
        DATABASE_URL: SQLAlchemy-compatible database URL.
        CACHE_ENABLED: Whether semantic caching is enabled.
        CACHE_SIMILARITY_THRESHOLD: Minimum similarity for cache hit.
        CACHE_TTL_SECONDS: Cache entry time-to-live in seconds.
        CACHE_MAX_SIZE: Maximum number of cache entries.
    """

    model_config = SettingsConfigDict(
        # Look for .env file in project root
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        # Use WEB_ prefix for web-specific environment variables
        # e.g., WEB_DATA_DIR, WEB_CHROMA_DIR
        env_prefix="WEB_",
        extra="ignore",
        # Make settings immutable after initialization
        frozen=True,
        # Allow Path type
        arbitrary_types_allowed=True,
        case_sensitive=False,
    )

    # ==========================================================================
    # Path Configuration
    # ==========================================================================

    PROJECT_ROOT: Path = Field(
        default=_PROJECT_ROOT,
        description="Project root directory. Auto-detected, not typically overridden."
    )

    DATA_DIR: Path = Field(
        default=_DEFAULT_DATA_DIR,
        description="Root directory for all web data storage. "
        "All other *_DIR paths are relative to this unless absolute."
    )

    DB_DIR: Path | None = Field(
        default=None,
        description="Directory for SQLite database files. "
        "Defaults to DATA_DIR/db if not specified."
    )

    CHROMA_DIR: Path | None = Field(
        default=None,
        description="Directory for ChromaDB vector storage. "
        "Defaults to DATA_DIR/chroma if not specified."
    )

    DUCKDB_DIR: Path | None = Field(
        default=None,
        description="Directory for DuckDB storage. "
        "Defaults to DATA_DIR/duckdb if not specified."
    )

    SEEKDB_DIR: Path | None = Field(
        default=None,
        description="Directory for SeekDB vector storage. "
        "Defaults to DATA_DIR/seekdb if not specified."
    )

    # ==========================================================================
    # Semantic Cache Configuration
    # ==========================================================================

    CACHE_ENABLED: bool = Field(
        default=True,
        description="Enable semantic caching to reduce redundant searches. "
        "Set WEB_CACHE_ENABLED=false to disable."
    )

    CACHE_SIMILARITY_THRESHOLD: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for cache hit. "
        "Higher values (0.98) require near-exact matches. "
        "Lower values (0.90) allow more semantic variation."
    )

    CACHE_TTL_SECONDS: int = Field(
        default=3600,
        ge=0,
        description="Cache entry time-to-live in seconds. "
        "0 means no expiration. Default: 3600 (1 hour)."
    )

    CACHE_MAX_SIZE: int = Field(
        default=1000,
        ge=0,
        description="Maximum number of cached entries. "
        "When exceeded, oldest entries are evicted (LRU). "
        "0 means unlimited."
    )

    # ==========================================================================
    # Validators
    # ==========================================================================

    @model_validator(mode="after")
    def _set_defaults_and_create_dirs(self) -> "WebSettings":
        """
        Set default subdirectory paths and create all required directories.

        This validator runs after all fields are set, allowing us to:
        1. Set default paths based on DATA_DIR
        2. Create directories if they don't exist

        Note: We use object.__setattr__ because the model is frozen.
        """
        # Set default subdirectory paths if not explicitly configured
        if self.DB_DIR is None:
            object.__setattr__(self, "DB_DIR", self.DATA_DIR / "db")
        if self.CHROMA_DIR is None:
            object.__setattr__(self, "CHROMA_DIR", self.DATA_DIR / "chroma")
        if self.DUCKDB_DIR is None:
            object.__setattr__(self, "DUCKDB_DIR", self.DATA_DIR / "duckdb")
        if self.SEEKDB_DIR is None:
            object.__setattr__(self, "SEEKDB_DIR", self.DATA_DIR / "seekdb")

        # Create all directories
        # Using exist_ok=True and parents=True for robustness
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.DB_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.DUCKDB_DIR.mkdir(parents=True, exist_ok=True)
        self.SEEKDB_DIR.mkdir(parents=True, exist_ok=True)

        return self

    # ==========================================================================
    # Computed Properties
    # ==========================================================================

    @property
    def DATABASE_URL(self) -> str:
        """
        SQLAlchemy-compatible database URL for the application database.

        Returns:
            SQLite database URL pointing to app.db in DB_DIR.
        """
        return f"sqlite:///{self.DB_DIR / 'app.db'}"

    @property
    def ASYNC_DATABASE_URL(self) -> str:
        """
        Async SQLAlchemy-compatible database URL.

        For use with aiosqlite and async database engines.

        Returns:
            Async SQLite database URL.
        """
        return f"sqlite+aiosqlite:///{self.DB_DIR / 'app.db'}"


# =============================================================================
# Global Settings Instance
# =============================================================================

# Singleton settings instance, loaded at module import time.
settings = WebSettings()


def get_settings() -> WebSettings:
    """
    Get the global web settings instance.

    Useful for dependency injection with FastAPI's Depends().

    Returns:
        The global WebSettings instance.

    Example:
        from fastapi import Depends
        from web.config import get_settings, WebSettings

        @app.get("/config")
        def show_config(settings: WebSettings = Depends(get_settings)):
            return {"data_dir": str(settings.DATA_DIR)}
    """
    return settings


# =============================================================================
# Backward Compatibility
# =============================================================================

# These module-level constants are provided for backward compatibility
# with existing code that imports them directly.
#
# DEPRECATED: Prefer using `settings.DATA_DIR` etc. instead.
# These will be removed in a future version.

PROJECT_ROOT = settings.PROJECT_ROOT
DATA_DIR = settings.DATA_DIR
DB_DIR = settings.DB_DIR
CHROMA_DIR = settings.CHROMA_DIR
DUCKDB_DIR = settings.DUCKDB_DIR
SEEKDB_DIR = settings.SEEKDB_DIR
DATABASE_URL = settings.DATABASE_URL
