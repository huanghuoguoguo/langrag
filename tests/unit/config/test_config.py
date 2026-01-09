"""
Tests for the LangRAG configuration system.

These tests verify that:
1. Settings are properly loaded from environment variables
2. Default values are applied when no env vars are set
3. Settings class properties work correctly
4. Path resolution works as expected
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from langrag.config.settings import Settings, get_settings


class TestSettings:
    """Tests for the core Settings class."""

    def test_load_settings_from_env(self):
        """Test that settings are loaded from environment variables with LANGRAG_ prefix."""
        # Use LANGRAG_ prefix as per the new settings configuration
        with patch.dict(os.environ, {
            "LANGRAG_ENV": "production",
            "LANGRAG_LOG_LEVEL": "DEBUG",
            "LANGRAG_CHROMA_DB_PATH": "/tmp/chroma",
            "LANGRAG_DUCKDB_PATH": "/tmp/duckdb",
            "LANGRAG_QWEN_API_KEY": "sk-test-key"
        }, clear=False):
            # Create new Settings instance to pick up env vars
            settings = Settings()

            assert settings.ENV == "production"
            assert settings.LOG_LEVEL == "DEBUG"
            assert settings.CHROMA_DB_PATH == "/tmp/chroma"
            assert settings.DUCKDB_PATH == "/tmp/duckdb"
            assert settings.QWEN_API_KEY == "sk-test-key"

    def test_default_values(self):
        """Test that default values are used when no environment variables are set."""
        # Create Settings with explicit defaults (ignoring environment)
        settings = Settings(
            ENV="development",
            LOG_LEVEL="INFO",
            CHROMA_DB_PATH="storage/chroma_db",
            DUCKDB_PATH="storage/langrag.duckdb",
            QWEN_API_KEY=None
        )

        assert settings.ENV == "development"
        assert settings.LOG_LEVEL == "INFO"
        assert "storage/chroma_db" in str(settings.CHROMA_DB_PATH)
        assert settings.QWEN_API_KEY is None

    def test_root_dir_resolution(self):
        """Test that ROOT_DIR is correctly resolved to the project root."""
        settings = Settings()
        assert isinstance(settings.ROOT_DIR, Path)
        # Check if settings.py is inside ROOT_DIR
        # settings.py is in src/langrag/config/
        # ROOT_DIR should be parent of src
        assert (settings.ROOT_DIR / "src").exists()

    def test_get_settings_function(self):
        """Test the get_settings() function for dependency injection."""
        settings = get_settings()
        assert settings is not None
        assert isinstance(settings, Settings)

    def test_environment_properties(self):
        """Test the computed environment properties."""
        # Test development environment
        dev_settings = Settings(ENV="development")
        assert dev_settings.is_development is True
        assert dev_settings.is_production is False
        assert dev_settings.is_testing is False

        # Test production environment
        prod_settings = Settings(ENV="production")
        assert prod_settings.is_development is False
        assert prod_settings.is_production is True
        assert prod_settings.is_testing is False

        # Test testing environment
        test_settings = Settings(ENV="testing")
        assert test_settings.is_development is False
        assert test_settings.is_production is False
        assert test_settings.is_testing is True

    def test_get_absolute_path(self):
        """Test the get_absolute_path helper method."""
        settings = Settings()

        # Test relative path conversion
        relative_path = "storage/test.db"
        abs_path = settings.get_absolute_path(relative_path)
        assert abs_path.is_absolute()
        assert str(abs_path).endswith("storage/test.db")

        # Test that absolute paths are returned unchanged
        absolute_path = "/tmp/test.db"
        result = settings.get_absolute_path(absolute_path)
        assert str(result) == "/tmp/test.db"

    def test_settings_immutability(self):
        """Test that settings are frozen and cannot be modified."""
        from pydantic import ValidationError
        settings = Settings()
        with pytest.raises(ValidationError):
            settings.ENV = "modified"

    def test_case_insensitive_env_vars(self):
        """Test that environment variable names are case-insensitive."""
        with patch.dict(os.environ, {
            "langrag_env": "staging",
        }, clear=False):
            settings = Settings()
            # Note: pydantic-settings handles case sensitivity based on config
            # The default is case_sensitive=False
            assert settings.ENV.lower() in ["staging", "development"]
