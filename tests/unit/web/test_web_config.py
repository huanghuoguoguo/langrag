"""
Tests for the Web application configuration system.

These tests verify that:
1. WebSettings are properly loaded from environment variables
2. Default directory paths are correctly computed
3. Directory creation works properly
4. Backward compatibility constants are exported
"""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from web.config import (
    CHROMA_DIR,
    DATA_DIR,
    DATABASE_URL,
    DB_DIR,
    DUCKDB_DIR,
    PROJECT_ROOT,
    SEEKDB_DIR,
    WebSettings,
    get_settings,
    settings,
)


class TestWebSettings:
    """Tests for the WebSettings class."""

    def test_default_paths(self):
        """Test that default paths are correctly set."""
        assert settings.DATA_DIR == PROJECT_ROOT / "web" / "data"
        assert settings.DB_DIR == settings.DATA_DIR / "db"
        assert settings.CHROMA_DIR == settings.DATA_DIR / "chroma"
        assert settings.DUCKDB_DIR == settings.DATA_DIR / "duckdb"
        assert settings.SEEKDB_DIR == settings.DATA_DIR / "seekdb"

    def test_backward_compatibility_constants(self):
        """Test that module-level constants match settings."""
        assert DATA_DIR == settings.DATA_DIR
        assert DB_DIR == settings.DB_DIR
        assert CHROMA_DIR == settings.CHROMA_DIR
        assert DUCKDB_DIR == settings.DUCKDB_DIR
        assert SEEKDB_DIR == settings.SEEKDB_DIR
        assert DATABASE_URL == settings.DATABASE_URL

    def test_database_url_format(self):
        """Test that DATABASE_URL is correctly formatted."""
        assert settings.DATABASE_URL.startswith("sqlite:///")
        assert "app.db" in settings.DATABASE_URL

    def test_async_database_url(self):
        """Test the async database URL property."""
        assert settings.ASYNC_DATABASE_URL.startswith("sqlite+aiosqlite:///")
        assert "app.db" in settings.ASYNC_DATABASE_URL

    def test_get_settings_function(self):
        """Test the get_settings() function for dependency injection."""
        result = get_settings()
        assert result is settings
        assert isinstance(result, WebSettings)

    def test_directories_exist(self):
        """Test that required directories are created on settings initialization."""
        assert settings.DATA_DIR.exists()
        assert settings.DB_DIR.exists()
        assert settings.CHROMA_DIR.exists()
        assert settings.DUCKDB_DIR.exists()
        assert settings.SEEKDB_DIR.exists()

    def test_custom_data_dir(self):
        """Test that custom DATA_DIR properly affects subdirectory defaults."""
        with TemporaryDirectory() as tmpdir:
            custom_data_dir = Path(tmpdir) / "custom_data"
            custom_settings = WebSettings(DATA_DIR=custom_data_dir)

            # Subdirectories should be relative to custom DATA_DIR
            assert custom_settings.DB_DIR == custom_data_dir / "db"
            assert custom_settings.CHROMA_DIR == custom_data_dir / "chroma"
            assert custom_settings.DUCKDB_DIR == custom_data_dir / "duckdb"
            assert custom_settings.SEEKDB_DIR == custom_data_dir / "seekdb"

            # Directories should be created
            assert custom_settings.DATA_DIR.exists()
            assert custom_settings.DB_DIR.exists()

    def test_env_vars_with_prefix(self):
        """Test that environment variables with WEB_ prefix are recognized."""
        with TemporaryDirectory() as tmpdir:
            env_data_dir = Path(tmpdir) / "env_data"
            with patch.dict(os.environ, {
                "WEB_DATA_DIR": str(env_data_dir)
            }, clear=False):
                env_settings = WebSettings()
                assert env_settings.DATA_DIR == env_data_dir

    def test_settings_immutability(self):
        """Test that settings are frozen and cannot be modified."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            settings.DATA_DIR = Path("/tmp/modified")

    def test_project_root_detection(self):
        """Test that PROJECT_ROOT is correctly detected."""
        # PROJECT_ROOT should contain web/ and src/ directories
        assert (settings.PROJECT_ROOT / "web").exists()
        assert (settings.PROJECT_ROOT / "src").exists()
