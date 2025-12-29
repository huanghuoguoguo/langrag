import os
import pytest
from unittest.mock import patch
from langrag.config.settings import load_settings, Settings
from pathlib import Path

class TestSettings:
    
    @patch.dict(os.environ, {
        "ENV": "production",
        "LOG_LEVEL": "DEBUG",
        "CHROMA_DB_PATH": "/tmp/chroma",
        "DUCKDB_PATH": "/tmp/duckdb",
        "QWEN_API_KEY": "sk-test-key"
    }, clear=True)
    def test_load_settings_from_env(self):
        # Force reload because settings might be cached or instantiated globally
        settings = load_settings()
        
        assert settings.ENV == "production"
        assert settings.LOG_LEVEL == "DEBUG"
        assert settings.CHROMA_DB_PATH == "/tmp/chroma"
        assert settings.DUCKDB_PATH == "/tmp/duckdb"
        assert settings.QWEN_API_KEY == "sk-test-key"

    def test_default_values(self):
        # Ensure clearing specific vars to test defaults
        with patch.dict(os.environ, {}, clear=True):
            settings = load_settings()
            
            assert settings.ENV == "development"
            assert settings.LOG_LEVEL == "INFO"
            assert "storage/chroma_db" in str(settings.CHROMA_DB_PATH) # Verify relative path structure
            assert settings.QWEN_API_KEY is None

    def test_root_dir_resolution(self):
        settings = load_settings()
        assert isinstance(settings.ROOT_DIR, Path)
        # Check if settings.py is inside ROOT_DIR
        # settings.py is in src/langrag/config/
        # ROOT_DIR should be parent of src
        assert (settings.ROOT_DIR / "src").exists()
