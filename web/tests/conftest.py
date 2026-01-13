"""Pytest configuration for Web Demo tests."""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "local_llm: Tests that require a local LLM model"
    )
