#!/bin/bash

# Script to run all tests for LangRAG project
# This script uses uv for dependency management

set -e

echo "🧪 Setting up test environment..."

# Check if uv is installed
if ! command -v uv >/dev/null 2>&1; then
    echo "❌ Error: uv is not installed"
    echo "Please install uv: https://github.com/astral-sh/uv"
    exit 1
fi

echo "📦 Ensuring dependencies are installed..."
uv sync --all-extras

echo ""
echo "🎯 Choose test scope:"
echo "  1) All tests (default)"
echo "  2) Smoke tests only"
echo "  3) Unit tests only"
echo "  4) Integration tests only"
echo "  5) E2E tests only"
echo ""

# Parse first argument or default to "all"
TEST_SCOPE="${1:-all}"

case "$TEST_SCOPE" in
    smoke|2)
        echo "🔥 Running smoke tests..."
        uv run pytest tests/smoke/ -v -m smoke
        ;;
    unit|3)
        echo "🔬 Running unit tests..."
        uv run pytest tests/unit/ -v \
            --cov=src/langrag \
            --cov-report=xml \
            --cov-report=term-missing
        ;;
    integration|4)
        echo "🔗 Running integration tests..."
        uv run pytest tests/integration/ -v --tb=short
        ;;
    e2e|5)
        echo "🌐 Running E2E tests..."
        uv run pytest tests/e2e/ -v --tb=short
        ;;
    all|1|*)
        echo "🚀 Running all tests with coverage..."
        uv run pytest tests/ -v \
            --cov=src/langrag \
            --cov-report=xml \
            --cov-report=html \
            --cov-report=term-missing
        echo ""
        echo "📊 Coverage report:"
        echo "  - XML: coverage.xml"
        echo "  - HTML: htmlcov/index.html"
        ;;
esac

echo ""
echo "✅ Test run complete!"