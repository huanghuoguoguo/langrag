"""Tests for Search router."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from langrag.entities.document import Document
from web.core.database import get_session
from web.routers.search import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    get_rag_kernel,
    router,
)


@pytest.fixture
def mock_session():
    """Create mock database session."""
    return MagicMock()


@pytest.fixture
def mock_rag_kernel():
    """Create mock RAG kernel."""
    return MagicMock()


@pytest.fixture
def mock_kb():
    """Create mock KB object."""
    kb = MagicMock()
    kb.kb_id = "kb-123"
    kb.name = "Test KB"
    return kb


class TestSearchRouter:
    """Tests for Search router endpoints."""

    def test_search_success(self, mock_rag_kernel, mock_session, mock_kb):
        """Search returns results."""
        mock_rag_kernel.search.return_value = (
            [
                Document(
                    page_content="Result content",
                    metadata={"score": 0.95, "source": "doc.txt"},
                )
            ],
            "vector",
        )

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_session] = lambda: mock_session
        app.dependency_overrides[get_rag_kernel] = lambda: mock_rag_kernel

        with patch("web.routers.search.KBService") as mock_service:
            mock_service.get_kb.return_value = mock_kb

            client = TestClient(app)
            response = client.post(
                "/api/search",
                json={"kb_id": "kb-123", "query": "test query", "top_k": 5},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["search_type"] == "vector"
            assert len(data["results"]) == 1
            assert data["results"][0]["content"] == "Result content"
            assert data["results"][0]["score"] == 0.95

    def test_search_kb_not_found(self, mock_rag_kernel, mock_session):
        """Search with invalid KB returns 404."""
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_session] = lambda: mock_session
        app.dependency_overrides[get_rag_kernel] = lambda: mock_rag_kernel

        with patch("web.routers.search.KBService") as mock_service:
            mock_service.get_kb.return_value = None

            client = TestClient(app)
            response = client.post(
                "/api/search",
                json={"kb_id": "nonexistent", "query": "test", "top_k": 5},
            )

            assert response.status_code == 404
            assert "not found" in response.json()["detail"]

    def test_search_error(self, mock_rag_kernel, mock_session, mock_kb):
        """Search error returns 500."""
        mock_rag_kernel.search.side_effect = Exception("Search failed")

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_session] = lambda: mock_session
        app.dependency_overrides[get_rag_kernel] = lambda: mock_rag_kernel

        with patch("web.routers.search.KBService") as mock_service:
            mock_service.get_kb.return_value = mock_kb

            client = TestClient(app)
            response = client.post(
                "/api/search",
                json={"kb_id": "kb-123", "query": "test", "top_k": 5},
            )

            assert response.status_code == 500
            assert "Search failed" in response.json()["detail"]

    def test_search_empty_results(self, mock_rag_kernel, mock_session, mock_kb):
        """Search with no results returns empty list."""
        mock_rag_kernel.search.return_value = ([], "vector")

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_session] = lambda: mock_session
        app.dependency_overrides[get_rag_kernel] = lambda: mock_rag_kernel

        with patch("web.routers.search.KBService") as mock_service:
            mock_service.get_kb.return_value = mock_kb

            client = TestClient(app)
            response = client.post(
                "/api/search",
                json={"kb_id": "kb-123", "query": "test", "top_k": 5},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["results"] == []
            assert data["search_type"] == "vector"


class TestSearchModels:
    """Tests for Search Pydantic models."""

    def test_search_request_defaults(self):
        """SearchRequest has correct defaults."""
        req = SearchRequest(kb_id="kb-1", query="test")

        assert req.top_k == 5

    def test_search_result_item(self):
        """SearchResultItem validates correctly."""
        item = SearchResultItem(
            content="test content",
            score=0.85,
            source="doc.pdf",
            search_type="hybrid",
        )

        assert item.content == "test content"
        assert item.score == 0.85
        assert item.source == "doc.pdf"
        assert item.search_type == "hybrid"

    def test_search_response(self):
        """SearchResponse validates correctly."""
        resp = SearchResponse(
            results=[
                SearchResultItem(
                    content="content",
                    score=0.9,
                    source="source",
                    search_type="vector",
                )
            ],
            search_type="vector",
        )

        assert len(resp.results) == 1
        assert resp.search_type == "vector"
