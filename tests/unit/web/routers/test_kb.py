"""Tests for KB router."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from web.core.database import get_session
from web.routers.kb import KBCreateRequest, KBResponse, get_rag_kernel, router


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
    kb.description = "Test description"
    kb.vdb_type = "chroma"
    kb.embedder_name = "text-embedding-3-small"
    kb.collection_name = "test_collection"
    kb.chunk_size = 1000
    kb.chunk_overlap = 100
    return kb


class TestKBRouter:
    """Tests for KB router endpoints."""

    def test_create_kb_success(self, mock_rag_kernel, mock_session, mock_kb):
        """Create KB returns KB response."""
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_session] = lambda: mock_session
        app.dependency_overrides[get_rag_kernel] = lambda: mock_rag_kernel

        with patch("web.routers.kb.KBService") as mock_service:
            mock_service.create_kb.return_value = mock_kb

            client = TestClient(app)
            response = client.post(
                "/api/kb",
                json={
                    "name": "Test KB",
                    "description": "Test description",
                    "vdb_type": "chroma",
                    "embedder_name": "text-embedding-3-small",
                    "chunk_size": 1000,
                    "chunk_overlap": 100,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "kb-123"
            assert data["name"] == "Test KB"
            assert data["vdb_type"] == "chroma"

    def test_create_kb_missing_embedder(self, mock_rag_kernel, mock_session):
        """Create KB without embedder returns 400."""
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_session] = lambda: mock_session
        app.dependency_overrides[get_rag_kernel] = lambda: mock_rag_kernel

        client = TestClient(app)
        response = client.post(
            "/api/kb",
            json={
                "name": "Test KB",
                "embedder_name": "",  # Empty embedder
            },
        )

        assert response.status_code == 400
        assert "Embedder is required" in response.json()["detail"]

    def test_get_kb_success(self, mock_rag_kernel, mock_session, mock_kb):
        """Get KB returns KB details."""
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_session] = lambda: mock_session
        app.dependency_overrides[get_rag_kernel] = lambda: mock_rag_kernel

        with patch("web.routers.kb.KBService") as mock_service:
            mock_service.get_kb.return_value = mock_kb

            client = TestClient(app)
            response = client.get("/api/kb/kb-123")

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "kb-123"
            assert data["name"] == "Test KB"

    def test_get_kb_not_found(self, mock_rag_kernel, mock_session):
        """Get non-existent KB returns 404."""
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_session] = lambda: mock_session
        app.dependency_overrides[get_rag_kernel] = lambda: mock_rag_kernel

        with patch("web.routers.kb.KBService") as mock_service:
            mock_service.get_kb.return_value = None

            client = TestClient(app)
            response = client.get("/api/kb/nonexistent")

            assert response.status_code == 404
            assert "not found" in response.json()["detail"]

    def test_list_kbs_success(self, mock_rag_kernel, mock_session, mock_kb):
        """List KBs returns all knowledge bases."""
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_session] = lambda: mock_session
        app.dependency_overrides[get_rag_kernel] = lambda: mock_rag_kernel

        with patch("web.routers.kb.KBService") as mock_service:
            mock_service.list_kbs.return_value = [mock_kb]

            client = TestClient(app)
            response = client.get("/api/kb")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["id"] == "kb-123"

    def test_list_kbs_empty(self, mock_rag_kernel, mock_session):
        """List KBs returns empty list when no KBs exist."""
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_session] = lambda: mock_session
        app.dependency_overrides[get_rag_kernel] = lambda: mock_rag_kernel

        with patch("web.routers.kb.KBService") as mock_service:
            mock_service.list_kbs.return_value = []

            client = TestClient(app)
            response = client.get("/api/kb")

            assert response.status_code == 200
            assert response.json() == []

    def test_delete_kb_success(self, mock_rag_kernel, mock_session):
        """Delete KB returns success message."""
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_session] = lambda: mock_session
        app.dependency_overrides[get_rag_kernel] = lambda: mock_rag_kernel

        with patch("web.routers.kb.KBService") as mock_service:
            mock_service.delete_kb.return_value = True

            client = TestClient(app)
            response = client.delete("/api/kb/kb-123")

            assert response.status_code == 200
            assert "deleted successfully" in response.json()["message"]

    def test_delete_kb_not_found(self, mock_rag_kernel, mock_session):
        """Delete non-existent KB returns 404."""
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_session] = lambda: mock_session
        app.dependency_overrides[get_rag_kernel] = lambda: mock_rag_kernel

        with patch("web.routers.kb.KBService") as mock_service:
            mock_service.delete_kb.return_value = False

            client = TestClient(app)
            response = client.delete("/api/kb/nonexistent")

            assert response.status_code == 404


class TestKBModels:
    """Tests for KB Pydantic models."""

    def test_kb_create_request_defaults(self):
        """KBCreateRequest has correct defaults."""
        req = KBCreateRequest(name="Test", embedder_name="test-embedder")

        assert req.description is None
        assert req.vdb_type == "chroma"
        assert req.chunk_size == 1000
        assert req.chunk_overlap == 100

    def test_kb_response_model(self):
        """KBResponse validates correctly."""
        resp = KBResponse(
            id="kb-1",
            name="Test",
            description="desc",
            vdb_type="chroma",
            embedder_name="embedder",
            collection_name="coll",
            chunk_size=500,
            chunk_overlap=50,
        )

        assert resp.id == "kb-1"
        assert resp.name == "Test"
