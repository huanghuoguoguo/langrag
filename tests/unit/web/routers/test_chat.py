"""Tests for Chat router."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from web.core.database import get_session
from web.routers.chat import (
    ChatRequest,
    ChatResponse,
    Message,
    SourceItem,
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


@pytest.fixture
def app(mock_session, mock_rag_kernel):
    """Create test FastAPI app with dependency overrides."""
    app = FastAPI()
    app.include_router(router)

    # Override dependencies
    app.dependency_overrides[get_session] = lambda: mock_session
    app.dependency_overrides[get_rag_kernel] = lambda: mock_rag_kernel

    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestChatRouter:
    """Tests for Chat router endpoints."""

    def test_chat_success(self, mock_rag_kernel, mock_session, mock_kb):
        """Chat returns answer with sources."""
        # Setup mock
        mock_rag_kernel.chat = AsyncMock(
            return_value={
                "answer": "This is the answer.",
                "sources": [
                    {
                        "content": "Source content",
                        "score": 0.9,
                        "source": "doc.txt",
                    }
                ],
            }
        )

        # Create app with overrides
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_session] = lambda: mock_session
        app.dependency_overrides[get_rag_kernel] = lambda: mock_rag_kernel

        client = TestClient(app)
        response = client.post(
            "/api/chat",
            json={"query": "What is this?", "kb_ids": ["kb-123"], "stream": False},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "This is the answer."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["content"] == "Source content"

    def test_chat_auto_select_kbs(self, mock_rag_kernel, mock_session, mock_kb):
        """Chat auto-selects all KBs when none provided."""
        from unittest.mock import patch

        mock_rag_kernel.chat = AsyncMock(
            return_value={"answer": "Answer", "sources": []}
        )

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_session] = lambda: mock_session
        app.dependency_overrides[get_rag_kernel] = lambda: mock_rag_kernel

        with patch("web.routers.chat.KBService") as mock_service:
            mock_service.list_kbs.return_value = [mock_kb]

            client = TestClient(app)
            response = client.post(
                "/api/chat",
                json={"query": "Question", "kb_ids": [], "stream": False},
            )

            assert response.status_code == 200
            # Verify KBService.list_kbs was called to get all KBs
            mock_service.list_kbs.assert_called_once()

    def test_chat_llm_not_configured(self, mock_rag_kernel, mock_session):
        """Chat returns 400 when LLM not configured."""
        mock_rag_kernel.chat = AsyncMock(side_effect=ValueError("LLM not configured"))

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_session] = lambda: mock_session
        app.dependency_overrides[get_rag_kernel] = lambda: mock_rag_kernel

        client = TestClient(app)
        response = client.post(
            "/api/chat",
            json={"query": "Question", "kb_ids": ["kb-1"], "stream": False},
        )

        assert response.status_code == 400
        assert "LLM not configured" in response.json()["detail"]

    def test_chat_internal_error(self, mock_rag_kernel, mock_session):
        """Chat returns 500 on internal error."""
        mock_rag_kernel.chat = AsyncMock(side_effect=Exception("Internal error"))

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_session] = lambda: mock_session
        app.dependency_overrides[get_rag_kernel] = lambda: mock_rag_kernel

        client = TestClient(app)
        response = client.post(
            "/api/chat",
            json={"query": "Question", "kb_ids": ["kb-1"], "stream": False},
        )

        assert response.status_code == 500
        assert "Internal error" in response.json()["detail"]

    def test_chat_with_history(self, mock_rag_kernel, mock_session):
        """Chat handles conversation history."""
        mock_rag_kernel.chat = AsyncMock(
            return_value={"answer": "Follow-up answer", "sources": []}
        )

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_session] = lambda: mock_session
        app.dependency_overrides[get_rag_kernel] = lambda: mock_rag_kernel

        client = TestClient(app)
        response = client.post(
            "/api/chat",
            json={
                "query": "Follow-up question",
                "kb_ids": ["kb-1"],
                "history": [
                    {"role": "user", "content": "First question"},
                    {"role": "assistant", "content": "First answer"},
                ],
                "stream": False,
            },
        )

        assert response.status_code == 200
        # Verify history was passed to kernel
        call_args = mock_rag_kernel.chat.call_args
        assert len(call_args[1]["history"]) == 2


class TestChatModels:
    """Tests for Chat Pydantic models."""

    def test_message_model(self):
        """Message validates correctly."""
        msg = Message(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_request_defaults(self):
        """ChatRequest has correct defaults."""
        req = ChatRequest(query="test")

        assert req.kb_ids == []
        assert req.history == []
        assert req.stream is False

    def test_source_item_optional_fields(self):
        """SourceItem handles optional fields."""
        item = SourceItem(
            content="content",
            score=0.9,
            source="source.txt",
        )

        assert item.kb_id is None
        assert item.kb_name is None
        assert item.title is None
        assert item.link is None
        assert item.type is None

    def test_source_item_all_fields(self):
        """SourceItem with all fields."""
        item = SourceItem(
            content="content",
            score=0.9,
            source="source.txt",
            kb_id="kb-1",
            kb_name="KB Name",
            title="Document Title",
            link="http://example.com",
            type="pdf",
        )

        assert item.kb_id == "kb-1"
        assert item.kb_name == "KB Name"
        assert item.title == "Document Title"
        assert item.link == "http://example.com"
        assert item.type == "pdf"

    def test_chat_response(self):
        """ChatResponse validates correctly."""
        resp = ChatResponse(
            answer="This is the answer",
            sources=[SourceItem(content="src", score=0.8, source="doc.txt")],
        )

        assert resp.answer == "This is the answer"
        assert len(resp.sources) == 1
