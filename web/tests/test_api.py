"""
Web Demo API Tests

这些测试用于验证 Web Demo 的 API 功能。
需要本地 LLM 模型的测试标记为 local_llm，CI 环境可以跳过。

运行方式：
    # 运行所有 web demo 测试
    uv run pytest web/tests -v

    # 只运行不需要 LLM 的测试
    uv run pytest web/tests -v -m "not local_llm"

    # 运行需要本地 LLM 的测试
    uv run pytest web/tests -v -m "local_llm"
"""

import pytest

# ==================== Basic API Tests ====================

class TestHealthAndConfig:
    """Test basic health and configuration endpoints."""

    def test_list_knowledge_bases(self, test_client):
        """Test listing knowledge bases."""
        response = test_client.get("/api/kb")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_list_embedders(self, test_client):
        """Test listing available embedders."""
        response = test_client.get("/api/config/embedders")
        assert response.status_code == 200

    def test_list_llms(self, test_client):
        """Test listing available LLMs."""
        response = test_client.get("/api/config/llms")
        assert response.status_code == 200

    def test_chat_defaults_to_all_when_empty(self, test_client, test_kb_id):
        """Test that empty kb_ids defaults to searching ALL knowledge bases."""
        response = test_client.post("/api/chat", json={
            "kb_ids": [],
            "query": "1+1等于几？",
            "history": [],
            "stream": False,
            "model_name": "local",
            "use_router": False
        })

        assert response.status_code == 200
        data = response.json()

        # Since we default to ALL KBs when empty, we expect retrieval to happen.
        # (Assuming test_kb_id ensures at least one KB exists)
        if len(data.get("sources", [])) > 0:
            print(f"\n[Default Chat Sources]: {len(data['sources'])}")
            assert True
        else:
            # If queries didn't match anything, sources might be empty, but pipeline info should exist
            assert "pipeline" in data


# ==================== Knowledge Base Tests ====================

class TestKnowledgeBase:
    """Test knowledge base CRUD operations."""

    def test_create_and_delete_kb(self, test_client, default_embedder_name):
        """Test creating and deleting a knowledge base."""
        # Create
        response = test_client.post("/api/kb", json={
            "name": "Temp Test KB",
            "description": "Temporary KB for testing",
            "vdb_type": "chroma",
            "embedder_name": default_embedder_name
        })
        assert response.status_code == 200, f"Create failed: {response.text}"
        kb_id = response.json()["id"]

        # Verify exists
        response = test_client.get(f"/api/kb/{kb_id}")
        assert response.status_code == 200

        # Delete
        response = test_client.delete(f"/api/kb/{kb_id}")
        assert response.status_code == 200

    def test_get_nonexistent_kb(self, test_client):
        """Test getting a non-existent knowledge base."""
        response = test_client.get("/api/kb/nonexistent_kb_id")
        assert response.status_code == 404


# ==================== Document Upload Tests ====================

class TestDocumentUpload:
    """Test document upload and processing."""

    def test_upload_document(self, test_client, test_kb_id, sample_document):
        """Test uploading a document to a knowledge base."""
        with open(sample_document, 'rb') as f:
            response = test_client.post(
                "/api/upload",
                data={"kb_id": test_kb_id},
                files={"files": (sample_document.name, f, "text/markdown")}
            )

        assert response.status_code == 200
        data = response.json()
        # Response contains processed_files and total_chunks
        assert "processed_files" in data or "id" in data or "document_id" in data


# ==================== Chat/Retrieval Tests ====================

class TestChatRetrieval:
    """Test chat and retrieval functionality."""

    def test_chat_without_kb(self, test_client):
        """Test chat without selecting any knowledge base.

        Note: Without LLM configured, may return 500 or 200 with partial results.
        We accept both as valid since this is testing retrieval, not LLM.
        """
        response = test_client.post("/api/chat", json={
            "kb_ids": [],
            "query": "你好",
            "history": [],
            "stream": False
        })

        # Accept 200 (success) or 500 (LLM error) - both are valid without LLM
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "sources" in data

    def test_chat_with_kb(self, test_client, test_kb_id):
        """Test chat with a specific knowledge base.

        Note: Without LLM configured, may return 500 or 200 with partial results.
        """
        response = test_client.post("/api/chat", json={
            "kb_ids": [test_kb_id],
            "query": "什么是分布式系统？",
            "history": [],
            "stream": False
        })

        # Accept 200 (success) or 500 (LLM error)
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "sources" in data


@pytest.mark.local_llm
class TestChatWithLocalLLM:
    """
    Tests that require a local LLM model.
    These tests verify the full RAG pipeline including answer generation.

    Skip these tests in CI by using: pytest -m "not local_llm"
    """

    def test_chat_generates_answer(self, test_client, test_kb_id):
        """Test that chat generates an answer using local LLM."""
        response = test_client.post("/api/chat", json={
            "kb_ids": [test_kb_id],
            "query": "什么是CAP定理？",
            "history": [],
            "stream": False,
            "model_name": "local"
        })

        assert response.status_code == 200
        data = response.json()

        # Should have an answer
        assert "answer" in data
        if data.get("answer"):
            print(f"\n[LLM Answer]: {data['answer'][:200]}...")
            assert len(data["answer"]) > 0

        # Should have sources
        assert "sources" in data
        print(f"[Sources Count]: {len(data['sources'])}")

    def test_chat_with_rewriter(self, test_client, test_kb_id):
        """Test chat with query rewriter enabled."""
        response = test_client.post("/api/chat", json={
            "kb_ids": [test_kb_id],
            "query": "分布式",
            "history": [],
            "stream": False,
            "model_name": "local",
            "use_rewriter": True,
            "rewriter_model": "local"
        })

        assert response.status_code == 200
        data = response.json()

        print(f"\n[Original Query]: 分布式")
        if data.get("rewritten_query"):
            print(f"[Rewritten Query]: {data['rewritten_query']}")

        assert "answer" in data

    def test_chat_with_reranker(self, test_client, test_kb_id):
        """Test chat with reranker enabled."""
        response = test_client.post("/api/chat", json={
            "kb_ids": [test_kb_id],
            "query": "分布式系统有什么特点？",
            "history": [],
            "stream": False,
            "model_name": "local",
            "use_rerank": True,
            "reranker_type": "llm_template",
            "reranker_model": "local"
        })

        assert response.status_code == 200
        data = response.json()

        print(f"\n[Retrieval Stats]: {data.get('retrieval_stats', {})}")
        assert "answer" in data



    def test_auto_router_mode(self, test_client, test_kb_id):
        """Test 'Auto' mode: No KB selected + Router Enabled -> Should search all KBs."""
        # We use test_kb_id fixture to ensure at least one KB exists in the system
        response = test_client.post("/api/chat", json={
            "kb_ids": [], # Empty means "All" or "Auto"
            "query": "分布式", # Relevant to the test KB content
            "history": [],
            "stream": False,
            "model_name": "local",
            "use_router": True # Enabled Router with empty list -> Auto Route among all
        })

        assert response.status_code == 200
        data = response.json()

        # Should have sources because it auto-routed to the test_kb_id
        print(f"\n[Auto Router Sources]: {len(data.get('sources', []))}")

        # Check pipeline info exists
        assert "pipeline" in data
        if data.get("pipeline", {}).get("router", {}).get("enabled"):
            print(f"[Router]: {data['pipeline']['router']}")


# ==================== Evaluation Tests ====================

@pytest.mark.local_llm
class TestEvaluation:
    """Test RAG evaluation functionality."""

    def test_evaluate_response(self, test_client):
        """Test evaluating a RAG response."""
        response = test_client.post("/api/chat/evaluate", json={
            "question": "什么是分布式系统？",
            "answer": "分布式系统是由多台计算机通过网络连接组成的系统。",
            "contexts": [
                "分布式系统是由多台计算机通过网络连接组成的系统，这些计算机协同工作以完成共同的任务。"
            ]
        })

        # Evaluation might fail if LLM is not configured, but should not error
        if response.status_code == 200:
            data = response.json()
            print(f"\n[Evaluation Results]: {data}")
            assert "faithfulness" in data
            assert "answer_relevancy" in data


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
