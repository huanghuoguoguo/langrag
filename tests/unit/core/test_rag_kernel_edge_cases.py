import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from web.core.rag_kernel import RAGKernel
from langrag.entities.document import Document

class TestRAGKernelEdgeCases:
    
    @pytest.fixture
    def kernel(self):
        k = RAGKernel()
        # Mock minimal components
        k.llm_client = MagicMock()
        k.llm_config = {"model": "mock-model", "temperature": 0, "max_tokens": 100}
        k.kv_store = MagicMock()
        return k

    @pytest.mark.asyncio
    async def test_empty_retrieval_result(self, kernel):
        """Test behavior when Vector Store returns no results (e.g., empty KB)."""
        # Setup: VDB returns empty
        mock_vdb = MagicMock()
        mock_vdb.search.return_value = [] # Nothing found
        kernel.vector_stores["kb_empty"] = mock_vdb
        
        # Mock LLM generation to just return simple fallback
        # Note: In real kernel, if results are empty, it constructs a prompt saying "Answer based on your knowledge".
        mock_r = MagicMock()
        mock_r.choices[0].message.content = "I don't know."
        kernel.llm_client.chat.completions.create = AsyncMock(return_value=mock_r)
        
        # Execute
        res = await kernel.chat(kb_ids=["kb_empty"], query="Something")
        
        # Verify
        assert res["answer"] == "I don't know."
        assert res["sources"] == [] # Sources must be empty list, not None

    @pytest.mark.asyncio
    async def test_embedder_failure(self, kernel):
        """Test graceful handling when Embedder raises an exception."""
        # Setup: Manually inject a mock embedder that fails
        mock_embedder = MagicMock()
        mock_embedder.embed_query.side_effect = Exception("API Timeout")
        kernel.embedder = mock_embedder
        
        # We need to mock create_vector_store to NOT try to initialize real components
        # We just want to test chat() failing when it calls embed_query()
        # So we manually inject a dummy vector store that would be called if embedder succeeded (or failed inside kernel)
        kernel.vector_stores["kb_fail"] = MagicMock()
        
        # FIX: Ensure LLM client is AsyncMock for await
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "Fallback Answer"
        kernel.llm_client.chat.completions.create = AsyncMock(return_value=mock_resp)
        
        # ACT: Call chat
        res = await kernel.chat(kb_ids=["kb_fail"], query="test")
        
        # ASSERT: It finished successfully.
        assert res["answer"] == "Fallback Answer"

    @pytest.mark.asyncio
    async def test_reranker_failure_fallback(self, kernel):
        """Test fallback to original results if Reranker crashes."""
        # Setup: Normal VDB results
        mock_vdb = MagicMock()
        docs = [Document(page_content="doc1", metadata={"score": 0.5})]
        mock_vdb.search.return_value = docs
        kernel.vector_stores["kb_rerank"] = mock_vdb
        
        # Setup: Reranker that crashes
        mock_reranker = MagicMock()
        mock_reranker.rerank.side_effect = Exception("Rerank Service Down")
        kernel.reranker = mock_reranker
        
        # Setup: LLM
        mock_r = MagicMock()
        mock_r.choices[0].message.content = "Answer"
        kernel.llm_client.chat.completions.create = AsyncMock(return_value=mock_r)

        # Execute
        # Should NOT raise exception, but log error and use original docs
        res = await kernel.chat(kb_ids=["kb_rerank"], query="test")
        
        # Verify
        assert res["sources"][0]["content"] == "doc1"
        # Since rerank failed, the score should remain the original one (or simulate fallback)
        # kernel logic: `results = results[:top_k]` on error.

    @pytest.mark.asyncio
    async def test_long_context_truncation(self, kernel):
        """Test that extremely long contexts don't crash the prompt construction."""
        # Setup: VDB returns HUGE number of docs or Huge content
        mock_vdb = MagicMock()
        # Simulate retrieval of 5 docs, each 10k chars
        long_docs = [Document(page_content="A"*10000, metadata={"score": 0.9}) for _ in range(5)]
        mock_vdb.search.return_value = long_docs
        kernel.vector_stores["kb_long"] = mock_vdb
        
        # Setup LLM
        mock_r = MagicMock()
        mock_r.choices[0].message.content = "Answer"
        kernel.llm_client.chat.completions.create = AsyncMock(return_value=mock_r)
        
        # Execute
        await kernel.chat(kb_ids=["kb_long"], query="test")
        
        # In this test, we mostly check that it DOES NOT raise "Context too long" error 
        # inside the kernel (assuming kernel doesn't validate token length yet).
        # But verifying it creates a call is enough for now.
        assert kernel.llm_client.chat.completions.create.called
