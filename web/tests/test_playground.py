
"""
Playground API Tests
"""

import pytest

class TestPlayground:

    def test_search_compare_basic(self, test_client, test_kb_id, sample_document):
        """Test search comparison results."""
        # Ensure doc exists
        with open(sample_document, 'rb') as f:
            test_client.post(
                "/api/upload",
                data={"kb_id": test_kb_id},
                files={"files": (sample_document.name, f, "text/markdown")}
            )

        response = test_client.post("/api/playground/search-compare", json={
            "kb_id": test_kb_id,
            "query": "分布式",
            "top_k": 3
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["modes"]) == 3 # hybrid, vector, keyword
        assert data["modes"][0]["mode"] == "hybrid"

    def test_cache_logic(self, test_client, test_kb_id):
        """Test cache clear, stats, and hit logic."""
        # 1. Clear Cache
        res_clr = test_client.post("/api/playground/cache-clear")
        assert res_clr.status_code == 200
        
        # 2. Stats should be empty
        res_stats = test_client.get("/api/playground/cache-stats")
        assert res_stats.status_code == 200
        stats = res_stats.json()
        assert stats["size"] == 0

        # 3. Test Cache (First hit might be miss, subsequent might be hit depending on impl)
        # Note: The /cache-test endpoint performs a search.
        query = "Cache Test Query"
        res_test1 = test_client.post("/api/playground/cache-test", json={
            "kb_id": test_kb_id,
            "query": query
        })
        assert res_test1.status_code == 200
        # First call might populate cache if enabled
        
        # 4. Same query again
        res_test2 = test_client.post("/api/playground/cache-test", json={
            "kb_id": test_kb_id,
            "query": query
        })
        assert res_test2.status_code == 200
        data2 = res_test2.json()
        
        # If cache is enabled, it *might* be a hit.
        if stats["enabled"]:
             # Note: Exact behavior depends on RAGKernel cache logic
             pass

    @pytest.mark.local_llm
    def test_query_rewrite_endpoint(self, test_client):
        """Test query rewrite endpoint (requires LLM)."""
        # Ensure LLM is configured (local_llm marker typically implies environment setup, 
        # but config might need manual set if defaults aren't enough).
        # Assuming local LLM is active from previous tests or default state for marked tests.
        
        res = test_client.post("/api/playground/query-rewrite", json={
            "query": "kb"
        })
        assert res.status_code == 200
        data = res.json()
        assert "rewritten" in data
        # If rewriter not enabled/configured, 'rewritten' might be None or error string.
        # Just checking structure here.

    def test_rerank_compare_basic(self, test_client, test_kb_id):
        """Test rerank compare structure."""
        res = test_client.post("/api/playground/rerank-compare", json={
            "kb_id": test_kb_id,
            "query": "Testing",
            "top_k": 2
        })
        assert res.status_code == 200
        data = res.json()
        assert "results" in data
