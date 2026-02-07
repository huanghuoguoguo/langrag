"""Tests for TreeAgentRetriever."""

import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from langrag.entities.document import Document, DocumentType
from langrag.retrieval.search.tree_agent import TreeAgentRetriever


class TestTreeAgentRetriever:
    """Tests for TreeAgentRetriever class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.embed_query_async = AsyncMock(return_value=[0.1, 0.2, 0.3])
        llm.chat_async = AsyncMock(return_value="YES")
        return llm

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        store = MagicMock()
        store.search_async = AsyncMock(return_value=[])
        return store

    @pytest.fixture
    def sample_summary_docs(self):
        """Create sample summary documents."""
        return [
            Document(
                id="summary-1",
                page_content="Summary of Chapter 1: Financial Overview",
                metadata={
                    "node_id": "node-1",
                    "is_summary": True,
                    "is_leaf": False,
                    "level": 1,
                    "children_ids": ["leaf-1", "leaf-2"],
                    "title": "Chapter 1",
                },
            ),
            Document(
                id="summary-2", 
                page_content="Summary of Chapter 2: Risk Assessment",
                metadata={
                    "node_id": "node-2",
                    "is_summary": True,
                    "is_leaf": False,
                    "level": 1,
                    "children_ids": ["leaf-3"],
                    "title": "Chapter 2",
                },
            ),
        ]

    @pytest.fixture
    def sample_leaf_docs(self):
        """Create sample leaf documents."""
        return [
            Document(
                id="leaf-1",
                page_content="Detailed content about revenue streams.",
                metadata={
                    "node_id": "leaf-1",
                    "is_summary": False,
                    "is_leaf": True,
                    "level": 2,
                    "parent_id": "node-1",
                },
            ),
            Document(
                id="leaf-2",
                page_content="Expense breakdown and analysis.",
                metadata={
                    "node_id": "leaf-2",
                    "is_summary": False,
                    "is_leaf": True,
                    "level": 2,
                    "parent_id": "node-1",
                },
            ),
        ]

    def test_init(self, mock_llm, mock_vector_store):
        """Initialize retriever with components."""
        retriever = TreeAgentRetriever(
            llm=mock_llm,
            vector_store=mock_vector_store,
            max_steps=5,
        )

        assert retriever.llm == mock_llm
        assert retriever.vector_store == mock_vector_store
        assert retriever.max_steps == 5

    def test_init_default_max_steps(self, mock_llm, mock_vector_store):
        """Default max_steps is 3."""
        retriever = TreeAgentRetriever(
            llm=mock_llm,
            vector_store=mock_vector_store,
        )

        assert retriever.max_steps == 3

    @pytest.mark.asyncio
    async def test_retrieve_no_results(self, mock_llm, mock_vector_store):
        """Returns empty list when no results found."""
        mock_vector_store.search_async.return_value = []

        retriever = TreeAgentRetriever(llm=mock_llm, vector_store=mock_vector_store)
        results = await retriever.retrieve("test query")

        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_fallback_no_summaries(
        self, mock_llm, mock_vector_store, sample_leaf_docs
    ):
        """Falls back to standard search when no summary nodes found."""
        # Return only leaf docs (no summaries)
        mock_vector_store.search_async.return_value = sample_leaf_docs

        retriever = TreeAgentRetriever(llm=mock_llm, vector_store=mock_vector_store)
        results = await retriever.retrieve("test query", top_k=2)

        # Should return the leaf docs directly
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_retrieve_with_summaries(
        self, mock_llm, mock_vector_store, sample_summary_docs
    ):
        """Processes summary nodes correctly."""
        mock_vector_store.search_async.return_value = sample_summary_docs

        retriever = TreeAgentRetriever(llm=mock_llm, vector_store=mock_vector_store)
        results = await retriever.retrieve("financial overview", top_k=2)

        # LLM should be called for relevance check
        assert mock_llm.chat_async.called

    @pytest.mark.asyncio
    async def test_check_relevance_yes(self, mock_llm, mock_vector_store):
        """Relevance check returns True for YES response."""
        mock_llm.chat_async.return_value = "YES, this section is relevant."

        retriever = TreeAgentRetriever(llm=mock_llm, vector_store=mock_vector_store)

        doc = Document(
            page_content="Financial summary",
            metadata={"is_summary": True},
        )
        result = await retriever._check_relevance("budget question", doc)

        assert result is True

    @pytest.mark.asyncio
    async def test_check_relevance_no(self, mock_llm, mock_vector_store):
        """Relevance check returns False for NO response."""
        mock_llm.chat_async.return_value = "NO, this is not relevant."

        retriever = TreeAgentRetriever(llm=mock_llm, vector_store=mock_vector_store)

        doc = Document(
            page_content="HR policies",
            metadata={"is_summary": True},
        )
        result = await retriever._check_relevance("financial question", doc)

        assert result is False

    @pytest.mark.asyncio
    async def test_check_relevance_error_fallback(self, mock_llm, mock_vector_store):
        """Relevance check returns True on error (conservative fallback)."""
        mock_llm.chat_async.side_effect = Exception("LLM error")

        retriever = TreeAgentRetriever(llm=mock_llm, vector_store=mock_vector_store)

        doc = Document(
            page_content="Some content",
            metadata={"is_summary": True},
        )
        result = await retriever._check_relevance("query", doc)

        # Conservative fallback: assume relevant on error
        assert result is True

    @pytest.mark.asyncio
    async def test_retrieve_respects_top_k(
        self, mock_llm, mock_vector_store
    ):
        """Retrieve respects top_k parameter."""
        # Create many leaf docs
        many_leaves = [
            Document(
                id=f"leaf-{i}",
                page_content=f"Content {i}",
                metadata={"is_leaf": True, "is_summary": False},
            )
            for i in range(10)
        ]
        mock_vector_store.search_async.return_value = many_leaves

        retriever = TreeAgentRetriever(llm=mock_llm, vector_store=mock_vector_store)
        results = await retriever.retrieve("query", top_k=3)

        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_retrieve_embeds_query(self, mock_llm, mock_vector_store):
        """Retrieve embeds the query for vector search."""
        mock_vector_store.search_async.return_value = []

        retriever = TreeAgentRetriever(llm=mock_llm, vector_store=mock_vector_store)
        await retriever.retrieve("test query")

        mock_llm.embed_query_async.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_retrieve_passes_query_vector(self, mock_llm, mock_vector_store):
        """Retrieve passes query vector to vector store."""
        mock_vector_store.search_async.return_value = []
        expected_vector = [0.1, 0.2, 0.3]
        mock_llm.embed_query_async.return_value = expected_vector

        retriever = TreeAgentRetriever(llm=mock_llm, vector_store=mock_vector_store)
        await retriever.retrieve("test query")

        call_args = mock_vector_store.search_async.call_args
        assert call_args.kwargs.get("query_vector") == expected_vector


class TestTreeAgentRetrieverExpansion:
    """Tests for tree expansion logic in TreeAgentRetriever."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.embed_query_async = AsyncMock(return_value=[0.1, 0.2, 0.3])
        llm.chat_async = AsyncMock(return_value="YES")
        return llm

    @pytest.fixture
    def mock_vector_store_with_get_by_ids(self):
        """Create a mock vector store with get_by_ids support."""
        store = MagicMock()
        store.search_async = AsyncMock(return_value=[])

        # Support get_by_ids for tree expansion
        async def mock_get_by_ids(ids):
            return [
                Document(
                    id=id,
                    page_content=f"Content for {id}",
                    metadata={"is_leaf": True, "node_id": id},
                )
                for id in ids
            ]

        store.get_by_ids = mock_get_by_ids
        return store

    @pytest.mark.asyncio
    async def test_expand_children_when_supported(
        self, mock_llm, mock_vector_store_with_get_by_ids
    ):
        """Expands children when vector store supports get_by_ids."""
        summary_doc = Document(
            id="summary-1",
            page_content="Chapter Summary",
            metadata={
                "is_summary": True,
                "is_leaf": False,
                "children_ids": ["child-1", "child-2"],
                "node_id": "summary-1",
            },
        )
        mock_vector_store_with_get_by_ids.search_async.return_value = [summary_doc]

        retriever = TreeAgentRetriever(
            llm=mock_llm,
            vector_store=mock_vector_store_with_get_by_ids,
        )
        results = await retriever.retrieve("query", top_k=4)

        # Should have attempted to expand children
        # Results should include leaf nodes
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_no_expansion_without_get_by_ids(self, mock_llm):
        """Does not crash when vector store lacks get_by_ids."""
        store = MagicMock()
        summary_doc = Document(
            id="summary-1",
            page_content="Chapter Summary",
            metadata={
                "is_summary": True,
                "is_leaf": False,
                "children_ids": ["child-1"],
                "node_id": "summary-1",
            },
        )
        store.search_async = AsyncMock(return_value=[summary_doc])
        # No get_by_ids method

        retriever = TreeAgentRetriever(llm=mock_llm, vector_store=store)
        
        # Should not crash
        results = await retriever.retrieve("query")
        
        # Falls back to initial results
        assert len(results) <= 4
