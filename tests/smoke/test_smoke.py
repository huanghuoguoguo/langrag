"""
Enhanced Smoke Tests for LangRAG
================================
These tests verify that the core system components can be instantiated and
work together in a realistic flow. They serve as a "sanity check" before deployment.

Key coverage:
1. Component Instantiation
2. Document Indexing Flow (Parser -> Splitter -> VDB)
3. RAG Chat Flow (Query -> Retrieve -> Generate)
4. Parent-Child Indexing Flow
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

from langrag.entities.document import Document
from langrag.entities.dataset import Dataset
from langrag.index_processor.extractor import SimpleTextParser
from langrag.index_processor.splitter import RecursiveCharacterChunker
from langrag.index_processor.processor.parent_child import ParentChildIndexProcessor
from langrag.datasource.kv.in_memory import InMemoryKV
from langrag.datasource.vdb.duckdb import DuckDBVector

from web.core.rag_kernel import RAGKernel


@pytest.mark.smoke
class TestCoreComponentsSmoke:
    """Basic instantiation tests."""
    
    def test_components_instantiation(self):
        """Verify core components can be instantiated."""
        parser = SimpleTextParser()
        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=0)
        dataset = Dataset(name="test", collection_name="smoke_test")
        kv = InMemoryKV()
        
        assert parser is not None
        assert chunker is not None
        assert dataset is not None
        assert kv is not None

    def test_rag_kernel_instantiation(self):
        """Verify RAGKernel can be instantiated."""
        kernel = RAGKernel()
        assert kernel is not None
        assert kernel.vector_stores is not None
        assert kernel.kv_store is not None


@pytest.mark.smoke
class TestIndexingFlowSmoke:
    """Test the document indexing pipeline."""
    
    @pytest.fixture
    def temp_text_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a sample document for smoke testing. It contains multiple sentences for chunking.")
            path = f.name
        yield path
        os.remove(path)
    
    @pytest.fixture
    def temp_db_path(self):
        # Just generate a temp path, don't create file (DuckDB will create it)
        path = tempfile.mktemp(suffix='.duckdb')
        yield path
        if os.path.exists(path):
            os.remove(path)

    def test_basic_indexing_flow(self, temp_text_file, temp_db_path):
        """Smoke test: Parse -> Chunk -> Index -> Search."""
        # 1. Parse
        parser = SimpleTextParser()
        docs = parser.parse(temp_text_file)
        assert len(docs) > 0
        
        # 2. Chunk
        chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.split(docs)
        assert len(chunks) > 0
        
        # 3. Index (using real DuckDB)
        dataset = Dataset(name="smoke", collection_name="smoke_col")
        store = DuckDBVector(dataset=dataset, database_path=temp_db_path)
        
        # Simulate embedding (we just assign fake vectors)
        for chunk in chunks:
            chunk.vector = [0.1] * 10
        
        store.create(chunks)
        
        # 4. Search
        results = store.search("sample", query_vector=[0.1]*10, top_k=1)
        assert len(results) >= 1


@pytest.mark.smoke
class TestRAGChatFlowSmoke:
    """Test the full RAG chat pipeline with mocked LLM."""
    
    @pytest.mark.asyncio
    async def test_full_chat_flow(self):
        """Smoke test: Query -> Retrieve -> Generate Answer."""
        # 1. Setup Kernel
        kernel = RAGKernel()
        
        # 2. Mock LLM Client
        mock_llm_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "This is a smoke test answer."
        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        kernel.llm_client = mock_llm_client
        kernel.llm_config = {"model": "mock-model", "temperature": 0, "max_tokens": 100}
        
        # 3. Mock Embedder
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.1] * 10]
        kernel.embedder = mock_embedder
        
        # 4. Mock Vector Store with some data
        mock_vdb = MagicMock()
        mock_vdb.search.return_value = [
            Document(page_content="Relevant context for smoke test.", metadata={"score": 0.9})
        ]
        kernel.vector_stores["smoke_kb"] = mock_vdb
        
        # 5. Execute Chat
        result = await kernel.chat(kb_ids=["smoke_kb"], query="What is this?")
        
        # 6. Assertions
        assert result["answer"] == "This is a smoke test answer."
        assert len(result["sources"]) == 1
        assert "Relevant context" in result["sources"][0]["content"]


@pytest.mark.smoke
class TestParentChildFlowSmoke:
    """Test Parent-Child indexing flow."""
    
    def test_parent_child_indexing(self):
        """Smoke test: Parent-Child splitting and KV storage."""
        # 1. Setup Components
        kv = InMemoryKV()
        mock_vdb = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.1]*10, [0.2]*10]
        
        parent_splitter = MagicMock()
        child_splitter = MagicMock()
        
        # 2. Setup Mock Returns
        parent_doc = Document(page_content="Parent Content", id="p1")
        child_doc = Document(page_content="Child Content", id="c1")
        
        parent_splitter.split_documents.return_value = [parent_doc]
        child_splitter.split_documents.return_value = [child_doc]
        
        # 3. Create Processor
        processor = ParentChildIndexProcessor(
            vector_store=mock_vdb,
            kv_store=kv,
            embedder=mock_embedder,
            parent_splitter=parent_splitter,
            child_splitter=child_splitter
        )
        
        # 4. Run
        original_doc = Document(page_content="Full Original Text", metadata={"document_id": "orig1"})
        dataset = Dataset(name="pc_smoke", collection_name="pc_col")
        processor.process(dataset, [original_doc])
        
        # 5. Verify KV Store has parent
        assert kv.get("p1") == "Parent Content"
        
        # 6. Verify VDB was called
        mock_vdb.create.assert_called_once()


@pytest.mark.smoke
class TestStreamingFlowSmoke:
    """Test streaming chat flow."""
    
    @pytest.mark.asyncio
    async def test_streaming_chat(self):
        """Smoke test: Streaming chat yields chunks."""
        import json
        
        # 1. Setup Kernel
        kernel = RAGKernel()
        kernel.llm_config = {"model": "mock", "temperature": 0, "max_tokens": 100}
        
        # 2. Mock Streaming LLM
        async def mock_stream():
            chunks = ["Hello", " ", "World"]
            for c in chunks:
                chunk = MagicMock()
                chunk.choices[0].delta.content = c
                yield chunk
        
        mock_llm_client = MagicMock()
        mock_llm_client.chat.completions.create = AsyncMock(return_value=mock_stream())
        kernel.llm_client = mock_llm_client
        
        # 3. Mock retrieval
        kernel.multi_search = MagicMock(return_value=([], "none"))
        
        # 4. Execute
        generator = await kernel.chat(kb_ids=[], query="test", stream=True)
        
        # 5. Consume
        items = []
        async for item in generator:
            items.append(item)
        
        # 6. Verify
        assert len(items) > 0
        first = json.loads(items[0])
        assert first["type"] == "sources"
