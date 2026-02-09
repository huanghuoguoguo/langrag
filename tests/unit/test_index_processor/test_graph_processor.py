"""Unit tests for GraphIndexProcessor."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from langrag.entities.dataset import Dataset
from langrag.entities.document import Document
from langrag.entities.graph import Entity, Relationship
from langrag.index_processor.processor.graph import GraphIndexProcessor, GraphProcessingError


@pytest.fixture
def mock_graph_store():
    """Mock graph store."""
    store = MagicMock()
    store.add_entities = AsyncMock()
    store.add_relationships = AsyncMock()
    return store


@pytest.fixture
def mock_vector_store():
    """Mock vector store."""
    store = MagicMock()
    store.create = MagicMock()
    store.create_async = AsyncMock()
    return store


@pytest.fixture
def mock_llm():
    """Mock LLM that returns valid JSON."""
    llm = MagicMock()
    llm.chat = MagicMock(return_value=json.dumps({
        "entities": [
            {"name": "Alice", "type": "Person", "properties": {"role": "Engineer"}},
            {"name": "Acme Corp", "type": "Organization", "properties": {}},
        ],
        "relationships": [
            {"source": "Alice", "target": "Acme Corp", "type": "WORKS_AT", "properties": {}}
        ]
    }))
    llm.chat_async = AsyncMock(return_value=json.dumps({
        "entities": [
            {"name": "Alice", "type": "Person", "properties": {"role": "Engineer"}},
            {"name": "Acme Corp", "type": "Organization", "properties": {}},
        ],
        "relationships": [
            {"source": "Alice", "target": "Acme Corp", "type": "WORKS_AT", "properties": {}}
        ]
    }))
    return llm


@pytest.fixture
def mock_embedder():
    """Mock embedder."""
    embedder = MagicMock()
    embedder.embed = MagicMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    embedder.embed_async = AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    embedder.dimension = 3
    return embedder


@pytest.fixture
def sample_dataset():
    """Sample dataset for testing."""
    return Dataset(id="ds-1", name="test-dataset", collection_name="test_collection")


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        Document(
            id="doc-1",
            page_content="Alice works at Acme Corp as a software engineer.",
            metadata={"source": "test.txt"}
        ),
    ]


class TestGraphIndexProcessorInit:
    """Tests for GraphIndexProcessor initialization."""

    def test_init_minimal(self, mock_graph_store):
        """Test initialization with minimal arguments."""
        processor = GraphIndexProcessor(graph_store=mock_graph_store)
        assert processor.graph_store == mock_graph_store
        assert processor.vector_store is None
        assert processor.llm is None
        assert len(processor.entity_types) > 0
        assert len(processor.relationship_types) > 0

    def test_init_full(self, mock_graph_store, mock_vector_store, mock_llm, mock_embedder):
        """Test initialization with all arguments."""
        processor = GraphIndexProcessor(
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            llm=mock_llm,
            embedder=mock_embedder,
            entity_types=["Person", "Organization"],
            relationship_types=["WORKS_AT"],
        )
        assert processor.entity_types == ["Person", "Organization"]
        assert processor.relationship_types == ["WORKS_AT"]


class TestGraphIndexProcessorProcess:
    """Tests for GraphIndexProcessor.process_async."""

    @pytest.mark.asyncio
    async def test_process_requires_llm(self, mock_graph_store, sample_dataset, sample_documents):
        """Test that process raises error without LLM."""
        processor = GraphIndexProcessor(graph_store=mock_graph_store)

        with pytest.raises(GraphProcessingError, match="LLM is required"):
            await processor.process_async(sample_dataset, sample_documents)

    @pytest.mark.asyncio
    async def test_process_extracts_entities(
        self, mock_graph_store, mock_llm, mock_embedder, sample_dataset, sample_documents
    ):
        """Test that process extracts entities from documents."""
        processor = GraphIndexProcessor(
            graph_store=mock_graph_store,
            llm=mock_llm,
            embedder=mock_embedder,
        )

        stats = await processor.process_async(sample_dataset, sample_documents)

        assert stats["total_chunks"] == 1
        assert stats["total_entities"] == 2
        assert stats["failed_chunks"] == 0

        # Verify graph store was called
        mock_graph_store.add_entities.assert_called_once()
        entities = mock_graph_store.add_entities.call_args[0][0]
        assert len(entities) == 2

        entity_names = {e.name for e in entities}
        assert "Alice" in entity_names
        assert "Acme Corp" in entity_names

    @pytest.mark.asyncio
    async def test_process_extracts_relationships(
        self, mock_graph_store, mock_llm, mock_embedder, sample_dataset, sample_documents
    ):
        """Test that process extracts relationships."""
        processor = GraphIndexProcessor(
            graph_store=mock_graph_store,
            llm=mock_llm,
            embedder=mock_embedder,
        )

        stats = await processor.process_async(sample_dataset, sample_documents)

        assert stats["total_relationships"] == 1

        # Verify relationships were added
        mock_graph_store.add_relationships.assert_called_once()
        relationships = mock_graph_store.add_relationships.call_args[0][0]
        assert len(relationships) == 1
        assert relationships[0].type == "WORKS_AT"

    @pytest.mark.asyncio
    async def test_process_embeds_entities(
        self, mock_graph_store, mock_llm, mock_embedder, sample_dataset, sample_documents
    ):
        """Test that process generates embeddings for entities."""
        processor = GraphIndexProcessor(
            graph_store=mock_graph_store,
            llm=mock_llm,
            embedder=mock_embedder,
        )

        await processor.process_async(sample_dataset, sample_documents)

        # Verify embedder was called
        mock_embedder.embed_async.assert_called_once()

        # Verify entities have embeddings
        entities = mock_graph_store.add_entities.call_args[0][0]
        for entity in entities:
            assert entity.embedding is not None

    @pytest.mark.asyncio
    async def test_process_stores_in_vector_store(
        self, mock_graph_store, mock_vector_store, mock_llm, mock_embedder,
        sample_dataset, sample_documents
    ):
        """Test that process stores entities in vector store."""
        processor = GraphIndexProcessor(
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            llm=mock_llm,
            embedder=mock_embedder,
        )

        await processor.process_async(sample_dataset, sample_documents)

        # Verify vector store was called
        mock_vector_store.create_async.assert_called_once()
        docs = mock_vector_store.create_async.call_args[0][0]
        assert len(docs) == 2

        # Verify documents have entity metadata
        for doc in docs:
            assert doc.metadata.get("is_entity") is True
            assert doc.metadata.get("entity_id") is not None

    @pytest.mark.asyncio
    async def test_process_handles_llm_error(
        self, mock_graph_store, mock_embedder, sample_dataset, sample_documents
    ):
        """Test that process handles LLM errors gracefully."""
        mock_llm = MagicMock()
        mock_llm.chat_async = AsyncMock(side_effect=Exception("LLM error"))

        processor = GraphIndexProcessor(
            graph_store=mock_graph_store,
            llm=mock_llm,
            embedder=mock_embedder,
        )

        stats = await processor.process_async(sample_dataset, sample_documents)

        assert stats["failed_chunks"] == 1
        assert len(stats["failed_chunk_ids"]) == 1

    @pytest.mark.asyncio
    async def test_process_deduplicates_entities(
        self, mock_graph_store, mock_embedder, sample_dataset
    ):
        """Test that process deduplicates entities across chunks."""
        # LLM returns same entity in both chunks
        mock_llm = MagicMock()
        mock_llm.chat_async = AsyncMock(return_value=json.dumps({
            "entities": [
                {"name": "Alice", "type": "Person", "properties": {}}
            ],
            "relationships": []
        }))

        # Adjust embedder to return single embedding
        mock_embedder = MagicMock()
        mock_embedder.embed_async = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        documents = [
            Document(id="doc-1", page_content="Alice is an engineer."),
            Document(id="doc-2", page_content="Alice works on AI projects."),
        ]

        processor = GraphIndexProcessor(
            graph_store=mock_graph_store,
            llm=mock_llm,
            embedder=mock_embedder,
        )

        stats = await processor.process_async(sample_dataset, documents)

        # Should deduplicate to 1 entity
        assert stats["total_entities"] == 1

        entities = mock_graph_store.add_entities.call_args[0][0]
        assert len(entities) == 1
        assert entities[0].name == "Alice"
        # Should have both chunk IDs as sources
        assert len(entities[0].source_chunk_ids) == 2


class TestGraphIndexProcessorParsing:
    """Tests for response parsing."""

    def test_parse_valid_json(self, mock_graph_store):
        """Test parsing valid JSON response."""
        processor = GraphIndexProcessor(graph_store=mock_graph_store)

        response = json.dumps({
            "entities": [
                {"name": "Test", "type": "Concept", "properties": {"key": "value"}}
            ],
            "relationships": [
                {"source": "A", "target": "B", "type": "RELATED_TO", "properties": {}}
            ]
        })

        entities, relationships = processor._parse_extraction_response(response, "chunk-1")

        assert len(entities) == 1
        assert entities[0].name == "Test"
        assert entities[0].type == "Concept"
        assert entities[0].properties == {"key": "value"}
        assert entities[0].source_chunk_ids == ["chunk-1"]

        assert len(relationships) == 1
        assert relationships[0].type == "RELATED_TO"

    def test_parse_json_with_markdown(self, mock_graph_store):
        """Test parsing JSON wrapped in markdown code blocks."""
        processor = GraphIndexProcessor(graph_store=mock_graph_store)

        response = """```json
{
    "entities": [{"name": "Test", "type": "Concept", "properties": {}}],
    "relationships": []
}
```"""

        entities, relationships = processor._parse_extraction_response(response, "chunk-1")

        assert len(entities) == 1
        assert entities[0].name == "Test"

    def test_parse_invalid_json(self, mock_graph_store):
        """Test parsing invalid JSON returns empty lists."""
        processor = GraphIndexProcessor(graph_store=mock_graph_store)

        entities, relationships = processor._parse_extraction_response("not json", "chunk-1")

        assert entities == []
        assert relationships == []

    def test_parse_empty_entities(self, mock_graph_store):
        """Test parsing response with empty entities."""
        processor = GraphIndexProcessor(graph_store=mock_graph_store)

        response = json.dumps({"entities": [], "relationships": []})
        entities, relationships = processor._parse_extraction_response(response, "chunk-1")

        assert entities == []
        assert relationships == []

    def test_parse_skips_invalid_entities(self, mock_graph_store):
        """Test that parsing skips entities without name."""
        processor = GraphIndexProcessor(graph_store=mock_graph_store)

        response = json.dumps({
            "entities": [
                {"name": "", "type": "Person"},  # Empty name - skip
                {"type": "Person"},  # No name - skip
                {"name": "Valid", "type": "Person"},  # Valid
            ],
            "relationships": []
        })

        entities, relationships = processor._parse_extraction_response(response, "chunk-1")

        assert len(entities) == 1
        assert entities[0].name == "Valid"


class TestGraphIndexProcessorRelationshipResolution:
    """Tests for relationship entity resolution."""

    def test_resolve_relationships(self, mock_graph_store):
        """Test resolving relationship names to IDs."""
        processor = GraphIndexProcessor(graph_store=mock_graph_store)

        relationships = [
            Relationship(source_id="alice", target_id="acme corp", type="WORKS_AT"),
        ]

        entity_name_to_id = {
            "person:alice": "entity-1",
            "organization:acme corp": "entity-2",
        }

        resolved = processor._resolve_relationships(relationships, entity_name_to_id)

        assert len(resolved) == 1
        assert resolved[0].source_id == "entity-1"
        assert resolved[0].target_id == "entity-2"

    def test_resolve_relationships_missing_entity(self, mock_graph_store):
        """Test that relationships with missing entities are skipped."""
        processor = GraphIndexProcessor(graph_store=mock_graph_store)

        relationships = [
            Relationship(source_id="alice", target_id="unknown", type="KNOWS"),
        ]

        entity_name_to_id = {
            "person:alice": "entity-1",
        }

        resolved = processor._resolve_relationships(relationships, entity_name_to_id)

        assert len(resolved) == 0
