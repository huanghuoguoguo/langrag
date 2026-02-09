
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langrag.retrieval.graph.graph_retriever import GraphRetriever, GraphRetrieverConfig
from langrag.entities.graph import Entity, Relationship, Subgraph
from langrag.entities.dataset import RetrievalContext

@pytest.fixture
def mock_graph_store():
    store = MagicMock()
    store.search_entities = AsyncMock()
    store.get_neighbors = AsyncMock()
    store.get_entity = AsyncMock()
    return store

@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed_async = AsyncMock()
    embedder.embed = MagicMock()
    return embedder

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat_async = AsyncMock()
    llm.chat = MagicMock()
    return llm

@pytest.fixture
def retriever(mock_graph_store, mock_embedder, mock_llm):
    config = GraphRetrieverConfig(
        use_llm_recognition=True,
        use_vector_recognition=True
    )
    return GraphRetriever(
        graph_store=mock_graph_store,
        embedder=mock_embedder,
        llm=mock_llm,
        config=config
    )


def test_init_validation(mock_graph_store):
    # Test invalid config raising ValueError
    with pytest.raises(ValueError, match="At least one recognition method"):
        GraphRetriever(
            mock_graph_store,
            config=GraphRetrieverConfig(
                use_llm_recognition=False,
                use_vector_recognition=False
            )
        )



    # Test fallback warning (logging warning, not python warning)
    r = GraphRetriever(
        mock_graph_store,
        embedder=MagicMock(),
        config=GraphRetrieverConfig(use_llm_recognition=True),
        llm=None
    )
    assert r.config.use_llm_recognition is False

@pytest.mark.asyncio
async def test_recognize_entities_llm_success(retriever, mock_llm, mock_graph_store):
    query = "Who is Steve Jobs?"
    
    # Mock LLM response
    mock_llm.chat_async.return_value = '[\n  {"name": "Steve Jobs", "type": "Person"}\n]'
    
    # Mock GraphStore search
    entity = Entity(id="1", name="Steve Jobs", type="Person")
    mock_graph_store.search_entities.return_value = [entity]
    
    entities = await retriever._recognize_entities(query)
    
    assert len(entities) == 1
    assert entities[0].name == "Steve Jobs"
    mock_llm.chat_async.assert_called_once()
    mock_graph_store.search_entities.assert_called_once_with(query="Steve Jobs", top_k=1, threshold=0.7)

@pytest.mark.asyncio
async def test_recognize_entities_llm_failure_fallback(retriever, mock_llm, mock_embedder, mock_graph_store):
    query = "Who is Steve Jobs?"
    
    # Mock LLM failure (e.g. invalid JSON)
    mock_llm.chat_async.return_value = "Not JSON"
    
    # Mock Vector search success
    mock_embedder.embed_async.return_value = [[0.1, 0.2]]
    entity = Entity(id="1", name="Steve Jobs", type="Person")
    mock_graph_store.search_entities.side_effect = [
        [], # First call from LLM part (if it reached there? No, because json parse fails first)
        [entity] # Second call from vector part
    ] 
    # Actually if LLM fails, it logs warning and goes to vector
    # vector search calls graph_store.search_entities with query_vector
    
    # Reset mock to be clear
    mock_graph_store.search_entities.side_effect = None
    mock_graph_store.search_entities.return_value = [entity]
    
    entities = await retriever._recognize_entities(query)
    
    assert len(entities) == 1
    assert entities[0].name == "Steve Jobs"
    # LLM called
    mock_llm.chat_async.assert_called_once()
    # Embedder called
    mock_embedder.embed_async.assert_called_once()
    # Graph store search called with vector
    mock_graph_store.search_entities.assert_called_with(
        query_vector=[0.1, 0.2],
        top_k=retriever.config.vector_top_k,
        threshold=retriever.config.similarity_threshold
    )

@pytest.mark.asyncio
async def test_retrieve_async_flow(retriever, mock_graph_store, mock_llm):
    query = "context query"
    
    # 1. Recognize Entities (Simulate LLM success)
    mock_llm.chat_async.return_value = '[{"name": "E1", "type": "T"}]'
    e1 = Entity(id="e1", name="E1", type="T")
    mock_graph_store.search_entities.return_value = [e1]
    
    # 2. Get Neighbors
    e2 = Entity(id="e2", name="E2", type="T")
    rel = Relationship(id="r1", source_id="e1", target_id="e2", type="REL")
    subgraph = Subgraph(entities=[e1, e2], relationships=[rel])
    
    mock_graph_store.get_neighbors.return_value = subgraph
    
    # Execute
    contexts = await retriever.retrieve_async(query)
    
    assert len(contexts) >= 1
    assert contexts[0].document_id == "graph_context"
    assert "E1" in contexts[0].content
    assert "E2" in contexts[0].content
    assert "REL" in contexts[0].content

@pytest.mark.asyncio
async def test_retrieve_async_no_entities(retriever, mock_llm, mock_embedder):
    # Mock no recognition
    mock_llm.chat_async.return_value = "[]"
    mock_embedder.embed_async.return_value = [[0.0]]
    # graph store search with vector returns empty
    retriever.graph_store.search_entities.return_value = []
    
    contexts = await retriever.retrieve_async("query")
    assert len(contexts) == 0

@pytest.mark.asyncio
async def test_retrieve_async_empty_subgraph(retriever, mock_llm, mock_graph_store):
    # Recognize entities
    mock_llm.chat_async.return_value = '[{"name": "E1", "type": "T"}]'
    e1 = Entity(id="e1", name="E1", type="T")
    mock_graph_store.search_entities.return_value = [e1]
    
    # Empty subgraph
    mock_graph_store.get_neighbors.return_value = Subgraph(entities=[], relationships=[])
    
    contexts = await retriever.retrieve_async("query")
    assert len(contexts) == 0

@pytest.mark.asyncio
async def test_get_entity_context(retriever, mock_graph_store):
    e1 = Entity(id="e1", name="E1", type="T")
    mock_graph_store.get_entity.return_value = e1
    
    subgraph = Subgraph(entities=[e1], relationships=[])
    mock_graph_store.get_neighbors.return_value = subgraph
    
    ctx = await retriever.get_entity_context("e1")
    

    assert ctx.metadata["entity_name"] == "E1"

@pytest.mark.asyncio
async def test_retrieve_subgraph(retriever, mock_graph_store, mock_llm):
    # Mock recognition
    mock_llm.chat_async.return_value = '[{"name": "E1", "type": "T"}]'
    e1 = Entity(id="e1", name="E1", type="T")
    mock_graph_store.search_entities.return_value = [e1]

    # Mock neighbors
    e2 = Entity(id="e2", name="E2", type="T")
    rel = Relationship(id="r1", source_id="e1", target_id="e2", type="REL")
    subgraph = Subgraph(entities=[e1, e2], relationships=[rel])
    
    mock_graph_store.get_neighbors.return_value = subgraph

    result_subgraph = await retriever.retrieve_subgraph("query")

    assert len(result_subgraph.entities) == 2
    assert len(result_subgraph.relationships) == 1

