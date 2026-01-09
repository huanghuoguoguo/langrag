# Retrieval Workflow

The main retrieval workflow orchestrating search operations.

## RetrievalWorkflow

```python
from langrag import RetrievalWorkflow

workflow = RetrievalWorkflow(
    vector_store=vector_store,
    embedder=embedder,
    reranker=reranker,  # Optional
    rewriter=rewriter,  # Optional
)

# Search
results = workflow.search(
    query="What is machine learning?",
    top_k=5,
    search_type="hybrid"
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `vector_store` | BaseVector | Vector store backend |
| `embedder` | BaseEmbedder | Embedding model |
| `reranker` | BaseReranker \| None | Optional reranker |
| `rewriter` | BaseRewriter \| None | Optional query rewriter |

### Methods

#### search()

```python
results = workflow.search(
    query: str,
    top_k: int = 10,
    search_type: str = "vector",  # "vector", "keyword", "hybrid"
    filters: dict = None,
    rerank: bool = True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | required | Search query |
| `top_k` | int | 10 | Number of results |
| `search_type` | str | "vector" | Search type |
| `filters` | dict | None | Metadata filters |
| `rerank` | bool | True | Apply reranking |

### Search Types

| Type | Description |
|------|-------------|
| `vector` | Semantic similarity search |
| `keyword` | Full-text search (BM25) |
| `hybrid` | Combined vector + keyword with RRF fusion |

### Example

```python
from langrag import RetrievalWorkflow
from langrag.datasource.vdb.duckdb import DuckDBVector

# Setup
vector_store = DuckDBVector(
    collection_name="docs",
    dimension=1536
)

workflow = RetrievalWorkflow(
    vector_store=vector_store,
    embedder=my_embedder
)

# Vector search
results = workflow.search("What is Python?", top_k=5)

# Hybrid search
results = workflow.search(
    "Python programming",
    search_type="hybrid",
    top_k=10
)
```
