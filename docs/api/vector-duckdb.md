# DuckDB Vector Store

DuckDB-based vector store with full-text search support.

## DuckDBVector

```python
from langrag.datasource.vdb.duckdb import DuckDBVector

store = DuckDBVector(
    collection_name="my_collection",
    dimension=1536,
    persist_directory="./data",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection_name` | str | required | Name of the collection |
| `dimension` | int | required | Embedding dimension |
| `persist_directory` | str | None | Directory for persistence |

### Features

- Embedded database (no server required)
- Full-text search (FTS) with BM25
- Hybrid search with RRF fusion
- HNSW index for fast approximate search

### Methods

```python
# Add documents
store.add_texts(documents)

# Vector search
results = store.search(query_vector, top_k=10)

# Keyword search
results = store.keyword_search("python programming", top_k=10)

# Hybrid search
results = store.hybrid_search(
    query_vector=query_vector,
    query_text="python programming",
    top_k=10
)

# Delete
store.delete(["id1", "id2"])
```

### Example

```python
from langrag import Document
from langrag.datasource.vdb.duckdb import DuckDBVector

# Create store
store = DuckDBVector(
    collection_name="knowledge_base",
    dimension=1536,
    persist_directory="./vector_data"
)

# Add documents
docs = [
    Document(id="1", page_content="Python is...", vector=[...]),
    Document(id="2", page_content="JavaScript is...", vector=[...]),
]
store.add_texts(docs)

# Search
query_vector = embedder.embed(["What is Python?"])[0]
results = store.search(query_vector, top_k=5)
```
