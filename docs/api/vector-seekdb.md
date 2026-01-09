# SeekDB Vector Store

SeekDB-based high-performance vector store.

## SeekDBVector

```python
from langrag.datasource.vdb.seekdb import SeekDBVector

store = SeekDBVector(
    collection_name="my_collection",
    dimension=384,  # all-MiniLM-L6-v2 default
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

- High-performance vector search
- Built-in embedding support (all-MiniLM-L6-v2)
- Hybrid search with keyword matching
- Optimized for large datasets

### Methods

```python
# Add documents
store.add_texts(documents)

# Vector search
results = store.search(query_vector, top_k=10)

# Keyword search
results = store.keyword_search("search query", top_k=10)

# Hybrid search
results = store.hybrid_search(
    query_vector=query_vector,
    query_text="search query",
    top_k=10
)

# Delete
store.delete(["id1", "id2"])
```

### Example

```python
from langrag import Document
from langrag.datasource.vdb.seekdb import SeekDBVector

# Create store with SeekDB's built-in embedder (384 dim)
store = SeekDBVector(
    collection_name="docs",
    dimension=384,
    persist_directory="./seekdb_data"
)

# Add documents
store.add_texts(documents)

# Search
results = store.hybrid_search(
    query_vector=query_vector,
    query_text="Python programming",
    top_k=5
)
```
