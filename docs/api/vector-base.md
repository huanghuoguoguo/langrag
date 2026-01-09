# Base Vector Store

Abstract base class for vector store implementations.

## BaseVector

```python
from langrag import BaseVector

class CustomVectorStore(BaseVector):
    def add_texts(self, documents: list[Document]) -> None:
        # Add documents to store
        pass

    def search(self, query_vector: list[float], top_k: int) -> list[SearchResult]:
        # Search by vector
        pass

    def delete(self, ids: list[str]) -> None:
        # Delete by IDs
        pass
```

### Abstract Methods

| Method | Description |
|--------|-------------|
| `add_texts(documents)` | Add documents to the store |
| `search(query_vector, top_k)` | Search by vector similarity |
| `keyword_search(query, top_k)` | Full-text search |
| `hybrid_search(query_vector, query, top_k)` | Combined search |
| `delete(ids)` | Delete documents by ID |

### Implementations

- [DuckDB Vector Store](vector-duckdb.md)
- [SeekDB Vector Store](vector-seekdb.md)
