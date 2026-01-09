# Search Result

Search result structure returned by retrieval operations.

## SearchResult

```python
from langrag import SearchResult

# SearchResult is returned by search operations
results = workflow.search("query", top_k=5)

for result in results:
    print(f"Score: {result.score}")
    print(f"Content: {result.content}")
    print(f"Metadata: {result.metadata}")
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `content` | str | The matched text content |
| `score` | float | Relevance score (higher is better) |
| `metadata` | dict | Associated metadata |
| `id` | str \| None | Document ID if available |
| `vector` | list[float] \| None | Embedding vector if available |

### Example

```python
from langrag import RetrievalWorkflow

# Search returns list of SearchResult
results = workflow.search("What is machine learning?", top_k=3)

for i, result in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Score: {result.score:.4f}")
    print(f"Content: {result.content[:200]}...")
    if result.metadata:
        print(f"Source: {result.metadata.get('source', 'unknown')}")
```
