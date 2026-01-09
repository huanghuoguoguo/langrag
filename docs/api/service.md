# Retrieval Service

High-level retrieval service with caching and enhancement support.

## RetrievalService

```python
from langrag import RetrievalService

service = RetrievalService(
    embedder=embedder,
    reranker=reranker,      # Optional
    rewriter=rewriter,      # Optional
    cache=semantic_cache,   # Optional
)

results = service.search(
    vector_store=vector_store,
    query="What is Python?",
    top_k=5,
    use_cache=True,
    use_rewriter=True,
    use_reranker=True,
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `embedder` | BaseEmbedder | Embedding model |
| `reranker` | BaseReranker \| None | Optional reranker |
| `rewriter` | BaseRewriter \| None | Optional query rewriter |
| `cache` | SemanticCache \| None | Optional semantic cache |

### Methods

#### search()

```python
results = service.search(
    vector_store: BaseVector,
    query: str,
    top_k: int = 10,
    use_cache: bool = True,
    use_rewriter: bool = False,
    use_reranker: bool = False,
)
```

### Example

```python
from langrag import RetrievalService, SemanticCache

# Setup with caching
cache = SemanticCache(
    similarity_threshold=0.95,
    ttl_seconds=3600
)

service = RetrievalService(
    embedder=my_embedder,
    cache=cache
)

# First search - computes and caches
results1 = service.search(
    vector_store=store,
    query="What is Python?",
    use_cache=True
)

# Similar query - returns cached result
results2 = service.search(
    vector_store=store,
    query="What's Python?",  # Cache hit!
    use_cache=True
)
```
