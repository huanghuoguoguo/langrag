# Semantic Cache

Query caching based on embedding similarity.

## SemanticCache

```python
from langrag import SemanticCache

cache = SemanticCache(
    similarity_threshold=0.95,  # Minimum similarity for cache hit
    ttl_seconds=3600,           # Time-to-live in seconds
    max_size=1000               # Maximum cache entries
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `similarity_threshold` | float | 0.95 | Minimum cosine similarity for cache hit |
| `ttl_seconds` | int | 3600 | Time-to-live for cache entries |
| `max_size` | int | 1000 | Maximum number of entries (LRU eviction) |

### Methods

```python
# Check for similar cached query
result = cache.get_by_similarity(query_embedding)

# Store result with embedding
cache.set_with_embedding(query_embedding, results)

# Get cache statistics
stats = cache.stats()

# Clear all entries
cache.clear()
```

### Example

```python
from langrag import SemanticCache

cache = SemanticCache(
    similarity_threshold=0.95,
    ttl_seconds=3600,
    max_size=1000
)

# Generate query embedding
query_embedding = embedder.embed(["What is Python?"])[0]

# Check cache
cached = cache.get_by_similarity(query_embedding)
if cached:
    results = cached
else:
    # Execute search
    results = workflow.search("What is Python?")
    # Cache results
    cache.set_with_embedding(query_embedding, results)

# View stats
print(cache.stats())
# {"size": 1, "hits": 0, "misses": 1}
```

## BaseCache

Abstract base class for cache implementations.

```python
from langrag import BaseCache

class CustomCache(BaseCache):
    def get(self, key: str):
        pass

    def set(self, key: str, value, ttl: int = None):
        pass

    def delete(self, key: str):
        pass
```

## CacheEntry

Data class for cache entries.

```python
from langrag import CacheEntry

entry = CacheEntry(
    key="query-hash",
    value=search_results,
    embedding=[0.1, 0.2, ...],
    created_at=datetime.now(),
    ttl_seconds=3600
)
```
