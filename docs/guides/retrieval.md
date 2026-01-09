# Retrieval Workflow

This guide covers the retrieval system, search types, and optimization strategies.

## Overview

The `RetrievalWorkflow` orchestrates the search process:

```
Query → [Rewrite] → Embed → Search → [Rerank] → Results
```

## Basic Usage

```python
from langrag import RetrievalWorkflow

workflow = RetrievalWorkflow(
    vector_store=vector_store,
    embedder=embedder,
)

results = workflow.search(
    query="What is machine learning?",
    top_k=5
)
```

## Search Types

### Vector Search

Pure semantic similarity search:

```python
results = workflow.search(
    query="machine learning",
    search_type="vector",
    top_k=10
)
```

**Pros:**

- Understands semantic meaning
- Handles synonyms and paraphrasing
- Works across languages (with multilingual embeddings)

**Cons:**

- May miss exact keyword matches
- Requires embeddings for all documents

### Keyword Search (FTS)

Full-text search using BM25:

```python
results = workflow.search(
    query="machine learning",
    search_type="keyword",
    top_k=10
)
```

**Pros:**

- Exact keyword matching
- Fast on large datasets
- No embedding required

**Cons:**

- Misses semantic similarity
- Sensitive to exact wording

### Hybrid Search

Combines vector and keyword search with RRF fusion:

```python
results = workflow.search(
    query="machine learning",
    search_type="hybrid",
    top_k=10
)
```

**How RRF Fusion Works:**

```
score = Σ 1 / (k + rank_i)
```

Where `k` is a constant (default 60) and `rank_i` is the rank in each result list.

**Pros:**

- Best of both worlds
- More robust results
- Better coverage

**Configuration:**

```python
results = workflow.search(
    query="ML algorithms",
    search_type="hybrid",
    vector_weight=0.7,  # Weight for vector results
    keyword_weight=0.3,  # Weight for keyword results
    top_k=10
)
```

## Query Enhancement

### Query Rewriting

Improve queries automatically:

```python
from langrag.retrieval.rewriter import QueryRewriter

rewriter = QueryRewriter(llm=my_llm)

workflow = RetrievalWorkflow(
    vector_store=vector_store,
    embedder=embedder,
    rewriter=rewriter,
)

# Query is automatically rewritten
results = workflow.search("ML")
# Internal: "ML" → "machine learning algorithms and applications"
```

### Query Expansion

Generate multiple query variants:

```python
class MultiQueryRewriter:
    def rewrite(self, query: str) -> list[str]:
        return [
            query,
            f"{query} definition",
            f"{query} examples",
        ]
```

## Reranking

Improve result quality with a second-stage ranker:

```python
from langrag.retrieval.reranker import Reranker

reranker = Reranker(
    api_url="https://api.example.com/rerank",
    api_key="your-key"
)

workflow = RetrievalWorkflow(
    vector_store=vector_store,
    embedder=embedder,
    reranker=reranker,
)

# Results are reranked for better relevance
results = workflow.search(query, top_k=5, rerank_top_k=20)
```

**Reranking Flow:**

1. Initial search returns `rerank_top_k` results (e.g., 20)
2. Reranker rescores all results
3. Top `top_k` results returned (e.g., 5)

## Caching

Enable semantic caching to avoid redundant searches:

```python
from langrag import SemanticCache

cache = SemanticCache(
    similarity_threshold=0.95,
    ttl_seconds=3600,
    max_size=1000
)

workflow = RetrievalWorkflow(
    vector_store=vector_store,
    embedder=embedder,
    cache=cache,
)

# First call: executes search
results1 = workflow.search("What is Python?")

# Similar query: returns cached result
results2 = workflow.search("What's Python?")  # Cache hit!
```

## Filtering

Filter results by metadata:

```python
results = workflow.search(
    query="installation guide",
    top_k=10,
    filter={
        "doc_type": "tutorial",
        "language": "english"
    }
)
```

## Retrieval Service

The `RetrievalService` provides a higher-level interface:

```python
from langrag import RetrievalService

service = RetrievalService(
    embedder=embedder,
    reranker=reranker,
    rewriter=rewriter,
    cache=cache,
)

# Search with all enhancements
results = service.search(
    vector_store=vector_store,
    query="machine learning",
    top_k=5,
    use_cache=True,
    use_rewriter=True,
    use_reranker=True,
)
```

## Performance Optimization

### 1. Use Appropriate Index

```python
# DuckDB with HNSW index
from langrag.datasource.vdb.duckdb import DuckDBVector

store = DuckDBVector(
    collection_name="my_kb",
    dimension=1536,
    index_type="hnsw"  # Faster approximate search
)
```

### 2. Limit Search Scope

```python
# Search specific partition
results = workflow.search(
    query="installation",
    filter={"category": "docs"}
)
```

### 3. Enable Caching

```python
cache = SemanticCache(
    similarity_threshold=0.95,
    max_size=5000
)
```

### 4. Batch Queries

```python
# Process multiple queries efficiently
queries = ["query1", "query2", "query3"]
results = [workflow.search(q) for q in queries]
```

## Troubleshooting

### Poor Relevance

1. Check embedding quality
2. Try hybrid search
3. Add reranking
4. Adjust chunk size

### Slow Search

1. Use approximate indexes (HNSW)
2. Enable caching
3. Reduce `top_k`
4. Add metadata filters

### Missing Results

1. Check if documents are indexed
2. Try keyword search for exact matches
3. Lower similarity threshold
4. Expand query

## Example: Production Setup

```python
from langrag import (
    RetrievalWorkflow,
    SemanticCache,
)
from langrag.datasource.vdb.duckdb import DuckDBVector

# Setup components
vector_store = DuckDBVector(
    collection_name="production_kb",
    dimension=1536,
    persist_directory="/data/vectors"
)

cache = SemanticCache(
    similarity_threshold=0.95,
    ttl_seconds=3600,
    max_size=10000
)

workflow = RetrievalWorkflow(
    vector_store=vector_store,
    embedder=production_embedder,
    reranker=production_reranker,
    cache=cache,
)

# Production search
def search(query: str, top_k: int = 5):
    return workflow.search(
        query=query,
        search_type="hybrid",
        top_k=top_k,
        rerank_top_k=top_k * 4
    )
```
