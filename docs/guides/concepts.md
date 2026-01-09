# Core Concepts

This guide explains the key concepts and architecture of LangRAG.

## Architecture Overview

LangRAG follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                       │
│  (Web API, CLI, Custom Applications)                        │
├─────────────────────────────────────────────────────────────┤
│                      LangRAG Core                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Retrieval  │  │   Index     │  │    Evaluation       │ │
│  │  Workflow   │  │  Processor  │  │    Framework        │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Cache     │  │    Batch    │  │   Observability     │ │
│  │   Layer     │  │  Processor  │  │   (Tracing)         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Data Source Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   DuckDB    │  │   SeekDB    │  │    ChromaDB         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### Documents

The `Document` class is the fundamental data unit in LangRAG:

```python
from langrag import Document, DocumentType

doc = Document(
    id="unique-id",
    page_content="The actual text content",
    metadata={"source": "file.pdf", "page": 1},
    doc_type=DocumentType.TEXT,
    vector=[0.1, 0.2, ...]  # Optional: embedding vector
)
```

**Document Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique identifier |
| `page_content` | str | The text content |
| `metadata` | dict | Arbitrary metadata |
| `doc_type` | DocumentType | Type of document |
| `vector` | list[float] | Embedding vector (optional) |

### Index Processors

Index processors handle the document ingestion pipeline:

```
Raw Document → Parser → Chunker → Embedder → Indexed Chunks
```

**Available Processors:**

1. **ParagraphIndexProcessor**: Standard paragraph-based chunking
2. **ParentChildIndexProcessor**: Hierarchical chunks with parent context
3. **QAIndexProcessor**: Optimized for question-answering use cases

### Retrieval Workflow

The `RetrievalWorkflow` orchestrates the search process:

```python
workflow = RetrievalWorkflow(
    vector_store=store,
    embedder=embedder,
    reranker=reranker,  # Optional
    rewriter=rewriter,  # Optional
)
```

**Search Types:**

- **Vector Search**: Pure semantic similarity
- **Keyword Search**: Full-text search (FTS)
- **Hybrid Search**: Combines vector + keyword with RRF fusion

### Vector Stores

Vector stores manage document storage and retrieval:

```python
from langrag.datasource.vdb.duckdb import DuckDBVector

store = DuckDBVector(
    collection_name="my_collection",
    dimension=1536,
    persist_directory="./data"
)
```

**Supported Backends:**

| Backend | Features | Best For |
|---------|----------|----------|
| DuckDB | Embedded, FTS, Hybrid | Development, small datasets |
| SeekDB | High performance | Production, large datasets |
| ChromaDB | Popular, easy to use | General purpose |

### Semantic Cache

The cache layer reduces redundant computations:

```python
from langrag import SemanticCache

cache = SemanticCache(
    similarity_threshold=0.95,  # Minimum similarity for cache hit
    ttl_seconds=3600,           # Time-to-live
    max_size=1000               # Maximum cache entries
)
```

**How it works:**

1. Query embedding is computed
2. Cache checks for similar previous queries
3. If similarity > threshold, return cached result
4. Otherwise, execute search and cache result

### Batch Processing

For large-scale document processing:

```python
from langrag import BatchProcessor, BatchConfig

config = BatchConfig(
    embedding_batch_size=100,  # Documents per embedding batch
    storage_batch_size=500,    # Documents per storage batch
    max_retries=3,             # Retry on failure
    continue_on_error=True     # Don't stop on individual errors
)

processor = BatchProcessor(embedder, vector_store, config)
stats = processor.process_documents(documents)
```

### Evaluation Framework

Assess RAG quality with LLM-as-a-Judge:

```python
from langrag import (
    EvaluationSample,
    FaithfulnessEvaluator,
    AnswerRelevancyEvaluator,
    ContextRelevancyEvaluator,
    EvaluationRunner,
)

evaluators = [
    FaithfulnessEvaluator(llm),
    AnswerRelevancyEvaluator(llm),
    ContextRelevancyEvaluator(llm),
]

runner = EvaluationRunner(evaluators)
report = runner.run(samples)
```

**Metrics:**

| Metric | Measures | Range |
|--------|----------|-------|
| Faithfulness | Answer grounded in context | 0.0 - 1.0 |
| Answer Relevancy | Answer addresses question | 0.0 - 1.0 |
| Context Relevancy | Retrieved context is relevant | 0.0 - 1.0 |

## Design Principles

### 1. Modularity

Every component can be replaced or extended:

```python
# Use any embedder that implements BaseEmbedder
class CustomEmbedder(BaseEmbedder):
    def embed(self, texts: list[str]) -> list[list[float]]:
        # Your implementation
        pass
```

### 2. Dependency Injection

Components are injected, not hardcoded:

```python
# The workflow doesn't care which vector store you use
workflow = RetrievalWorkflow(
    vector_store=any_compatible_store,
    embedder=any_compatible_embedder,
)
```

### 3. Fail-Fast

Errors are raised immediately, not hidden:

```python
# No silent failures - errors propagate clearly
try:
    results = workflow.search(query)
except EmbeddingError as e:
    # Handle embedding failure
    pass
```

### 4. Observability

Built-in support for monitoring and tracing:

```python
# OpenTelemetry integration
from langrag.observability import configure_tracing
configure_tracing(service_name="my-rag-app")
```

## Next Steps

- [Document Processing Guide](document-processing.md)
- [Retrieval Guide](retrieval.md)
- [Evaluation Guide](evaluation.md)
