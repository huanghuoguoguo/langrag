<p align="center">
  <img src="docs/logo.png" alt="LangRAG Logo" width="200"/>
</p>

<h1 align="center">LangRAG</h1>

<p align="center">
  <strong>A Modular, Production-Ready RAG Kernel for Building Intelligent Knowledge Systems</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#roadmap">Roadmap</a> •
  <a href="#comparison">Comparison</a>
</p>

---

## What is LangRAG?

**LangRAG** is a **RAG (Retrieval-Augmented Generation) Kernel** — a modular, extensible core library designed to power intelligent knowledge systems. Unlike end-to-end RAG applications, LangRAG focuses on providing **high-quality, reusable RAG primitives** that can be integrated into any application.

> 🎯 **Core Philosophy**: LangRAG is NOT an application. It's a **kernel** that your application drives.

The `web/` directory contains a **demo application** that showcases how to integrate LangRAG into a real-world system.

---

## Features

### ✅ Implemented (v0.2)

| Category | Feature | Description |
|----------|---------|-------------|
| **Indexing** | Multi-Format Parsing | PDF, DOCX, Markdown, HTML, TXT |
| | Smart Chunking | Recursive Character Splitter with overlap |
| | Parent-Child Indexing | Hierarchical retrieval for long documents |
| | QA Indexing | Question-Answer pair extraction for precise matching |
| | Batch Processing | Efficient large-scale document indexing with progress tracking |
| **Storage** | Vector Stores | DuckDB (persistent, hybrid search), ChromaDB, SeekDB (hybrid) |
| | KV Store | SQLite-based persistent key-value storage |
| | Web Search | Real-time web integration (Bing, Google, DuckDuckGo) |
| **Retrieval** | Hybrid Search | Vector + BM25 Full-text with RRF fusion (DuckDB, SeekDB) |
| | Agentic Router | LLM-powered knowledge base selection |
| | Query Rewriter | Semantic query optimization |
| | Reranker | Cohere, Qwen, NoOp providers |
| | Semantic Cache | Similarity-based query caching with TTL and LRU eviction |
| **Evaluation** | LLM Judge | Faithfulness, Answer Relevancy, Context Relevancy metrics |
| | Batch Evaluation | Evaluate multiple samples with progress callbacks |
| | Evaluation Report | Aggregated statistics and per-sample results |
| **Observability** | OpenTelemetry | Distributed tracing for retrieval and indexing pipelines |
| **Generation** | Streaming | Server-Sent Events for real-time responses |
| | LLM Abstraction | OpenAI-compatible interface with injection |
| **Testing** | Full Suite | Unit, Integration, E2E, Smoke tests (357 tests) |

### 🔧 Architecture Highlights

- **Dependency Injection**: LLM, Embedder, and VectorStore are injected, not managed internally.
- **Factory Pattern**: Easily register and create custom components.
- **Async-First**: Core APIs support async/await for high concurrency.
- **Type-Safe**: Pydantic models for all configurations and entities.
- **Observable**: Built-in OpenTelemetry tracing support.

---

## Architecture

```
src/langrag/
├── config/            # Configuration management (Pydantic)
├── core/              # Callbacks and event system
├── datasource/        # Storage abstractions
│   ├── kv/            # Key-Value stores (InMemory, SQLite)
│   └── vdb/           # Vector databases (Chroma, DuckDB, SeekDB, Web)
├── entities/          # Domain models (Document, Dataset, SearchResult)
├── index_processor/   # Indexing pipeline
│   ├── extractor/     # Document parsers (PDF, DOCX, MD, HTML, TXT)
│   ├── splitter/      # Text chunkers (Recursive, FixedSize)
│   ├── processor/     # Index strategies (Paragraph, ParentChild, QA)
│   └── cleaner/       # Text normalization
├── llm/               # LLM abstractions
│   └── embedder/      # Embedding providers
├── retrieval/         # Retrieval pipeline
│   ├── router/        # Knowledge base routing
│   ├── rewriter/      # Query rewriting
│   ├── rerank/        # Result reranking
│   ├── compressor/    # Context compression
│   └── workflow.py    # Orchestration
├── cache/             # Semantic caching layer
├── batch/             # Batch processing for large-scale indexing
├── evaluation/        # LLM Judge evaluation framework
│   └── metrics/       # Faithfulness, Answer/Context Relevancy
├── observability/     # OpenTelemetry tracing integration
└── utils/             # Utilities (RRF, similarity, async helpers)
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/huanghuoguoguo/langrag.git
cd langrag

# Install with uv (recommended)
uv sync --dev
```

### Option 1: Run the Web Demo

```bash
./web/start.sh
# Or: uv run python -m web.app
```

Visit: [http://localhost:8000](http://localhost:8000)

### Option 2: Use as a Library

```python
from langrag import (
    Dataset,
    SimpleTextParser,
    RecursiveCharacterChunker,
    ParentChildIndexProcessor
)
from langrag.datasource.vdb.duckdb import DuckDBVector
from langrag.datasource.kv.sqlite import SQLiteKV

# 1. Parse documents
parser = SimpleTextParser()
docs = parser.parse("knowledge_base.txt")

# 2. Create dataset and stores
dataset = Dataset(name="my_kb", collection_name="my_collection")
vector_store = DuckDBVector(dataset, database_path="./vectors.duckdb")
kv_store = SQLiteKV(db_path="./parents.sqlite")

# 3. Index with Parent-Child strategy
processor = ParentChildIndexProcessor(
    vector_store=vector_store,
    kv_store=kv_store,
    embedder=my_embedder,  # Inject your embedder
    parent_splitter=...,
    child_splitter=...
)
processor.process(dataset, docs)

# 4. Search
results = vector_store.search("your query", query_vector=[...], top_k=5)
```

---

## Documentation

LangRAG uses MkDocs with Material theme for comprehensive documentation.

### View Documentation

```bash
# Install documentation dependencies
uv sync --extra docs

# Serve documentation locally
uv run mkdocs serve

# Build static documentation
uv run mkdocs build
```

Visit: [http://localhost:8000](http://localhost:8000) for local docs.

### Documentation Structure

- **Getting Started**: Installation, Quick Start
- **User Guide**: Core Concepts, Document Processing, Retrieval Workflow, Evaluation
- **API Reference**: Complete API documentation for all modules

---

## Roadmap

### ✅ v0.2 (Completed)

- [x] **DuckDB FTS**: Full-text search with BM25 and RRF hybrid fusion
- [x] **Semantic Cache**: Similarity-based caching with TTL and LRU eviction
- [x] **Batch Processing**: Large-scale document indexing with progress tracking
- [x] **LLM Judge**: Evaluation framework (Faithfulness, Answer/Context Relevancy)
- [x] **OpenTelemetry**: Distributed tracing integration
- [x] **API Documentation**: MkDocs-based comprehensive documentation

### 🚀 v0.3 (Planned)
- [ ] **Graph RAG**: Knowledge graph integration
- [ ] **Adaptive Retrieval**: Dynamic strategy selection based on query type
- [ ] **Multi-Tenant**: Full tenant isolation with namespace support
- [ ] **Evaluation Benchmark**: Built-in eval datasets (BEIR, MTEB)

### Future
- [ ] **Docker**: Official Docker image and Compose file
- [ ] **Multi-Modal**: Image and audio document support
- [ ] **Agents**: Tool-use and multi-step reasoning
- [ ] **Cloud Connectors**: S3, GCS, Azure Blob for document ingestion

---

## Comparison with Other RAG Frameworks

| Feature | LangRAG | LangChain | LlamaIndex | Haystack |
|---------|---------|-----------|------------|----------|
| **Focus** | RAG Kernel | General LLM Framework | Data Framework | Production Pipelines |
| **Philosophy** | Inject, Don't Manage | All-in-one | Index-centric | Component-based |
| **Parent-Child Indexing** | ✅ Built-in | ❌ Manual | ✅ Supported | ❌ Manual |
| **QA Indexing** | ✅ Built-in | ❌ N/A | ❌ N/A | ❌ N/A |
| **Agentic Router** | ✅ LLM-powered | ✅ Chains | ✅ Router | ✅ Pipelines |
| **Hybrid Search** | ✅ DuckDB, SeekDB | ❌ External | ✅ External | ✅ External |
| **Semantic Cache** | ✅ Built-in | ❌ External | ❌ External | ❌ External |
| **LLM Judge Evaluation** | ✅ Built-in | ❌ External | ❌ External | ❌ External |
| **OpenTelemetry** | ✅ Native | ⚠️ Partial | ⚠️ Partial | ✅ Native |
| **Streaming** | ✅ Native SSE | ✅ Callbacks | ✅ Streaming | ✅ Streaming |
| **Web Search Integration** | ✅ Multi-provider | ✅ Tools | ✅ Tools | ✅ Nodes |
| **Lightweight** | ✅ ~3k LOC core | ❌ Large | ❌ Large | ⚠️ Medium |
| **Type Safety** | ✅ Pydantic | ⚠️ Partial | ✅ Pydantic | ✅ Pydantic |

### Why Choose LangRAG?

1. **Kernel, Not Framework**: LangRAG gives you RAG primitives without imposing an application structure.
2. **Injection-First**: Your app owns the LLM, Embedder, and storage. LangRAG just orchestrates.
3. **Advanced Indexing**: Built-in Parent-Child and QA indexing strategies out of the box.
4. **Built-in Evaluation**: LLM Judge framework for retrieval quality assessment.
5. **Production-Ready**: Semantic caching, batch processing, and OpenTelemetry tracing.
6. **Comprehensive Testing**: 357 tests with thorough edge case coverage.
7. **Minimal Dependencies**: Core library has minimal external dependencies.

---

## Development

```bash
# Run all tests
uv run pytest tests/

# Run smoke tests (fast sanity check)
uv run pytest tests/smoke/

# Run with coverage
./run_tests.sh

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

### Optional Dependencies

```bash
# Document parsers (PDF, DOCX, etc.)
pip install langrag[parsers]

# Reranker support
pip install langrag[reranker]

# OpenTelemetry observability
pip install langrag[observability]

# Documentation generation
pip install langrag[docs]

# All features
pip install langrag[all]
```

---

## License

MIT License

---

<p align="center">
  <sub>Built with ❤️ for the RAG community</sub>
</p>
