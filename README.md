<div align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">中文</a>
</div>

<p align="center">
  <img src="docs/logo.svg" alt="LangRAG Logo" width="300"/>
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

**LangRAG** is a **"Small and Beautiful" RAG Kernel**. It is designed to be the lightweight, robust engine at the heart of your intelligent knowledge systems.

LangRAG strikes a unique balance: it implements **industry-standard best practices** (like Parent-Child Indexing, Hybrid Search, and LLM Judges) while maintaining a **minimal footprint** and a **flat, transparent code structure**.

We believe in an **Out-of-the-Box** philosophy that doesn't sacrifice control. You get production-ready primitives—with built-in Telemetry, Caching, and Evaluation—without the weight and "magic" of monolithic frameworks.

> 🎯 **Why LangRAG?**
> *   **Opinionated & Ready**: Best-practices like RRF and Recursive Chunking are the default, not a config hell.
> *   **Transparent Kernel**: ~3k LOC core with no deep abstraction layers. You can read, understand, and mod the code in minutes.
> *   **Application Driven**: LangRAG is a library you use, not a framework that uses you.

The `web/` directory contains a **demo application** showcasing how a sophisticated, "industry-grade" RAG flow can be built with minimal glue code.

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
| | Multi-Stage LLM | Stage-based model configuration (chat, router, rewriter, reranker) |
| **Testing** | Full Suite | Unit, Integration tests (500+ tests) |

### 🔧 Architecture Highlights

- **Dependency Injection**: LLM, Embedder, and VectorStore are injected, not managed internally.
- **Multi-Stage LLM**: Configure different models for different tasks (chat, router, rewriter, reranker).
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
│   ├── embedder/      # Embedding providers
│   ├── providers/     # LLM providers (OpenAI-compatible, local)
│   └── stages.py      # Multi-stage LLM configuration (chat, router, rewriter, reranker)
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

### 🚀 v0.3 (In Progress)
- [x] **Agents**: Tool-use and multi-step reasoning framework
- [x] **RAPTOR**: Recursive Abstractive Processing for Tree-Organized Retrieval
- [ ] **Graph RAG**: Knowledge graph integration
- [ ] **Adaptive Retrieval**: Dynamic strategy selection based on query type
- [ ] **Evaluation Benchmark**: Built-in eval datasets (BEIR, MTEB)

### Future
- [ ] **Multi-Modal**: Image and audio document support
- [ ] **Cloud Connectors**: S3, GCS, Azure Blob for document ingestion

---

## Comparison with Other RAG Frameworks

| Feature | LangRAG | LangChain | LlamaIndex | PowerRAG |
|---------|---------|-----------|------------|----------|
| **Focus** | RAG Kernel | General LLM Framework | Data Framework | Production Platform |
| **Philosophy** | Inject, Don't Manage | All-in-one | Index-centric | Service-Oriented (DB-centric) |
| **Storage** | Flexible (DuckDB/SeekDB) | Agnostic | Agnostic | OceanBase (SQL+Vector) |
| **Agentic Router** | ✅ LLM-powered | ✅ Chains | ✅ Router | ✅ Conversational |
| **Parent-Child Indexing** | ✅ Built-in | ✅ Supported | ✅ Supported | ✅ Supported |
| **RAA/RAPTOR** | ✅ Built-in | ⚠️ Manual | ✅ Supported | ⚠️ Manual |
| **Hybrid Search** | ✅ DuckDB, SeekDB | ✅ Ensemble | ✅ External | ✅ OceanBase |
| **Semantic Cache** | ✅ Built-in | ❌ External | ❌ External | ❌ External |
| **LLM Judge Evaluation** | ✅ Built-in | ⚠️ Integration | ✅ Built-in | ✅ Integration (Langfuse) |
| **OpenTelemetry** | ✅ Native | ⚠️ Partial | ⚠️ Partial | ⚠️ Integration |
| **Web Search Integration** | ✅ Multi-provider | ✅ Tools | ✅ Tools | ✅ Tools |
| **Lightweight** | ✅ ~3k LOC core | ❌ Large | ❌ Large | ❌ Heavy (Docker Compose) |
| **Type Safety** | ✅ Pydantic | ⚠️ Partial | ✅ Pydantic | ✅ Pydantic |

### Why Choose LangRAG?

1. **Kernel, Not Framework**: LangRAG gives you RAG primitives without imposing an application structure.
2. **Injection-First**: Your app owns the LLM, Embedder, and storage. LangRAG just orchestrates.
3. **Advanced Indexing**: Built-in Parent-Child and QA indexing strategies out of the box.
4. **Built-in Evaluation**: LLM Judge framework for retrieval quality assessment.
5. **Production-Ready**: Semantic caching, batch processing, and OpenTelemetry tracing.
6. **Comprehensive Testing**: 500+ tests with thorough edge case coverage.
7. **Minimal Dependencies**: Core library has minimal external dependencies.

---

## Development

```bash
# Run core tests (LangRAG library)
uv run pytest tests/

# Run web demo tests
uv run pytest web/tests/ -m "not local_llm"

# Run integration tests
uv run pytest tests/integration/

# Run with coverage
./run_tests.sh

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

### Test Structure

```
tests/              # LangRAG core tests
├── unit/           # Unit tests (464 tests)
└── integration/    # Integration tests (DuckDB, SeekDB verification)

web/tests/          # Web Demo tests
├── unit/           # Unit tests for web components
└── test_api.py     # API integration tests
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
