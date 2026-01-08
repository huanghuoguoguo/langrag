# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-08

### Added

#### Core RAG Framework
- Modular retrieval workflow with configurable pipeline stages
- PostProcessor support for retrieval result post-processing
- Global configuration management via `Config` and `settings.py`

#### Vector Database Support
- **ChromaDB**: Full vector search with metadata filtering
- **DuckDB**: Vector-only search support (FTS not yet implemented)
- **SeekDB**: Hybrid search with both vector and full-text search capabilities

#### Advanced Indexing
- Parent-Child indexing strategy for hierarchical document retrieval
- QA indexing for question-answer pair extraction and retrieval

#### Agentic RAG
- LLM-powered query routing for intelligent knowledge base selection
- Automatic query rewriting for improved retrieval accuracy
- Smart knowledge base selection based on query intent

#### Web Search Integration
- Web search as a knowledge base type
- Multiple search provider support
- Seamless integration with existing RAG workflow

#### Reranking
- Reranker support for result re-ordering
- Configurable reranking strategies

#### Streaming & API
- Server-Sent Events (SSE) streaming support
- RESTful API with FastAPI
- Async-first architecture

#### Web UI
- Knowledge base management interface
- LLM configuration panel
- Interactive chat interface
- Document upload and indexing

#### Document Parsers
- PDF parsing via `pypdf`
- DOCX parsing via `python-docx`
- Markdown parsing
- HTML parsing via `beautifulsoup4`
- Automatic encoding detection via `chardet`

#### Developer Experience
- Comprehensive unit test suite
- pytest with async support
- Code coverage reporting (Cobertura XML)
- Ruff linting and formatting
- MyPy type checking

### Infrastructure
- Python 3.11+ support
- `uv` build system
- Optional dependency groups (`parsers`, `reranker`, `all`, `dev`)

---

[0.1.0]: https://github.com/huanghuoguoguo/langrag/releases/tag/v0.1.0
