# LangRAG Web Demo

> ⚠️ **This is a DEMO application**, not a production-ready product.

This directory contains a web application that demonstrates how to integrate and drive the **LangRAG kernel**. It serves as:

1. **Reference Implementation**: Shows best practices for using LangRAG in a real application.
2. **Interactive Playground**: A visual interface to experiment with RAG features.
3. **Integration Example**: Demonstrates dependency injection, LLM configuration, and multi-KB management.

---

## Purpose

The LangRAG core library (`src/langrag/`) is a **RAG kernel** — it provides primitives but doesn't manage business logic. This demo shows how a real application would:

- Manage knowledge base metadata (names, descriptions, configurations)
- Handle user authentication and sessions (placeholder)
- Configure LLM and Embedder credentials
- Orchestrate document upload and processing
- Provide a chat interface with streaming responses

---

## Architecture

```
web/
├── app.py              # FastAPI application entry point
├── config.py           # Configuration (data paths, settings)
├── core/               # Core integration layer
│   ├── database.py     # SQLite database connection
│   ├── rag_kernel.py   # RAGKernel wrapper (drives langrag)
│   ├── llm_adapter.py  # LLM adapter for injection
│   ├── vdb_manager.py  # Vector store manager
│   └── context.py      # Request context and dependencies
├── models/             # SQLModel entities
│   └── database.py     # KnowledgeBase, Document, LLMConfig models
├── services/           # Business logic layer
│   ├── kb_service.py       # Knowledge base CRUD
│   ├── document_service.py # Document processing
│   └── embedder_service.py # Model configuration
├── routers/            # FastAPI routers (API endpoints)
│   ├── kb.py           # /api/kb/* - Knowledge base management
│   ├── document.py     # /api/document/* - Document upload
│   ├── search.py       # /api/search/* - Retrieval testing
│   ├── chat.py         # /api/chat - RAG chat with streaming
│   └── config.py       # /api/config - LLM/Embedder settings
├── static/             # Frontend (Vanilla JS SPA)
│   ├── index.html      # Main HTML
│   ├── css/            # Stylesheets
│   └── js/             # JavaScript modules
└── data/               # Persistent storage (auto-created)
    ├── app.db          # SQLite business database
    ├── kv_store.sqlite # Parent-Child KV store
    ├── chroma/         # ChromaDB data
    ├── duckdb/         # DuckDB vector files
    └── seekdb/         # SeekDB data
```

---

## How It Drives LangRAG

### 1. Dependency Injection

The demo injects LLM and Embedder into LangRAG components:

```python
# web/core/rag_kernel.py
from langrag.retrieval.router.llm_router import LLMRouter
from langrag.retrieval.rewriter.llm_rewriter import LLMRewriter

class RAGKernel:
    def set_llm(self, base_url, api_key, model, ...):
        # Create adapter that implements BaseLLM
        self.llm_adapter = WebLLMAdapter(base_url, api_key, model)
        
        # Inject into LangRAG components
        self.router = LLMRouter(llm=self.llm_adapter)
        self.rewriter = LLMRewriter(llm=self.llm_adapter)
```

### 2. Vector Store Management

```python
# Create knowledge base with specific VDB type
store = kernel.create_vector_store(
    kb_id="kb_123",
    name="My Knowledge Base",
    vdb_type="seekdb"  # or "duckdb", "chroma", "web_search"
)
```

### 3. Document Processing

```python
# Process document through LangRAG pipeline
chunk_count = kernel.process_document(
    file_path=Path("document.pdf"),
    kb_id="kb_123",
    indexing_technique="parent_child"  # or "high_quality", "qa"
)
```

### 4. RAG Chat with Streaming

```python
# Streaming chat response
async for chunk in kernel.chat(kb_ids=["kb_123"], query="What is...", stream=True):
    # chunk is JSON: {"type": "sources", "data": [...]} or {"type": "content", "data": "..."}
    yield chunk
```

---

## Running the Demo

### Prerequisites

- Python 3.11+
- uv (recommended) or pip

### Quick Start

```bash
# From project root
cd langrag

# Install dependencies
uv sync --dev

# Start the server
./web/start.sh
# Or: uv run python -m web.app
```

### Access

- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs (Swagger)
- **ReDoc**: http://localhost:8000/redoc

---

## Key Features Demonstrated

| Feature | Location | Description |
|---------|----------|-------------|
| Multi-KB Management | `/api/kb/*` | Create, list, delete knowledge bases |
| Document Upload | `/api/document/upload/{kb_id}` | Upload and process documents |
| Indexing Strategies | `rag_kernel.py` | Paragraph, Parent-Child, QA indexing |
| Vector Store Types | `create_vector_store()` | DuckDB, ChromaDB, SeekDB, Web Search |
| Streaming Chat | `/api/chat` | SSE-based streaming responses |
| LLM Configuration | `/api/config/llm` | Dynamic LLM provider setup |
| Embedder Configuration | `/api/config/embedder` | OpenAI or SeekDB embeddings |
| Hybrid Search | SeekDB integration | Vector + Full-text search |
| Web Search KB | `web_search` type | Real-time web integration |

---

## Customization

To build your own application on top of LangRAG:

1. **Copy** `web/core/rag_kernel.py` as a starting point.
2. **Inject** your own LLM and Embedder implementations.
3. **Replace** the SQLite business layer with your database.
4. **Extend** the routers for your specific API needs.

---

## Not Included (Demo Limitations)

This demo intentionally omits:

- ❌ User authentication and authorization
- ❌ Rate limiting and quotas
- ❌ Production-grade error handling
- ❌ Horizontal scaling considerations
- ❌ Monitoring and observability

For production use, you should add these on top of the LangRAG kernel.

---

## API Reference

See the auto-generated OpenAPI docs at `/docs` after starting the server.

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/kb` | List all knowledge bases |
| POST | `/api/kb` | Create knowledge base |
| POST | `/api/document/upload/{kb_id}` | Upload document |
| POST | `/api/search/semantic/{kb_id}` | Search in KB |
| POST | `/api/chat` | RAG chat (supports streaming) |
| POST | `/api/config/llm` | Configure LLM provider |
| POST | `/api/config/embedder` | Configure Embedder |

---

## License

MIT License — Same as LangRAG core.
