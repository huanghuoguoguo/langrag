# LangRAG

A modular Retrieval-Augmented Generation framework for building production-ready RAG applications.

## Overview

LangRAG provides a complete toolkit for building RAG systems with:

- **Document Processing**: Parse, chunk, and embed documents from multiple formats
- **Vector Storage**: Pluggable vector store backends (DuckDB, SeekDB, ChromaDB)
- **Hybrid Search**: Combine vector similarity with full-text search
- **Semantic Caching**: Reduce redundant computations with embedding-based caching
- **Batch Processing**: Efficiently process large document collections
- **Evaluation**: LLM-as-a-Judge metrics for quality assessment

## Quick Start

```python
from langrag import (
    Document,
    ParagraphIndexProcessor,
    RecursiveCharacterChunker,
    SimpleTextParser,
    RetrievalWorkflow,
)

# Create documents
docs = [Document(page_content="Your document text here...")]

# Process with index pipeline
processor = ParagraphIndexProcessor(
    parser=SimpleTextParser(),
    chunker=RecursiveCharacterChunker(chunk_size=512),
    embedder=your_embedder,
)
chunks = processor.process(docs)

# Store in vector database
vector_store.add_texts(chunks)

# Search
workflow = RetrievalWorkflow(
    vector_store=vector_store,
    embedder=your_embedder,
)
results = workflow.search("your query", top_k=5)
```

## Features

### Document Processing

LangRAG supports multiple document formats:

- Plain text (`.txt`)
- PDF (`.pdf`)
- Word documents (`.docx`)
- Markdown (`.md`)
- HTML (`.html`)

### Index Processing Strategies

Choose from different indexing strategies based on your use case:

| Strategy | Use Case | Description |
|----------|----------|-------------|
| `ParagraphIndexProcessor` | General purpose | Standard chunking by paragraph |
| `ParentChildIndexProcessor` | Detailed retrieval | Hierarchical chunks for context |
| `QAIndexProcessor` | Q&A systems | Optimized for question-answering |

### Vector Store Backends

- **DuckDB**: Lightweight, embedded database with FTS support
- **SeekDB**: High-performance vector database
- **ChromaDB**: Popular open-source vector database

### Evaluation Metrics

Built-in LLM-as-a-Judge evaluation:

- **Faithfulness**: Is the answer grounded in context?
- **Answer Relevancy**: Does the answer address the question?
- **Context Relevancy**: Is the retrieved context relevant?

## Installation

```bash
pip install langrag
```

With optional dependencies:

```bash
# Document parsers
pip install langrag[parsers]

# All features
pip install langrag[all]
```

## Documentation

- [Installation Guide](guides/installation.md)
- [Quick Start Tutorial](guides/quickstart.md)
- [Core Concepts](guides/concepts.md)
- [API Reference](api/index.md)

## License

MIT License - see [LICENSE](https://github.com/huanghuoguoguo/langrag/blob/main/LICENSE) for details.
