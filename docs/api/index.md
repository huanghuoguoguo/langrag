# API Reference

Welcome to the LangRAG API Reference. This documentation is auto-generated from source code docstrings.

## Module Overview

### Core Entities

- [Document](document.md) - Document data model
- [Dataset](dataset.md) - Dataset and retrieval context
- [Search Result](search-result.md) - Search result structure

### Index Processing

- [Parsers](parsers.md) - Document parsers
- [Chunkers](chunkers.md) - Text chunking
- [Index Processors](index-processors.md) - Document processing pipelines

### Retrieval

- [Workflow](workflow.md) - Retrieval workflow
- [Service](service.md) - Retrieval service

### Vector Stores

- [Base Vector](vector-base.md) - Base vector store interface
- [DuckDB](vector-duckdb.md) - DuckDB implementation
- [SeekDB](vector-seekdb.md) - SeekDB implementation

### Embedding

- [Base Embedder](embedder.md) - Embedder interface
- [Factory](embedder-factory.md) - Embedder factory

### Cache

- [Semantic Cache](cache.md) - Query caching

### Batch Processing

- [Batch Processor](batch.md) - Batch document processing

### Evaluation

- [Metrics](evaluation.md) - Evaluation metrics
- [Runner](evaluation-runner.md) - Evaluation runner

## Quick Links

| Component | Import Path |
|-----------|-------------|
| Document | `from langrag import Document` |
| RetrievalWorkflow | `from langrag import RetrievalWorkflow` |
| BatchProcessor | `from langrag import BatchProcessor` |
| SemanticCache | `from langrag import SemanticCache` |
| FaithfulnessEvaluator | `from langrag import FaithfulnessEvaluator` |
