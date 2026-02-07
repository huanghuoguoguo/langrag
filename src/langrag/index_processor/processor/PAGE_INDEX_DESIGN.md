# PageIndex Strategy Implementation

This document describes the implementation of the **PageIndex** (Tree Index) retrieval strategy in LangRAG. This strategy is designed for complex, structured documents where traditional vector search struggles due to loss of context.

## Overview

PageIndex implements a "Vectorless RAG" approach inspired by the RAPTOR paper and similar tree-structured retrieval systems. Instead of treating documents as flat chunks, it:

1. **Preserves Document Structure**: Parses documents into a hierarchical tree based on headers (H1 → H2 → H3 → Content).
2. **Generates Summaries**: Uses LLM to create summaries for each section, aggregating information from children.
3. **Enables Agentic Navigation**: At retrieval time, an LLM agent navigates the tree top-down, deciding which branches to explore.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PageIndex Flow                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Document]                                                     │
│      │                                                          │
│      ▼                                                          │
│  ┌──────────────────────┐                                       │
│  │ MarkdownStructureParser │  ◄─── Parses headers into tree    │
│  └──────────────────────┘                                       │
│      │                                                          │
│      ▼                                                          │
│  ┌──────────┐                                                   │
│  │ TreeNode │ ─── Root (Level 0)                                │
│  └──────────┘                                                   │
│       │                                                         │
│       ├── H1 Node (Level 1) ─── Summary from LLM               │
│       │    │                                                    │
│       │    ├── H2 Node (Level 2) ─── Summary from LLM          │
│       │    │    │                                               │
│       │    │    └── Content Leaf ─── Original text             │
│       │    │                                                    │
│       │    └── Content Leaf                                     │
│       │                                                         │
│       └── H1 Node (Level 1)                                     │
│            └── ...                                              │
│                                                                 │
│      │                                                          │
│      ▼                                                          │
│  ┌──────────────────────┐                                       │
│  │ PageIndexProcessor   │  ◄─── Generates summaries, embeds    │
│  └──────────────────────┘                                       │
│      │                                                          │
│      ▼                                                          │
│  [Vector Store]  ◄─── Stores nodes with parent/child metadata   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Retrieval Flow                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [User Query]                                                   │
│      │                                                          │
│      ▼                                                          │
│  ┌──────────────────────┐                                       │
│  │ TreeAgentRetriever   │                                       │
│  └──────────────────────┘                                       │
│      │                                                          │
│      │  1. Search for high-level summaries                      │
│      ▼                                                          │
│  ┌──────────────────────┐                                       │
│  │ [Summary Candidates] │  ◄─── e.g., "Chapter 1 Summary"       │
│  └──────────────────────┘                                       │
│      │                                                          │
│      │  2. LLM Relevance Check: "Is this relevant?"             │
│      ▼                                                          │
│  ┌──────────────────────┐                                       │
│  │ Expand Children      │  ◄─── Fetch children_ids from VDB    │
│  └──────────────────────┘                                       │
│      │                                                          │
│      │  3. Repeat until leaf nodes reached                      │
│      ▼                                                          │
│  [Relevant Leaf Nodes]  ◄─── Final answer chunks                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. PageIndexProcessor (`src/langrag/index_processor/processor/page_index.py`)

**Purpose**: ETL for building the tree index.

**Key Classes**:
- `PageIndexConfig`: Configuration (prompts, header patterns, etc.)
- `TreeNode`: Internal tree representation (id, content, level, children, parent, summary)
- `MarkdownStructureParser`: Parses markdown into TreeNode hierarchy
- `PageIndexProcessor`: Main processor that orchestrates parsing, summarization, and storage

**Process Flow**:
1. Parse document using `MarkdownStructureParser`
2. Recursively summarize nodes using `_summarize_node()` (async LLM calls)
3. Flatten tree to list of `Document` objects with metadata
4. Embed all texts
5. Store in vector database

**Metadata Stored Per Node**:
```python
{
    "dataset_id": str,
    "document_id": str,
    "node_id": str,          # Unique ID for this node
    "parent_id": str | None,  # Parent node ID (for navigation)
    "children_ids": list,     # List of child node IDs
    "level": int,             # 0=Root, 1=H1, 2=H2, ...
    "is_leaf": bool,          # True if no children
    "is_summary": bool,       # True if this is a summary node
    "title": str,             # Section title
    "summary": str,           # LLM-generated summary
}
```

### 2. TreeAgentRetriever (`src/langrag/retrieval/search/tree_agent.py`)

**Purpose**: Agentic retrieval over the tree structure.

**Key Methods**:
- `retrieve(query, top_k)`: Main entry point for retrieval
- `_check_relevance(query, doc)`: LLM call to check if a section is relevant

**Algorithm**:
```python
1. Embed query using LLM
2. Search vector store for initial candidates (prioritize summaries)
3. Initialize queue with candidate nodes
4. While queue not empty and steps < max_steps:
    a. Pop node from queue
    b. Ask LLM: "Is this section relevant to the query?"
    c. If relevant AND is_leaf: add to results
    d. If relevant AND NOT is_leaf: expand children, add to queue
5. Return collected leaf nodes
```

### 3. Pipeline Integration (`src/langrag/pipeline/ingestion.py`)

The `IngestionPipeline` now supports `indexing_technique='page_index'`:

```python
pipeline = IngestionPipeline(
    vector_store=vector_store,
    llm=llm,
    embedder=embedder,
)

pipeline.run(
    file_path='report.md',
    indexing_technique='page_index'  # Uses PageIndexProcessor
)
```

## Usage

### Basic Usage

```python
from langrag.index_processor.processor.page_index import PageIndexProcessor, PageIndexConfig
from langrag.retrieval.search.tree_agent import TreeAgentRetriever

# Indexing
config = PageIndexConfig(
    min_content_length_for_summary=500,
    auto_detect_language=True,
)
processor = PageIndexProcessor(
    llm=your_llm,
    embedder=your_embedder,
    vector_manager=your_vector_store,
    config=config,
)
processor.process(dataset, documents)

# Retrieval
retriever = TreeAgentRetriever(
    llm=your_llm,
    vector_store=your_vector_store,
    max_steps=3,
)
results = await retriever.retrieve("What are the key risk factors?", top_k=4)
```

### Configuration Options

```python
PageIndexConfig(
    # Prompt used for summarization (supports {content} placeholder)
    summarize_prompt="...",
    summarize_prompt_zh="...",  # Chinese version
    
    # Auto-detect language and use appropriate prompt
    auto_detect_language=True,
    
    # Only LLM-summarize if content length exceeds this
    min_content_length_for_summary=500,
    
    # Max concurrent LLM calls
    max_concurrency=5,
    
    # Header regex patterns
    header_patterns=[
        (r"^#\s+(.*)", 1),   # H1
        (r"^##\s+(.*)", 2),  # H2
        ...
    ],
)
```

## Database Requirements

For optimal performance with tree expansion, the vector store should ideally support:

1. **Metadata Filtering**: Filter by `is_summary=True` during initial search
2. **Get by IDs**: `get_by_ids(ids: list[str])` method for fetching children

If `get_by_ids` is not supported, the agent will fall back to returning summary nodes directly without drilling down.

## Trade-offs

### Advantages
- **High Accuracy**: Preserves document structure, enabling precise navigation
- **Explainable**: Can trace the retrieval path (which sections were visited)
- **Better for Complex Questions**: Handles "global" questions that span multiple sections

### Disadvantages
- **Higher Latency**: Multiple LLM calls during both indexing and retrieval
- **Storage Overhead**: Summary nodes add to storage, though typically manageable
- **Document Format Dependency**: Works best with well-structured markdown documents

## Future Improvements

1. **Parallel LLM Calls**: Use semaphore-limited concurrency during summarization
2. **Caching**: Cache relevance checks for repeated queries
3. **Hybrid Mode**: Combine with standard vector search for speed
4. **Better Fallback**: If no tree structure detected, fall back to paragraph chunking
