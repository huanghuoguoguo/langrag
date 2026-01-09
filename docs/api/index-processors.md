# Index Processors

Document processing pipelines for parsing, chunking, and embedding documents.

## BaseIndexProcessor

Abstract base class for index processors.

```python
from langrag import BaseIndexProcessor

class CustomProcessor(BaseIndexProcessor):
    def process(self, documents: list[Document]) -> list[Document]:
        # Your processing implementation
        pass
```

## ParagraphIndexProcessor

Standard processor for general-purpose document indexing.

```python
from langrag import (
    ParagraphIndexProcessor,
    SimpleTextParser,
    RecursiveCharacterChunker,
)

processor = ParagraphIndexProcessor(
    parser=SimpleTextParser(),
    chunker=RecursiveCharacterChunker(chunk_size=512),
    embedder=my_embedder,
)

# Process documents
chunks = processor.process(documents)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `parser` | BaseParser | Document parser |
| `chunker` | BaseChunker | Text chunker |
| `embedder` | BaseEmbedder | Embedding model |

## ParentChildIndexProcessor

Creates hierarchical chunks with parent context for better retrieval.

```python
from langrag import ParentChildIndexProcessor

processor = ParentChildIndexProcessor(
    parser=SimpleTextParser(),
    parent_chunker=RecursiveCharacterChunker(chunk_size=2000),
    child_chunker=RecursiveCharacterChunker(chunk_size=400),
    embedder=my_embedder,
)

# Each child chunk references its parent
chunks = processor.process(documents)
```

### Use Cases

- Long documents where context is important
- Technical documentation
- Legal documents

## QAIndexProcessor

Optimized for question-answering systems.

```python
from langrag import QAIndexProcessor

processor = QAIndexProcessor(
    parser=SimpleTextParser(),
    chunker=RecursiveCharacterChunker(chunk_size=300),
    embedder=my_embedder,
    llm=my_llm,  # For generating questions
)

# Generates question-answer pairs
qa_pairs = processor.process(documents)
```
