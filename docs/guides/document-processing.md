# Document Processing

This guide covers document parsing, chunking, and the index processing pipeline.

## Overview

Document processing in LangRAG follows a pipeline:

```
Raw File → Parser → Plain Text → Chunker → Chunks → Embedder → Vectors
```

## Parsers

Parsers convert various file formats to plain text.

### SimpleTextParser

For plain text files:

```python
from langrag import SimpleTextParser

parser = SimpleTextParser()
content = parser.parse("path/to/file.txt")
```

### Built-in Parsers

| Parser | Formats | Dependencies |
|--------|---------|--------------|
| `SimpleTextParser` | `.txt` | None |
| `PDFParser` | `.pdf` | `pypdf` |
| `DocxParser` | `.docx` | `python-docx` |
| `MarkdownParser` | `.md` | `markdown` |
| `HTMLParser` | `.html` | `beautifulsoup4` |

### Custom Parsers

Implement `BaseParser` for custom formats:

```python
from langrag import BaseParser

class CustomParser(BaseParser):
    def parse(self, file_path: str) -> str:
        """Parse file and return plain text."""
        with open(file_path, 'r') as f:
            # Your parsing logic
            return processed_text

    def supported_extensions(self) -> list[str]:
        return [".custom"]
```

## Chunkers

Chunkers split text into smaller pieces for embedding.

### RecursiveCharacterChunker

The most commonly used chunker:

```python
from langrag import RecursiveCharacterChunker

chunker = RecursiveCharacterChunker(
    chunk_size=512,      # Target chunk size in characters
    chunk_overlap=50,    # Overlap between chunks
    separators=["\n\n", "\n", ". ", " "]  # Split priorities
)

chunks = chunker.split_text(long_text)
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 1000 | Maximum characters per chunk |
| `chunk_overlap` | 200 | Characters shared between chunks |
| `separators` | Various | List of separators to try |

### Chunking Strategies

**1. Fixed Size Chunking**
Simple but may break sentences:

```python
chunker = RecursiveCharacterChunker(
    chunk_size=500,
    chunk_overlap=0
)
```

**2. Semantic Chunking**
Respects natural boundaries:

```python
chunker = RecursiveCharacterChunker(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", "! ", "? "]
)
```

**3. Overlapping Chunks**
Better context continuity:

```python
chunker = RecursiveCharacterChunker(
    chunk_size=500,
    chunk_overlap=100  # 20% overlap
)
```

### Custom Chunkers

```python
from langrag import BaseChunker

class SentenceChunker(BaseChunker):
    def split_text(self, text: str) -> list[str]:
        """Split by sentences."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return sentences
```

## Index Processors

Index processors combine parsing, chunking, and embedding.

### ParagraphIndexProcessor

Standard processor for general use:

```python
from langrag import ParagraphIndexProcessor

processor = ParagraphIndexProcessor(
    parser=SimpleTextParser(),
    chunker=RecursiveCharacterChunker(chunk_size=512),
    embedder=my_embedder,
)

# Process a document
chunks = processor.process([document])
```

### ParentChildIndexProcessor

Creates hierarchical chunks with parent context:

```python
from langrag import ParentChildIndexProcessor

processor = ParentChildIndexProcessor(
    parser=SimpleTextParser(),
    parent_chunker=RecursiveCharacterChunker(chunk_size=2000),
    child_chunker=RecursiveCharacterChunker(chunk_size=400),
    embedder=my_embedder,
)

# Each child chunk references its parent
chunks = processor.process([document])
```

**Use Cases:**

- Long documents where context is important
- Technical documentation
- Legal documents

### QAIndexProcessor

Optimized for question-answering:

```python
from langrag import QAIndexProcessor

processor = QAIndexProcessor(
    parser=SimpleTextParser(),
    chunker=RecursiveCharacterChunker(chunk_size=300),
    embedder=my_embedder,
    llm=my_llm,  # For generating potential questions
)

# Generates question-answer pairs
qa_pairs = processor.process([document])
```

## Best Practices

### 1. Choose Chunk Size Carefully

```python
# Too small: loses context
chunker = RecursiveCharacterChunker(chunk_size=100)  # ❌

# Too large: may not fit in context window
chunker = RecursiveCharacterChunker(chunk_size=5000)  # ❌

# Good balance for most use cases
chunker = RecursiveCharacterChunker(chunk_size=512)  # ✓
```

### 2. Consider Document Type

```python
# For code: preserve functions/classes
code_chunker = RecursiveCharacterChunker(
    separators=["\n\nclass ", "\n\ndef ", "\n\n", "\n"]
)

# For prose: preserve paragraphs
prose_chunker = RecursiveCharacterChunker(
    separators=["\n\n", "\n", ". ", " "]
)
```

### 3. Preserve Metadata

```python
from langrag import Document

doc = Document(
    page_content=text,
    metadata={
        "source": "manual.pdf",
        "page": 5,
        "chapter": "Installation"
    }
)

# Metadata is preserved through processing
chunks = processor.process([doc])
for chunk in chunks:
    print(chunk.metadata)  # Contains original metadata
```

### 4. Handle Large Documents

```python
from langrag import BatchProcessor, BatchConfig

# Use batch processing for many documents
config = BatchConfig(
    embedding_batch_size=100,
    continue_on_error=True
)

processor = BatchProcessor(embedder, vector_store, config)
stats = processor.process_documents(all_chunks)
```

## Troubleshooting

### Encoding Issues

```python
# Specify encoding for text files
parser = SimpleTextParser(encoding="utf-8")
```

### Memory Issues with Large Files

```python
# Process in batches
from langrag import BatchConfig

config = BatchConfig(
    embedding_batch_size=50,  # Smaller batches
    storage_batch_size=100
)
```

### Poor Retrieval Quality

Try adjusting:

1. Smaller chunk size for more precise matching
2. Larger overlap for better context
3. Different separators for your content type
