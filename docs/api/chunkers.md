# Chunkers

Text chunking utilities for splitting documents into smaller pieces.

## BaseChunker

Abstract base class for all chunkers.

```python
from langrag import BaseChunker

class CustomChunker(BaseChunker):
    def split(self, documents: list[Document]) -> list[Document]:
        # Your chunking implementation
        pass
```

### Methods

| Method | Description |
|--------|-------------|
| `split(documents)` | Split documents into chunks. Returns list of Document objects. |

## RecursiveCharacterChunker

The most commonly used chunker that splits text recursively using a list of separators.

```python
from langrag import RecursiveCharacterChunker

chunker = RecursiveCharacterChunker(
    chunk_size=512,      # Target chunk size in characters
    chunk_overlap=50,    # Overlap between adjacent chunks
    separators=["\n\n", "\n", ". ", " "]  # Separators to try in order
)

# Split documents
chunks = chunker.split(documents)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunk_size` | int | 1000 | Maximum characters per chunk |
| `chunk_overlap` | int | 200 | Characters to overlap between chunks |
| `separators` | list[str] | See below | List of separators to try |
| `keep_separator` | bool | True | Keep separator in output |

### Default Separators

```python
["\n\n", "\n", " ", ""]
```

### Example

```python
from langrag import Document, RecursiveCharacterChunker

# Create chunker
chunker = RecursiveCharacterChunker(
    chunk_size=500,
    chunk_overlap=50
)

# Create document
doc = Document(
    id="doc1",
    page_content="Your long text here..."
)

# Split into chunks
chunks = chunker.split([doc])

for chunk in chunks:
    print(f"Chunk: {len(chunk.page_content)} chars")
```
