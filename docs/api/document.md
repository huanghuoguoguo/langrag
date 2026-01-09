# Document

The `Document` class represents a piece of content in LangRAG.

## Document

```python
from langrag import Document, DocumentType

doc = Document(
    id="unique-id",
    page_content="The actual text content",
    metadata={"source": "file.pdf", "page": 1},
    doc_type=DocumentType.TEXT,
    vector=[0.1, 0.2, ...]  # Optional embedding
)
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | str | Unique identifier for the document |
| `page_content` | str | The text content of the document |
| `metadata` | dict | Arbitrary metadata (source, page number, etc.) |
| `doc_type` | DocumentType | Type of document (TEXT, PDF, etc.) |
| `vector` | list[float] \| None | Optional embedding vector |

### Example

```python
from langrag import Document

# Create a document
doc = Document(
    id="doc-001",
    page_content="Python is a high-level programming language.",
    metadata={
        "source": "intro.txt",
        "topic": "programming"
    }
)

# Access attributes
print(doc.id)           # "doc-001"
print(doc.page_content) # "Python is a..."
print(doc.metadata)     # {"source": "intro.txt", ...}
```

## DocumentType

Enum for document types.

```python
from langrag import DocumentType

DocumentType.TEXT     # Plain text
DocumentType.PDF      # PDF document
DocumentType.DOCX     # Word document
DocumentType.MARKDOWN # Markdown
DocumentType.HTML     # HTML document
```
