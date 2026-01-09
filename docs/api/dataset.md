# Dataset

Dataset and retrieval context classes for managing document collections.

## Dataset

Container for a collection of documents with metadata.

```python
from langrag import Dataset, Document

dataset = Dataset(
    id="knowledge-base-1",
    name="Product Documentation",
    documents=[doc1, doc2, doc3],
    metadata={"version": "1.0"}
)
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | str | Unique identifier |
| `name` | str | Human-readable name |
| `documents` | list[Document] | Collection of documents |
| `metadata` | dict | Additional metadata |

## RetrievalContext

Context information for retrieval operations.

```python
from langrag import RetrievalContext

context = RetrievalContext(
    query="What is Python?",
    top_k=5,
    filters={"topic": "programming"},
    include_metadata=True
)
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `query` | str | The search query |
| `top_k` | int | Number of results to return |
| `filters` | dict | Metadata filters |
| `include_metadata` | bool | Include metadata in results |
