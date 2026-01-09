# Embedder Factory

Factory for creating embedder instances.

## EmbedderFactory

```python
from langrag import EmbedderFactory

# Create embedder by type
embedder = EmbedderFactory.create(
    embedder_type="openai",
    api_key="sk-...",
    model="text-embedding-3-small"
)
```

### Factory Methods

#### create()

```python
embedder = EmbedderFactory.create(
    embedder_type: str,
    **kwargs
)
```

### Supported Types

| Type | Description | Required Args |
|------|-------------|---------------|
| `openai` | OpenAI embeddings | `api_key`, `model` |
| `local` | Local model | `model_name` |

### Example

```python
from langrag import EmbedderFactory

# OpenAI embedder
embedder = EmbedderFactory.create(
    embedder_type="openai",
    api_key="sk-your-key",
    model="text-embedding-3-small"
)

# Generate embeddings
vectors = embedder.embed(["Hello world", "How are you?"])
print(f"Dimension: {embedder.dimension}")
```
