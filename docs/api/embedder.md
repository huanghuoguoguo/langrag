# Base Embedder

Abstract base class for embedding implementations.

## BaseEmbedder

```python
from langrag import BaseEmbedder

class CustomEmbedder(BaseEmbedder):
    def embed(self, texts: list[str]) -> list[list[float]]:
        # Your embedding implementation
        return [[0.1, 0.2, ...] for _ in texts]

    @property
    def dimension(self) -> int:
        return 768
```

### Abstract Methods

| Method | Description |
|--------|-------------|
| `embed(texts)` | Generate embeddings for a list of texts |
| `dimension` | Property returning embedding dimension |

### Example Implementation

```python
from langrag import BaseEmbedder
import httpx

class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self._dimension = 1536

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = httpx.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"input": texts, "model": self.model}
        )
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    @property
    def dimension(self) -> int:
        return self._dimension
```
