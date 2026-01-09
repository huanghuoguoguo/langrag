# Batch Processing

Efficient batch document processing for large-scale indexing.

## BatchProcessor

```python
from langrag import BatchProcessor, BatchConfig

config = BatchConfig(
    embedding_batch_size=100,
    storage_batch_size=500,
    max_retries=3,
    continue_on_error=True
)

processor = BatchProcessor(embedder, vector_store, config)
stats = processor.process_documents(documents)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `embedder` | BaseEmbedder | Embedding model |
| `vector_store` | BaseVector | Target vector store |
| `config` | BatchConfig | Processing configuration |

### Methods

```python
stats = processor.process_documents(
    documents: list[Document],
    on_progress: Callable[[BatchProgress], None] = None
)
```

Returns:
```python
{
    "total": 1000,      # Total documents
    "embedded": 1000,   # Successfully embedded
    "stored": 1000,     # Successfully stored
    "errors": 0,        # Error count
    "duration": 45.2    # Processing time (seconds)
}
```

## BatchConfig

```python
from langrag import BatchConfig

config = BatchConfig(
    embedding_batch_size=100,  # Docs per embedding batch
    storage_batch_size=500,    # Docs per storage batch
    max_retries=3,             # Retry attempts on failure
    retry_delay=1.0,           # Delay between retries (seconds)
    continue_on_error=False,   # Continue on individual failures
    show_progress=True         # Enable progress reporting
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_batch_size` | int | 100 | Documents per embedding API call |
| `storage_batch_size` | int | 500 | Documents per storage operation |
| `max_retries` | int | 3 | Maximum retry attempts |
| `retry_delay` | float | 1.0 | Delay between retries |
| `continue_on_error` | bool | False | Continue on failures |
| `show_progress` | bool | True | Enable progress callbacks |

## BatchProgress

Progress information during batch processing.

```python
def on_progress(progress: BatchProgress):
    print(f"Stage: {progress.stage}")
    print(f"Progress: {progress.percent:.0%}")
    print(f"Message: {progress.message}")

processor.process_documents(docs, on_progress=on_progress)
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `stage` | BatchStage | Current processing stage |
| `current` | int | Current item count |
| `total` | int | Total item count |
| `percent` | float | Progress percentage (0.0-1.0) |
| `message` | str | Status message |

### Example

```python
from langrag import BatchProcessor, BatchConfig, Document

# Create documents
documents = [
    Document(id=f"doc{i}", page_content=f"Content {i}")
    for i in range(10000)
]

# Configure batch processing
config = BatchConfig(
    embedding_batch_size=100,
    storage_batch_size=500,
    continue_on_error=True
)

# Process with progress tracking
def on_progress(p):
    print(f"[{p.stage.value}] {p.percent:.0%} - {p.message}")

processor = BatchProcessor(embedder, vector_store, config)
stats = processor.process_documents(documents, on_progress=on_progress)

print(f"Processed {stats['stored']}/{stats['total']} documents")
print(f"Errors: {stats['errors']}")
print(f"Duration: {stats['duration']:.1f}s")
```
