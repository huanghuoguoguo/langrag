# Quick Start

This guide walks you through building your first RAG application with LangRAG.

## Prerequisites

- LangRAG installed (`pip install langrag`)
- An embedding model (OpenAI API key or local model)

## Step 1: Set Up Your Embedder

LangRAG uses embedders to convert text into vectors. You can use any embedder that implements the `BaseEmbedder` interface.

```python
from langrag import BaseEmbedder

# Example: Create a simple embedder wrapper
class MyEmbedder(BaseEmbedder):
    def __init__(self, client, model):
        self.client = client
        self.model = model
        self._dimension = 1536

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Call your embedding API here
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [r.embedding for r in response.data]

    @property
    def dimension(self) -> int:
        return self._dimension
```

## Step 2: Create Documents

LangRAG uses a `Document` class to represent your content:

```python
from langrag import Document

# Create documents from your content
documents = [
    Document(
        id="doc1",
        page_content="Python is a high-level programming language...",
        metadata={"source": "wikipedia", "topic": "programming"}
    ),
    Document(
        id="doc2",
        page_content="Machine learning is a subset of AI...",
        metadata={"source": "textbook", "topic": "ml"}
    ),
]
```

## Step 3: Process and Index Documents

Use an index processor to chunk and embed your documents:

```python
from langrag import (
    ParagraphIndexProcessor,
    RecursiveCharacterChunker,
    SimpleTextParser,
)

# Create the index processor
processor = ParagraphIndexProcessor(
    parser=SimpleTextParser(),
    chunker=RecursiveCharacterChunker(
        chunk_size=512,
        chunk_overlap=50
    ),
    embedder=my_embedder,  # Your embedder from Step 1
)

# Process documents
chunks = processor.process(documents)
print(f"Created {len(chunks)} chunks")
```

## Step 4: Store in Vector Database

Store the processed chunks in a vector store:

```python
from langrag.datasource.vdb.duckdb import DuckDBVector

# Create vector store
vector_store = DuckDBVector(
    collection_name="my_knowledge_base",
    dimension=1536,  # Must match your embedder dimension
)

# Add documents
vector_store.add_texts(chunks)
```

## Step 5: Search

Create a retrieval workflow and search:

```python
from langrag import RetrievalWorkflow

# Create workflow
workflow = RetrievalWorkflow(
    vector_store=vector_store,
    embedder=my_embedder,
)

# Search
results = workflow.search(
    query="What is Python?",
    top_k=5
)

# Print results
for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Content: {result.content[:200]}...")
    print("---")
```

## Step 6: Evaluate Results (Optional)

Use the evaluation framework to assess quality:

```python
from langrag import (
    EvaluationRunner,
    EvaluationSample,
    FaithfulnessEvaluator,
    AnswerRelevancyEvaluator,
)

# Create evaluation samples
samples = [
    EvaluationSample(
        question="What is Python?",
        answer="Python is a high-level programming language.",
        contexts=[r.content for r in results]
    )
]

# Run evaluation
runner = EvaluationRunner([
    FaithfulnessEvaluator(llm),
    AnswerRelevancyEvaluator(llm),
])
report = runner.run(samples)

print(f"Faithfulness: {report.summary['faithfulness']['mean']:.2f}")
print(f"Relevancy: {report.summary['answer_relevancy']['mean']:.2f}")
```

## Complete Example

Here's the complete code in one script:

```python
from langrag import (
    Document,
    ParagraphIndexProcessor,
    RecursiveCharacterChunker,
    SimpleTextParser,
    RetrievalWorkflow,
)
from langrag.datasource.vdb.duckdb import DuckDBVector

# 1. Setup embedder (implement your own)
embedder = MyEmbedder(client, model)

# 2. Create documents
documents = [
    Document(id="1", page_content="Your content here..."),
]

# 3. Process documents
processor = ParagraphIndexProcessor(
    parser=SimpleTextParser(),
    chunker=RecursiveCharacterChunker(chunk_size=512),
    embedder=embedder,
)
chunks = processor.process(documents)

# 4. Store in vector database
vector_store = DuckDBVector(
    collection_name="my_kb",
    dimension=embedder.dimension,
)
vector_store.add_texts(chunks)

# 5. Search
workflow = RetrievalWorkflow(
    vector_store=vector_store,
    embedder=embedder,
)
results = workflow.search("your query", top_k=5)
```

## Next Steps

- [Core Concepts](concepts.md) - Understand the architecture
- [Document Processing](document-processing.md) - Learn about parsers and chunkers
- [Retrieval Workflow](retrieval.md) - Advanced retrieval options
- [API Reference](../api/index.md) - Detailed API documentation
