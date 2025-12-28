#!/usr/bin/env python3
"""
LangRAG Demo Application (Phase 2 Architecture)

Demonstrates the indexing and retrieval flow using the newly refactored
modular architecture.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("main")

# Import LangRAG components
try:
    from langrag import (
        Document, 
        DocumentType,
        Dataset,
        SimpleTextParser,
        RecursiveCharacterChunker,
        BaseVector
    )
except ImportError as e:
    import traceback
    traceback.print_exc()
    logger.error(f"Failed to import langrag components: {e}")
    sys.exit(1)


# === Mock Implementation of Vector Store ===
# Since we might not have a running vector DB for this demo, we use an in-memory one.
class InMemoryVectorStore(BaseVector):
    """Simple in-memory vector store for demonstration."""
    
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        self.documents: List[Document] = []
        logger.info(f"Initialized InMemoryVectorStore for collection '{self.collection_name}'")

    def create(self, texts: list[Document], **kwargs) -> None:
        self.add_texts(texts, **kwargs)

    def add_texts(self, texts: list[Document], **kwargs) -> None:
        self.documents.extend(texts)
        logger.info(f"Stored {len(texts)} documents in memory.")

    def search(
        self, 
        query: str, 
        query_vector: list[float] | None, 
        top_k: int = 4, 
        **kwargs
    ) -> list[Document]:
        # Perform simple keyword matching if no vectors or just for demo
        results = []
        query_lower = query.lower()
        
        # Simple scoring based on term presence
        for doc in self.documents:
            score = 0
            if query_lower in doc.page_content.lower():
                score = 1.0
            
            # If we had vectors, we would do cosine similarity here
            
            if score > 0:
                # Add score to metadata for result processing
                doc.metadata['score'] = score
                results.append(doc)
        
        # Sort by score
        results.sort(key=lambda x: x.metadata.get('score', 0), reverse=True)
        return results[:top_k]
        
    def delete_by_ids(self, ids: list[str]) -> None:
        pass
        
    def delete(self) -> None:
        self.documents = []


# === Utility Functions ===

def create_sample_document() -> Path:
    """Create a sample document for demonstration."""
    sample_file = Path("sample.txt")
    sample_content = """Retrieval-Augmented Generation (RAG) is a technique that combines information
retrieval with text generation. It allows language models to access external
knowledge bases to provide more accurate and up-to-date responses.

The RAG process consists of two main phases: indexing and retrieval. During
indexing, documents are parsed, chunked, embedded, and stored in a vector
database. During retrieval, user queries are embedded and similar chunks
are retrieved to provide context for generation.

RAG systems typically use dense vector representations created by neural
embedding models. These embeddings capture semantic meaning, allowing the
system to find relevant information even when exact keyword matches don't exist.

The quality of a RAG system depends on several factors: the chunking strategy,
the embedding model quality, the vector store's search algorithm, and optional
reranking mechanisms that refine the initial retrieval results.
"""
    sample_file.write_text(sample_content, encoding="utf-8")
    return sample_file


def main():
    logger.info("Starting LangRAG Phase 2 Demo")
    
    # 1. Setup Context
    dataset = Dataset(
        id="demo_dataset",
        tenant_id="demo_tenant",
        name="RAG Demo",
        description="A demo dataset for RAG",
        indexing_technique="high_quality", # implies semantic search
        collection_name="demo_collection"
    )
    
    # 2. Pipeline Components
    parser = SimpleTextParser()
    chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=20)
    vector_store = InMemoryVectorStore(dataset)
    
    # 3. Indexing Phase
    logger.info("--- Phase 1: Indexing ---")
    
    # a. Create & Load
    sample_path = create_sample_document()
    try:
        raw_docs = parser.parse(sample_path)
        logger.info(f"Loaded {len(raw_docs)} document(s) from {sample_path}")
        
        # b. Clean (Skip for now as SimpleTextParser is clean enough)
        
        # c. Split
        chunks = chunker.split(raw_docs)
        logger.info(f"Split into {len(chunks)} chunks")
        
        # d. Embed (Skipped in this demo as we use Mock VDB)
        # In real app: embedder.embed([c.page_content for c in chunks])
        
        # e. Store
        vector_store.add_texts(chunks)
        
    finally:
        if sample_path.exists():
            sample_path.unlink()
            
    # 4. Retrieval Phase
    logger.info("--- Phase 2: Retrieval ---")
    
    query = "indexing and retrieval"
    logger.info(f"Query: '{query}'")
    
    results = vector_store.search(query, query_vector=None, top_k=3)
    
    logger.info(f"Found {len(results)} results:")
    for i, doc in enumerate(results, 1):
        preview = doc.page_content.replace('\n', ' ')[:100]
        logger.info(f"[{i}] Score: {doc.metadata.get('score')}: {preview}...")
        
    logger.info("Demo complete!")


if __name__ == "__main__":
    main()
