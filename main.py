#!/usr/bin/env python3
"""
LangRAG Demo Application

Demonstrates the complete indexing and retrieval flow using RAGEngine.
"""

from pathlib import Path
import yaml
from loguru import logger

from langrag import RAGEngine, RAGConfig


def load_config(config_path: str = "config.yaml") -> RAGConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Parsed RAG configuration
    """
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    return RAGConfig(**config_dict)


def create_sample_document() -> Path:
    """Create a sample document for demonstration.

    Returns:
        Path to the created sample file
    """
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
    sample_file.write_text(sample_content)
    return sample_file


def main():
    """Run the demo."""
    logger.info("=" * 60)
    logger.info("LangRAG Phase 1 Demo")
    logger.info("=" * 60)

    # Load configuration and initialize engine
    config = load_config()
    engine = RAGEngine(config)

    # === INDEXING PHASE ===
    logger.info("")
    logger.info("=" * 60)
    logger.info("INDEXING PHASE")
    logger.info("=" * 60)

    sample_file = create_sample_document()
    logger.info(f"Created sample document: {sample_file}")

    num_chunks = engine.index(sample_file)
    logger.info(f"✓ Indexed {num_chunks} chunks")

    # === RETRIEVAL PHASE ===
    logger.info("")
    logger.info("=" * 60)
    logger.info("RETRIEVAL PHASE")
    logger.info("=" * 60)

    query = "What are the phases of RAG?"
    logger.info(f"Query: '{query}'")
    logger.info("")

    results = engine.retrieve(query)

    # Display results
    logger.info(f"Found {len(results)} results:")
    logger.info("")

    for i, result in enumerate(results, 1):
        logger.info(f"Result #{i} (score: {result.score:.4f})")
        logger.info(f"  Chunk ID: {result.chunk.id}")
        logger.info(f"  Content preview: {result.chunk.content[:100]}...")
        logger.info(f"  Source: {result.chunk.metadata.get('filename')}")
        logger.info(f"  Chunk index: {result.chunk.metadata.get('chunk_index')}")
        logger.info("")

    # === CLEANUP ===
    logger.info("=" * 60)
    logger.info("CLEANUP")
    logger.info("=" * 60)
    sample_file.unlink()
    logger.info("✓ Cleaned up sample file")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Demo complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
