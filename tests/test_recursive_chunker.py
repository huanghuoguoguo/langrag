"""Integration tests for the RecursiveCharacterChunker.

This test uses a realistic technical document to verify that the chunker
produces semantically coherent chunks.
"""

from pathlib import Path
import tempfile

from langrag.chunker import RecursiveCharacterChunker, FixedSizeChunker
from langrag.core.document import Document


# Sample technical content (DGA/RAG related)
SAMPLE_TECH_DOCUMENT = """Retrieval-Augmented Generation: A Comprehensive Overview

Abstract

Retrieval-Augmented Generation (RAG) represents a paradigm shift in how large language models interact with external knowledge. By combining dense retrieval mechanisms with generative models, RAG systems can produce more accurate, up-to-date, and verifiable responses compared to purely parametric approaches.

Introduction

Traditional language models store knowledge in their parameters during pre-training. However, this approach has several limitations: knowledge becomes stale over time, the model cannot cite sources, and updating knowledge requires expensive retraining. RAG addresses these challenges by augmenting the generation process with a retrieval step that accesses an external knowledge base.

The retrieval-augmented generation architecture consists of two main components: a retriever and a generator. The retriever is responsible for finding relevant documents or passages from a large corpus, while the generator produces the final output conditioned on both the input query and the retrieved context.

Retrieval Mechanisms

Modern RAG systems typically employ dense retrieval using learned embedding models. These models encode both queries and documents into a shared vector space, where semantic similarity can be computed efficiently using approximate nearest neighbor search algorithms.

The quality of retrieval is critical to RAG performance. Several factors influence retrieval effectiveness: the choice of embedding model, the chunking strategy for long documents, and the similarity metric used for ranking. Recent research has shown that fine-tuning embedders on domain-specific data significantly improves retrieval accuracy.

Chunking Strategies

Document chunking is a crucial preprocessing step in RAG systems. The goal is to split long documents into smaller, semantically coherent segments that can be effectively embedded and retrieved. Poor chunking can lead to fragmented context and reduced answer quality.

Several chunking approaches have been proposed: fixed-size chunking with overlap, sentence-based chunking, paragraph-based chunking, and semantic chunking that uses document structure. Recursive character splitting, which tries multiple separators in order of semantic significance, offers a good balance between simplicity and effectiveness.

Empirical studies have demonstrated that chunk size significantly impacts retrieval performance. Chunks that are too small may lack sufficient context, while overly large chunks can dilute relevant information with noise. Typical chunk sizes range from 200 to 1000 tokens, with 512 tokens being a common choice.

Generation with Retrieved Context

Once relevant chunks are retrieved, they are concatenated with the user's query and fed to the language model. The model then generates a response that ideally synthesizes information from the retrieved context while maintaining coherence and relevance.

Advanced RAG systems incorporate reranking steps to improve the quality of retrieved chunks before generation. Cross-encoder models, which jointly encode the query and candidate chunk, can provide more accurate relevance scores than bi-encoder retrieval models.

Challenges and Future Directions

Despite significant progress, RAG systems face several challenges. Retrieval can introduce latency, especially when searching large corpora. The system may retrieve irrelevant or contradictory information, leading to confused or incorrect outputs. Handling multi-hop reasoning, where answering a question requires combining information from multiple sources, remains difficult.

Future research directions include improving retrieval diversity to avoid echo chambers, developing better metrics for evaluating RAG systems beyond traditional QA benchmarks, and exploring hybrid approaches that combine retrieval with model editing or fine-tuning.

Conclusion

Retrieval-Augmented Generation has emerged as a powerful approach for building more reliable and knowledge-grounded language model applications. By separating parametric knowledge from non-parametric retrieval, RAG systems offer flexibility and maintainability that purely generative models cannot match. As embedding models, retrieval algorithms, and generation techniques continue to improve, RAG will play an increasingly important role in production NLP systems."""


def test_recursive_chunker_vs_fixed_size():
    """Compare RecursiveCharacterChunker with FixedSizeChunker."""

    # Create a document
    doc = Document(
        content=SAMPLE_TECH_DOCUMENT,
        metadata={"title": "RAG Overview", "type": "technical"}
    )

    # Test RecursiveCharacterChunker
    print("\n" + "="*80)
    print("RECURSIVE CHARACTER CHUNKER (chunk_size=500, overlap=50)")
    print("="*80)

    recursive_chunker = RecursiveCharacterChunker(
        chunk_size=500,
        chunk_overlap=50
    )
    recursive_chunks = recursive_chunker.split([doc])

    print(f"\nTotal chunks: {len(recursive_chunks)}")
    print(f"Average chunk size: {sum(len(c.content) for c in recursive_chunks) / len(recursive_chunks):.1f} chars")

    for i, chunk in enumerate(recursive_chunks[:5], 1):  # Show first 5 chunks
        print(f"\n--- Chunk {i} ({len(chunk.content)} chars) ---")
        # Show first 200 chars and last 100 chars to see boundaries
        preview = chunk.content[:200] + "\n...\n" + chunk.content[-100:] if len(chunk.content) > 300 else chunk.content
        print(preview)
        print(f"Metadata: {chunk.metadata.get('chunking_method', 'N/A')}")

    # Test FixedSizeChunker for comparison
    print("\n" + "="*80)
    print("FIXED SIZE CHUNKER (chunk_size=500, overlap=50)")
    print("="*80)

    fixed_chunker = FixedSizeChunker(
        chunk_size=500,
        overlap=50  # Note: FixedSizeChunker uses 'overlap', not 'chunk_overlap'
    )
    fixed_chunks = fixed_chunker.split([doc])

    print(f"\nTotal chunks: {len(fixed_chunks)}")
    print(f"Average chunk size: {sum(len(c.content) for c in fixed_chunks) / len(fixed_chunks):.1f} chars")

    for i, chunk in enumerate(fixed_chunks[:3], 1):  # Show first 3 chunks
        print(f"\n--- Chunk {i} ({len(chunk.content)} chars) ---")
        preview = chunk.content[:200] + "\n...\n" + chunk.content[-100:] if len(chunk.content) > 300 else chunk.content
        print(preview)

    # Analyze semantic coherence
    print("\n" + "="*80)
    print("SEMANTIC COHERENCE ANALYSIS")
    print("="*80)

    print("\nRecursive Chunker - Chunk boundaries:")
    for i, chunk in enumerate(recursive_chunks[:5], 1):
        # Check if chunk ends with sentence boundary
        ends_with_period = chunk.content.rstrip().endswith(('.', '。', '!', '?'))
        ends_with_para = chunk.content.rstrip().endswith('\n\n')
        print(f"  Chunk {i}: ends_with_period={ends_with_period}, ends_with_paragraph={ends_with_para}")

    print("\nFixed Chunker - Chunk boundaries:")
    for i, chunk in enumerate(fixed_chunks[:5], 1):
        ends_with_period = chunk.content.rstrip().endswith(('.', '。', '!', '?'))
        ends_with_para = chunk.content.rstrip().endswith('\n\n')
        print(f"  Chunk {i}: ends_with_period={ends_with_period}, ends_with_paragraph={ends_with_para}")

    return recursive_chunks, fixed_chunks


def test_with_custom_file():
    """Test with a custom file if provided."""
    test_file = Path("test_document.txt")

    if not test_file.exists():
        print("\nNo test_document.txt found. Skipping custom file test.")
        return

    print("\n" + "="*80)
    print("TESTING WITH CUSTOM FILE: test_document.txt")
    print("="*80)

    content = test_file.read_text(encoding="utf-8")
    doc = Document(
        content=content,
        metadata={"source": str(test_file.absolute())}
    )

    chunker = RecursiveCharacterChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.split([doc])

    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Average chunk size: {sum(len(c.content) for c in chunks) / len(chunks):.1f} chars")

    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n--- Chunk {i} ---")
        print(chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content)


if __name__ == "__main__":
    print("Running RecursiveCharacterChunker integration tests...")
    recursive_chunks, fixed_chunks = test_recursive_chunker_vs_fixed_size()
    test_with_custom_file()

    print("\n" + "="*80)
    print("TESTS COMPLETE")
    print("="*80)
    print(f"\nRecursive chunker created {len(recursive_chunks)} chunks")
    print(f"Fixed chunker created {len(fixed_chunks)} chunks")
    print("\nRecursive chunker should show better semantic boundaries!")
