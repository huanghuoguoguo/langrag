"""
Document processing service for the Web layer.

This module handles the complete document processing pipeline:
Parse → Chunk → Embed → Store

It demonstrates how to use LangRAG's index processing components to build
a document ingestion pipeline for a RAG application.

Design Decisions:
-----------------
1. **Parser Selection by File Type**: The processor automatically selects the
   appropriate parser based on file extension. This follows the "convention over
   configuration" principle - users don't need to specify parsers manually.

2. **Indexing Techniques**: Three indexing strategies are supported:
   - high_quality: Standard chunking with embeddings (default)
   - qa: QA-pair generation for question-based retrieval
   - parent_child: Hierarchical chunks for context-rich retrieval

3. **Dependency Injection**: The processor receives embedder, LLM adapter, and
   KV store as dependencies, making it testable and flexible.

4. **Fail Fast**: Embedding errors are not silently ignored - they raise
   exceptions to ensure data integrity.

Example Usage:
--------------
    processor = DocumentProcessor(
        embedder=embedder,
        llm_adapter=llm_adapter,
        kv_store=kv_store
    )

    chunk_count = processor.process(
        file_path=Path("document.pdf"),
        vector_store=store,
        chunk_size=500,
        chunk_overlap=50,
        indexing_technique="high_quality"
    )
"""

import logging
from pathlib import Path

from langrag import (
    BaseEmbedder,
    BaseVector,
    ParentChildIndexProcessor,
    QAIndexProcessor,
    RecursiveCharacterChunker,
    SimpleTextParser,
)
from langrag import (
    Document as LangRAGDocument,
)
from langrag.datasource.kv.base import BaseKVStore
from langrag.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Document processing service that orchestrates the indexing pipeline.

    This class encapsulates the logic for processing documents into a vector store,
    supporting multiple indexing techniques and file formats.

    The processing pipeline:
    1. Parse: Extract text from documents (PDF, Markdown, HTML, DOCX, TXT)
    2. Chunk: Split text into manageable pieces using configured strategy
    3. Embed: Generate vector embeddings for each chunk
    4. Store: Save chunks and embeddings to the vector store

    Attributes:
        embedder: The embedding model to use (optional for keyword-only search)
        llm_adapter: LLM for QA indexing technique (optional)
        kv_store: Key-value store for parent-child indexing (optional)
    """

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        llm_adapter: BaseLLM | None = None,
        kv_store: BaseKVStore | None = None
    ):
        """
        Initialize the document processor.

        Args:
            embedder: Embedding model for generating vectors. Required for
                     vector search, optional for keyword-only search.
            llm_adapter: LLM adapter for QA indexing technique. Required only
                        when using indexing_technique="qa".
            kv_store: Key-value store for parent-child indexing. Required only
                     when using indexing_technique="parent_child".
        """
        self.embedder = embedder
        self.llm_adapter = llm_adapter
        self.kv_store = kv_store

    def _get_parser(self, file_path: Path):
        """
        Select the appropriate parser based on file extension.

        This method implements automatic parser selection, falling back to
        SimpleTextParser if a specialized parser is unavailable.

        Parser selection logic:
        - .pdf → PdfParser (requires pypdf)
        - .md/.markdown → MarkdownParser
        - .html/.htm → HtmlParser
        - .docx/.doc → DocxParser (requires python-docx)
        - others → SimpleTextParser

        Args:
            file_path: Path to the document file

        Returns:
            An instance of the appropriate parser
        """
        file_ext = file_path.suffix.lower()

        if file_ext == '.pdf':
            try:
                from langrag.index_processor.extractor.providers.pdf import PdfParser
                logger.info(f"Using PdfParser for {file_ext} file")
                return PdfParser()
            except ImportError:
                logger.warning("pypdf not installed, falling back to SimpleTextParser")
                return SimpleTextParser()

        elif file_ext in ['.md', '.markdown']:
            try:
                from langrag.index_processor.extractor.providers.markdown import MarkdownParser
                logger.info(f"Using MarkdownParser for {file_ext} file")
                return MarkdownParser()
            except ImportError:
                return SimpleTextParser()

        elif file_ext in ['.html', '.htm']:
            try:
                from langrag.index_processor.extractor.providers.html import HtmlParser
                logger.info(f"Using HtmlParser for {file_ext} file")
                return HtmlParser()
            except ImportError:
                return SimpleTextParser()

        elif file_ext in ['.docx', '.doc']:
            try:
                from langrag.index_processor.extractor.providers.docx import DocxParser
                logger.info(f"Using DocxParser for {file_ext} file")
                return DocxParser()
            except ImportError:
                return SimpleTextParser()

        else:
            logger.info(f"Using SimpleTextParser for {file_ext} file")
            return SimpleTextParser()

    def _process_high_quality(
        self,
        raw_docs: list[LangRAGDocument],
        vector_store: BaseVector,
        chunk_size: int,
        chunk_overlap: int
    ) -> int:
        """
        Process documents using the standard high-quality indexing technique.

        This is the default indexing method that:
        1. Chunks documents using recursive character splitting
        2. Embeds each chunk (if embedder is configured)
        3. Stores chunks in the vector store

        Args:
            raw_docs: Parsed documents from the parser
            vector_store: Target vector store
            chunk_size: Maximum characters per chunk
            chunk_overlap: Character overlap between chunks

        Returns:
            Number of chunks created
        """
        # Chunk the documents
        logger.info(f"Chunking with size={chunk_size}, overlap={chunk_overlap}...")
        chunker = RecursiveCharacterChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = chunker.split(raw_docs)
        logger.info(f"Created {len(chunks)} chunks")

        # Embed if embedder is available
        if self.embedder:
            text_list = [c.page_content for c in chunks]
            try:
                logger.info(f"Embedding {len(text_list)} chunks with {self.embedder.__class__.__name__}...")
                vectors = self.embedder.embed(text_list)
                logger.info(f"Received {len(vectors)} embedding vectors")

                # Attach vectors to documents
                for doc, vec in zip(chunks, vectors):
                    doc.vector = vec

                logger.info("Embeddings attached to chunks")
            except Exception as e:
                logger.error(f"Embedding error: {e}")
                # Fail fast - don't store documents without embeddings if embedder was configured
                raise
        else:
            logger.info("No embedder configured, skipping embedding")

        # Store in vector store
        logger.info("Storing chunks in vector store...")
        vector_store.add_texts(chunks)
        logger.info(f"Successfully stored {len(chunks)} chunks")

        return len(chunks)

    def _process_qa(
        self,
        raw_docs: list[LangRAGDocument],
        vector_store: BaseVector,
        chunk_size: int,
        chunk_overlap: int
    ) -> int:
        """
        Process documents using QA indexing technique.

        QA indexing generates question-answer pairs from the document content
        using an LLM, then indexes the questions for retrieval. This is
        particularly effective for FAQ-style content or when users are
        expected to ask questions in natural language.

        How it works:
        1. Chunk the document
        2. For each chunk, generate relevant questions using LLM
        3. Index questions (linked to original answers)
        4. At retrieval time, match user query against questions
        5. Return the associated answers

        Args:
            raw_docs: Parsed documents
            vector_store: Target vector store
            chunk_size: Maximum characters per chunk
            chunk_overlap: Character overlap between chunks

        Returns:
            Estimated number of QA pairs created

        Raises:
            ValueError: If LLM or embedder is not configured
        """
        logger.info("Using QA Indexing Technique")

        if not self.llm_adapter:
            raise ValueError("LLM not configured, cannot use QA indexing")
        if not self.embedder:
            raise ValueError("Embedder not configured, cannot use QA indexing")

        chunker = RecursiveCharacterChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        processor = QAIndexProcessor(
            vector_store=vector_store,
            llm=self.llm_adapter,
            embedder=self.embedder,
            splitter=chunker
        )

        processor.process(vector_store.dataset, raw_docs)

        # Return estimate (actual count not tracked by processor)
        return len(raw_docs) * 2

    def _process_parent_child(
        self,
        raw_docs: list[LangRAGDocument],
        vector_store: BaseVector,
        chunk_size: int,
        chunk_overlap: int
    ) -> int:
        """
        Process documents using parent-child indexing technique.

        Parent-child indexing creates a two-level hierarchy:
        - Parent chunks: Larger context windows for retrieval response
        - Child chunks: Smaller chunks optimized for similarity matching

        How it works:
        1. Split into large "parent" chunks
        2. Further split parents into smaller "child" chunks
        3. Index child chunks in vector store
        4. Store parent chunks in KV store
        5. At retrieval, match against children, return parents

        Benefits:
        - Better semantic matching (small chunks)
        - Richer context in response (large parents)
        - Reduced hallucination from truncated context

        Args:
            raw_docs: Parsed documents
            vector_store: Target vector store for child chunks
            chunk_size: Base chunk size (parent = 2x, child = 0.5x)
            chunk_overlap: Base overlap

        Returns:
            Estimated number of child chunks created

        Raises:
            ValueError: If embedder or KV store is not configured
        """
        logger.info("Using Parent-Child Indexing Technique")

        if not self.embedder:
            raise ValueError("Embedder not configured, cannot use Parent-Child indexing")
        if not self.kv_store:
            raise ValueError("KV store not configured, cannot use Parent-Child indexing")

        # Configure chunk sizes
        # Parent: larger for context richness
        # Child: smaller for precise matching
        parent_chunk_size = chunk_size * 2
        child_chunk_size = max(chunk_size // 2, 200)  # Minimum 200 chars

        parent_splitter = RecursiveCharacterChunker(
            chunk_size=parent_chunk_size,
            chunk_overlap=chunk_overlap
        )
        child_splitter = RecursiveCharacterChunker(
            chunk_size=child_chunk_size,
            chunk_overlap=chunk_overlap // 2
        )

        processor = ParentChildIndexProcessor(
            vector_store=vector_store,
            kv_store=self.kv_store,
            embedder=self.embedder,
            parent_splitter=parent_splitter,
            child_splitter=child_splitter
        )

        processor.process(vector_store.dataset, raw_docs)

        # Return estimate
        return len(raw_docs) * 4

    def process(
        self,
        file_path: Path,
        vector_store: BaseVector,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        indexing_technique: str = "high_quality"
    ) -> int:
        """
        Process a document through the complete indexing pipeline.

        This is the main entry point for document processing. It:
        1. Selects the appropriate parser based on file type
        2. Parses the document into raw text
        3. Applies the specified indexing technique
        4. Returns the number of chunks/items created

        Args:
            file_path: Path to the document file
            vector_store: Target vector store for the processed chunks
            chunk_size: Maximum characters per chunk (default: 500)
            chunk_overlap: Character overlap between chunks (default: 50)
            indexing_technique: One of "high_quality", "qa", "parent_child"

        Returns:
            Number of chunks or items created

        Raises:
            ValueError: If required dependencies are missing for the technique
            Exception: If parsing or processing fails
        """
        logger.info(f"Processing document: {file_path}, technique={indexing_technique}")

        # Step 1: Parse document
        parser = self._get_parser(file_path)
        raw_docs = parser.parse(file_path)
        logger.info(f"Parsed {len(raw_docs)} raw documents")

        # Step 2: Apply indexing technique
        if indexing_technique == 'qa':
            return self._process_qa(raw_docs, vector_store, chunk_size, chunk_overlap)

        elif indexing_technique == 'parent_child':
            return self._process_parent_child(raw_docs, vector_store, chunk_size, chunk_overlap)

        else:
            # Default: high_quality
            return self._process_high_quality(raw_docs, vector_store, chunk_size, chunk_overlap)
