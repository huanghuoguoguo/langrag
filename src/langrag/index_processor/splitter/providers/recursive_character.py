"""Recursive character-based text chunker with semantic awareness.

This chunker implements a hierarchical splitting strategy similar to LangChain's
RecursiveCharacterTextSplitter, prioritizing semantic boundaries.
"""

from loguru import logger


from langrag.entities.document import Document, DocumentType
from ..base import BaseChunker


class RecursiveCharacterChunker(BaseChunker):
    """Recursively chunks text using a hierarchy of separators.

    This chunker attempts to keep text semantically coherent by splitting
    at natural boundaries in order of preference:
    1. Double newlines (paragraphs)
    2. Single newlines (lines)
    3. Sentences (periods, question marks, exclamation marks)
    4. Spaces (words)
    5. Characters (last resort)

    Attributes:
        chunk_size: Target maximum characters per chunk
        chunk_overlap: Number of characters to overlap between chunks
        separators: List of separator strings in order of preference
        keep_separator: Whether to keep the separator in the chunks
    """

    # Default separators in order of semantic significance
    DEFAULT_SEPARATORS = [
        "\n\n",  # Paragraphs
        "\n",  # Lines
        ". ",  # Sentences (with space after period)
        "。",  # Chinese/Japanese sentence end
        "! ",  # Exclamations
        "? ",  # Questions
        "; ",  # Semicolons
        "；",  # Chinese semicolon
        "，",  # Chinese comma
        ", ",  # Commas
        " ",  # Spaces (words)
        "",  # Characters (fallback)
    ]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
        keep_separator: bool = True,
    ):
        """Initialize the recursive character chunker.

        Args:
            chunk_size: Target maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
            separators: Custom separator list (uses defaults if None)
            keep_separator: Whether to keep separators in chunks

        Raises:
            ValueError: If chunk_size <= 0 or overlap >= chunk_size
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators if separators else self.DEFAULT_SEPARATORS
        self.keep_separator = keep_separator

        logger.info(
            f"Initialized RecursiveCharacterChunker: "
            f"size={chunk_size}, overlap={chunk_overlap}, "
            f"separators={len(self.separators)}"
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks using recursive splitting.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunks with metadata preserved from source documents
        """
        chunks = []

        for doc in documents:
            doc_chunks = self._split_document(doc)
            chunks.extend(doc_chunks)
            logger.debug(
                f"Split document {doc.id} into {len(doc_chunks)} chunks "
                f"(avg size: {sum(len(c.page_content) for c in doc_chunks) / len(doc_chunks):.0f})"
            )

        logger.info(
            f"Created {len(chunks)} chunks from {len(documents)} documents "
            f"(avg size: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f})"
        )
        return chunks

    def _split_document(self, doc: Document) -> list[Document]:
        """Split a single document into chunks.

        Args:
            doc: Document to split

        Returns:
            List of chunks from this document
        """
        text = doc.page_content

        # Split text recursively
        text_chunks = self._split_text_recursive(text, self.separators)

        # Create Chunk objects with metadata
        chunks = []
        char_position = 0

        for i, chunk_text in enumerate(text_chunks):
            # Find actual position in original text (approximate)
            chunk = Document(
                page_content=chunk_text,
                metadata={
                    **doc.metadata,
                    "source_doc_id": doc.id,
                    "chunk_index": i,
                    "chunk_size": len(chunk_text),
                    "chunking_method": "recursive_character",
                    "type": DocumentType.CHUNK,
                },
            )
            chunks.append(chunk)
            char_position += len(chunk_text)

        return chunks

    def _split_text_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using a hierarchy of separators.

        Args:
            text: Text to split
            separators: List of separators to try in order

        Returns:
            List of text chunks
        """
        final_chunks = []

        # Base case: empty separators means split by character
        separator = separators[0] if separators else ""
        new_separators = separators[1:] if len(separators) > 1 else []

        # Split by current separator
        splits = self._split_by_separator(text, separator)

        # Process each split
        current_chunk = []
        current_length = 0

        for split in splits:
            split_len = len(split)

            # If this single split is too large, recursively split it
            if split_len > self.chunk_size:
                # First, flush any accumulated chunks
                if current_chunk:
                    merged = self._merge_chunks(current_chunk, separator)
                    final_chunks.append(merged)
                    current_chunk = []
                    current_length = 0

                # Recursively split the large piece
                if new_separators:
                    sub_chunks = self._split_text_recursive(split, new_separators)
                    final_chunks.extend(sub_chunks)
                else:
                    # Last resort: split by character count
                    final_chunks.extend(self._split_by_character(split))
                continue

            # Check if adding this split would exceed chunk_size
            # Account for separator length when merging
            separator_len = len(separator) if self.keep_separator and current_chunk else 0
            potential_length = current_length + split_len + separator_len

            if current_chunk and potential_length > self.chunk_size:
                # Current chunk is full, save it
                merged = self._merge_chunks(current_chunk, separator)
                final_chunks.append(merged)

                # Start new chunk with overlap
                overlap_chunks = self._get_overlap_chunks(current_chunk, separator)
                current_chunk = overlap_chunks
                current_length = sum(len(c) for c in current_chunk)
                if current_chunk:
                    current_length += len(separator) * (len(current_chunk) - 1)

            # Add current split to chunk
            current_chunk.append(split)
            current_length += split_len
            if len(current_chunk) > 1:
                current_length += separator_len

        # Add remaining chunk
        if current_chunk:
            merged = self._merge_chunks(current_chunk, separator)
            final_chunks.append(merged)

        return final_chunks

    def _split_by_separator(self, text: str, separator: str) -> list[str]:
        """Split text by a separator, handling edge cases.

        Args:
            text: Text to split
            separator: Separator string

        Returns:
            List of split pieces (non-empty)
        """
        if separator == "":
            # Split into characters
            return list(text)

        if self.keep_separator:
            # Split but keep separator at the end of each piece
            splits = text.split(separator)
            result = []
            for split in splits[:-1]:
                if split:  # Skip empty strings
                    result.append(split + separator)
            # Add last piece without separator
            if splits[-1]:
                result.append(splits[-1])
            return result
        else:
            # Simple split, remove empty strings
            return [s for s in text.split(separator) if s]

    def _merge_chunks(self, chunks: list[str], separator: str) -> str:
        """Merge a list of chunks with separator.

        Args:
            chunks: List of text chunks to merge
            separator: Separator to use

        Returns:
            Merged text
        """
        if not chunks:
            return ""

        if self.keep_separator:
            # Chunks already contain separators
            return "".join(chunks)
        else:
            return separator.join(chunks)

    def _get_overlap_chunks(self, chunks: list[str], separator: str) -> list[str]:
        """Get chunks that fit within the overlap size.

        Args:
            chunks: Current chunks
            separator: Separator used

        Returns:
            List of chunks that fit in overlap
        """
        if self.chunk_overlap == 0:
            return []

        overlap_chunks = []
        overlap_length = 0
        separator_len = len(separator) if self.keep_separator else 0

        # Take chunks from the end until we exceed overlap
        for chunk in reversed(chunks):
            chunk_len = len(chunk)
            potential_length = overlap_length + chunk_len
            if overlap_chunks:
                potential_length += separator_len

            if potential_length > self.chunk_overlap:
                break

            overlap_chunks.insert(0, chunk)
            overlap_length = potential_length

        return overlap_chunks

    def _split_by_character(self, text: str) -> list[str]:
        """Split text by character count when all else fails.

        Args:
            text: Text to split

        Returns:
            List of character-based chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.chunk_overlap

        return chunks
