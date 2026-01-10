from typing import Any

from loguru import logger

from langrag.datasource.vdb.base import BaseVector
from langrag.entities.dataset import Dataset
from langrag.entities.document import Document
from langrag.index_processor.processor.base import BaseIndexProcessor
from langrag.llm.base import BaseLLM
from langrag.llm.embedder.base import BaseEmbedder

QA_GEN_PROMPT = """
You are a helpful assistant. Generate 1 question that can be answered by the following text.
Text:
{text}
Question:
"""


class QAProcessingError(Exception):
    """Raised when QA processing encounters errors."""
    pass


class QAIndexProcessor(BaseIndexProcessor):
    """
    QA Indexing Processor.
    Generates questions from document chunks and indexes the questions.
    """

    def __init__(
        self,
        vector_store: BaseVector,
        llm: BaseLLM,
        embedder: BaseEmbedder,
        splitter: Any = None
    ):
        self.vector_store = vector_store
        self.llm = llm
        self.embedder = embedder
        self.splitter = splitter

    def process(self, dataset: Dataset, documents: list[Document], **_kwargs) -> dict:
        """
        Process documents to generate QA pairs.

        Args:
            dataset: Dataset configuration
            documents: List of documents to process
            **_kwargs: Additional arguments (unused)

        Returns:
            dict: Processing statistics including:
                - total_chunks: Total number of chunks processed
                - successful: Number of successful QA generations
                - failed: Number of failed QA generations
                - failed_chunk_ids: List of chunk IDs that failed
        """
        # 1. Split (if needed)
        chunks = self.splitter.split_documents(documents) if self.splitter else documents

        qa_documents = []
        stats = {
            "total_chunks": len(chunks),
            "successful": 0,
            "failed": 0,
            "failed_chunk_ids": []
        }

        # 2. Generate Questions
        for idx, chunk in enumerate(chunks):
            chunk_id = chunk.id or f"chunk_{idx}"
            try:
                prompt = QA_GEN_PROMPT.format(text=chunk.page_content)
                response = self.llm.chat([{"role": "user", "content": prompt}])
                question = response.strip()

                if not question:
                    logger.warning(
                        f"Empty question generated for chunk '{chunk_id}', skipping"
                    )
                    stats["failed"] += 1
                    stats["failed_chunk_ids"].append(chunk_id)
                    continue

                qa_doc = Document(
                    page_content=question,
                    metadata={
                        "answer": chunk.page_content,
                        "original_doc_id": chunk.metadata.get("document_id"),
                        "source_chunk_id": chunk_id,
                        "dataset_id": dataset.id,
                        "is_qa": True
                    }
                )
                qa_documents.append(qa_doc)
                stats["successful"] += 1

            except Exception as e:
                stats["failed"] += 1
                stats["failed_chunk_ids"].append(chunk_id)
                logger.error(
                    f"Failed to generate QA for chunk '{chunk_id}': {type(e).__name__}: {e}"
                )

        # Log summary
        if stats["failed"] > 0:
            logger.warning(
                f"QA generation completed with {stats['failed']}/{stats['total_chunks']} failures. "
                f"Failed chunk IDs: {stats['failed_chunk_ids'][:10]}"  # Log first 10
                + (f"... and {len(stats['failed_chunk_ids']) - 10} more"
                   if len(stats['failed_chunk_ids']) > 10 else "")
            )

        if not qa_documents:
            logger.warning("No QA documents generated, skipping indexing")
            return stats

        # 3. Embed (Questions)
        texts = [d.page_content for d in qa_documents]
        embeddings = self.embedder.embed(texts)
        for i, doc in enumerate(qa_documents):
            doc.vector = embeddings[i]

        # 4. Save to VDB
        self.vector_store.create(qa_documents)

        logger.info(
            f"QA indexing completed: {stats['successful']} QA pairs indexed"
        )

        return stats
