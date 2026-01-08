from typing import Any, List
from langrag.entities.dataset import Dataset
from langrag.entities.document import Document
from langrag.index_processor.processor.base import BaseIndexProcessor
from langrag.datasource.vdb.base import BaseVector
from langrag.llm.base import BaseLLM
from langrag.llm.embedder.base import BaseEmbedder
from loguru import logger

QA_GEN_PROMPT = """
You are a helpful assistant. Generate 1 question that can be answered by the following text.
Text:
{text}
Question:
"""

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

    def process(self, dataset: Dataset, documents: list[Document], **kwargs) -> None:
        
        # 1. Split (if needed) - reusing similar logic as ParagraphProcessor
        chunks = []
        if self.splitter:
            chunks = self.splitter.split_documents(documents)
        else:
            chunks = documents # Assume already chunked or treat as is

        qa_documents = []
        
        # 2. Generate Questions
        for chunk in chunks:
            try:
                # Basic Prompting
                prompt = QA_GEN_PROMPT.format(text=chunk.page_content)
                response = self.llm.chat([{"role": "user", "content": prompt}])
                question = response.strip()
                
                if not question:
                    continue
                    
                # Create a document where content is the Question
                # Metadata contains the original Answer (chunk content)
                qa_doc = Document(
                    page_content=question,
                    metadata={
                        "answer": chunk.page_content,
                        "original_doc_id": chunk.metadata.get("document_id"),
                        "dataset_id": dataset.id,
                        "is_qa": True
                    }
                )
                qa_documents.append(qa_doc)
                
            except Exception as e:
                logger.error(f"Failed to generate QA for chunk: {e}")

        if not qa_documents:
            return

        # 3. Embed (Questions)
        texts = [d.page_content for d in qa_documents]
        embeddings = self.embedder.embed(texts)
        for i, doc in enumerate(qa_documents):
            doc.vector = embeddings[i]

        # 4. Save to VDB
        self.vector_store.create(qa_documents)
