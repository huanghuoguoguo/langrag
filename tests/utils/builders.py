"""Test builders - Use builder pattern to create test objects."""

from typing import Any, Dict, List, Optional
from langrag.config.models import ComponentConfig, RAGConfig, VectorStoreConfig
from langrag.entities.document import Document, DocumentType

class RAGConfigBuilder:
    """Builder for RAGConfig."""

    def __init__(self):
        self._parser = ComponentConfig(type="simple_text", params={})
        self._chunker = ComponentConfig(
            type="recursive", params={"chunk_size": 500, "chunk_overlap": 50}
        )
        self._embedder = ComponentConfig(type="simple", params={"dimension": 384})
        self._vector_store = VectorStoreConfig(type="in_memory", params={})
        self._reranker = None
        self._llm = None

    def with_parser(self, parser_type: str, **params):
        self._parser = ComponentConfig(type=parser_type, params=params)
        return self

    def with_chunker(self, chunker_type: str, **params):
        self._chunker = ComponentConfig(type=chunker_type, params=params)
        return self

    def with_embedder(self, embedder_type: str, **params):
        self._embedder = ComponentConfig(type=embedder_type, params=params)
        return self

    def with_vector_store(self, store_type: str, **params):
        self._vector_store = VectorStoreConfig(type=store_type, params=params)
        return self

    def with_reranker(self, reranker_type: str, **params):
        self._reranker = ComponentConfig(type=reranker_type, params=params)
        return self

    def with_llm(self, llm_type: str, **params):
        self._llm = ComponentConfig(type=llm_type, params=params)
        return self

    def build(self) -> RAGConfig:
        return RAGConfig(
            parser=self._parser,
            chunker=self._chunker,
            embedder=self._embedder,
            vector_store=self._vector_store,
            reranker=self._reranker,
            llm=self._llm,
        )


class DocumentBuilder:
    """Builder for Document entity."""

    def __init__(self):
        self._page_content = "Sample document content"
        self._metadata: Dict[str, Any] = {}
        self._vector: Optional[List[float]] = None
        self._type = DocumentType.ORIGINAL
        self._id = None

    def with_content(self, content: str):
        self._page_content = content
        return self

    def with_metadata(self, **metadata):
        self._metadata.update(metadata)
        return self

    def with_vector(self, vector: List[float]):
        self._vector = vector
        return self
        
    def with_type(self, doc_type: DocumentType):
        self._type = doc_type
        return self

    def with_id(self, doc_id: str):
        self._id = doc_id
        return self

    def as_chunk(self):
        self._type = DocumentType.CHUNK
        return self

    def build(self) -> Document:
        return Document(
            page_content=self._page_content,
            metadata=self._metadata,
            vector=self._vector,
            type=self._type,
            id=self._id
        )
