from typing import Any, List, Optional
from uuid import uuid4

from langrag.datasource.vdb.base import BaseVector
from langrag.entities.dataset import Dataset
from langrag.entities.document import Document
from langrag.entities.search_result import SearchResult

class InMemoryVectorStore(BaseVector):
    """
    A simple in-memory vector store for testing purposes.
    Stores documents in a list and performs exact vector similarity (dot product).
    """

    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        self._docs: List[Document] = []

    def create(self, texts: List[Document], **kwargs: Any) -> None:
        """Create the collection (noop) and add texts."""
        self.add_texts(texts, **kwargs)

    def add_texts(self, texts: List[Document], **kwargs: Any) -> None:
        """Add texts to the store."""
        for doc in texts:
            # Ensure doc has an ID
            if not doc.id:
                doc.id = str(uuid4())
            self._docs.append(doc)

    def search(
        self,
        query: str,
        query_vector: Optional[List[float]],
        top_k: int = 4,
        **kwargs: Any
    ) -> List[Document]:
        """
        Search for documents by vector similarity.
        Requires query_vector to be present for semantic search.
        """
        if not query_vector:
            # Fallback to simple keyword matching if no vector (just for basic testing)
            results = []
            for doc in self._docs:
                if query.lower() in doc.page_content.lower():
                    results.append(doc)
            return results[:top_k]

        # Calculate cosine similarity manually
        scored_docs = []
        for doc in self._docs:
            if doc.vector:
                score = self._cosine_similarity(query_vector, doc.vector)
                # Inject score into metadata so it can be viewed
                doc.metadata['score'] = score
                scored_docs.append((doc, score))
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scored_docs[:top_k]]

    def delete_by_ids(self, ids: List[str]) -> None:
        self._docs = [d for d in self._docs if d.id not in ids]

    def delete(self) -> None:
        self._docs = []

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm_a = sum(a * a for a in v1) ** 0.5
        norm_b = sum(b * b for b in v2) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)
