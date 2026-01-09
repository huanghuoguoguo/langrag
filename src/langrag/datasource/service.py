from typing import Any

from langrag.entities.dataset import Dataset
from langrag.entities.document import Document

# from langrag.datasource.keyword.base import BaseKeyword  # TODO: Implement Factory

class RetrievalService:
    """
    Unified service for retrieving data from underlying datasources.
    Handles concurrency and merging of results from different sources (within a single dataset).
    """

    @staticmethod
    def retrieve(
        dataset: Dataset,
        query: str,
        query_vector: list[float] | None = None,
        retrieval_method: str = "semantic_search", # semantic_search, keyword_search, hybrid_search
        top_k: int = 4,
        vector_manager: Any = None, # Optional injection
        reranking_model: dict | None = None,
    ) -> list[Document]:

        # 1. Initialize Manager
        if vector_manager is None:
             from langrag.datasource.vdb.global_manager import get_vector_manager
             vector_manager = get_vector_manager()

        # 2. Execute Search via Manager
        if retrieval_method == "semantic_search":
             return vector_manager.search(dataset, query, query_vector, top_k=top_k, search_type="similarity")

        elif retrieval_method == "hybrid_search":
             return vector_manager.search(dataset, query, query_vector, top_k=top_k, search_type="hybrid")

        elif retrieval_method == "keyword_search":
             # Manager could route to keyword store logic if implemented
             pass

        return []

    # TODO: Add concurrent execution for multi-path retrieval if needed internally
