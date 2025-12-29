import concurrent.futures
from typing import List, Optional

from langrag.entities.dataset import Dataset
from langrag.entities.document import Document
from langrag.datasource.vdb.base import BaseVector
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
        vector_store_cls: type[BaseVector] = None, # Dependency Injection ideally
        reranking_model: dict | None = None,
    ) -> list[Document]:
        
        # 1. Initialize Datasource (Should use a Factory here)
        # For now, we assume vector_store_cls is passed in or we pick a default
        # 1. Initialize Datasource (Should use a Factory here)
        # For now, we assume vector_store_cls is passed in or we pick a default
        if vector_store_cls is None:
             from langrag.datasource.vdb.factory import VectorStoreFactory
             vector_store = VectorStoreFactory.get_vector_store(dataset)
        else:
             vector_store = vector_store_cls(dataset)
        
        # 2. Execute Search
        if retrieval_method == "semantic_search":
             return vector_store.search(query, query_vector, top_k=top_k, search_type="similarity")
             
        elif retrieval_method == "hybrid_search":
             # If the underlying VDB supports hybrid naturally, call it.
             # Otherwise, we might need manual RRF here (but we aim to push down to VDB)
             return vector_store.search(query, query_vector, top_k=top_k, search_type="hybrid")
             
        elif retrieval_method == "keyword_search":
             # Call keyword store (omitted for brevity, would be similar to vector_store)
             pass
             
        return []

    # TODO: Add concurrent execution for multi-path retrieval if needed internally
