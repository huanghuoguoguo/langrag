import logging
from typing import List, Optional

from langrag.entities.dataset import Dataset, RetrievalContext
from langrag.datasource.service import RetrievalService
from langrag.retrieval.post_processor import PostProcessor
# from langrag.retrieval.router.base import BaseRouter
# from langrag.retrieval.rerank.base import BaseReranker

logger = logging.getLogger(__name__)

class RetrievalWorkflow:
    """
    Orchestrates the retrieval process:
    Query -> [Router] -> [RetrievalService] -> [Reranker] -> Results
    """

    def __init__(
        self, 
        router=None,   # BaseRouter
        reranker=None, # BaseReranker
        vector_store_cls=None
    ):
        self.router = router
        self.reranker = reranker
        self.vector_store_cls = vector_store_cls
        self.post_processor = PostProcessor()

    def retrieve(
        self, 
        query: str, 
        datasets: List[Dataset],
        top_k: int = 4,
        score_threshold: float = 0.0,
        rerank_top_k: Optional[int] = None
    ) -> List[RetrievalContext]:
        """
        Execute the retrieval workflow.
        
        Args:
            query: User query string.
            datasets: List of available datasets to potentially search.
            top_k: Initial retrieval count per dataset.
            score_threshold: Minimum score filter.
            rerank_top_k: How many results to return after reranking.
        """
        
        # 1. Routing (Agentic Step)
        # If router is present and we have multiple datasets, asking router which ones to query.
        selected_datasets = datasets
        if self.router and len(datasets) > 1:
            try:
                selected_datasets = self.router.route(query, datasets)
                logger.info(f"Router selected {len(selected_datasets)} datasets: {[d.name for d in selected_datasets]}")
            except Exception as e:
                logger.warning(f"Router failed, falling back to all datasets: {e}")

        if not selected_datasets:
            return []

        # 2. Parallel Retrieval (Dispatch)
        # TODO: This should be parallelized with ThreadPoolExecutor for production
        all_documents = []
        for dataset in selected_datasets:
            try:
                # Decide method based on dataset config (e.g. if economy -> keyword)
                if dataset.indexing_technique == 'economy':
                     method = "keyword_search"
                else:
                     method = "semantic_search" # or hybrid if configured

                docs = RetrievalService.retrieve(
                    dataset=dataset,
                    query=query,
                    retrieval_method=method,
                    top_k=top_k,
                    vector_store_cls=self.vector_store_cls
                )
                all_documents.extend(docs)
                
            except Exception as e:
                logger.error(f"Error retrieving from dataset {dataset.name}: {e}")

        # 3. Reranking (Optimization)
        if self.reranker and all_documents:
            try:
                # Convert Document objects to format expected by Reranker if needed
                # Reranker should return sorted list of Documents
                # all_documents = self.reranker.rerank(query, all_documents, top_n=rerank_top_k or top_k)
                pass # Placeholder for reranker call
            except Exception as e:
                logger.error(f"Reranking failed: {e}")

        # 4. Post Processing (Deduplication & Thresholding)
        all_documents = self.post_processor.run(all_documents, score_threshold=score_threshold)

        # 5. Context Formatting
        results = []
        for doc in all_documents:
            # Score is guaranteed to be >= threshold
            score = doc.metadata.get('score', 0.0)

            results.append(RetrievalContext(
                document_id=doc.metadata.get('document_id', 'unknown'),
                content=doc.page_content,
                score=score,
                metadata=doc.metadata
            ))

        # Basic sort if no reranker was used (and if PostProcessor didn't reorder, which it doesn't)
        if not self.reranker:
            results.sort(key=lambda x: x.score, reverse=True)
            
        final_k = rerank_top_k or top_k
        return results[:final_k]
