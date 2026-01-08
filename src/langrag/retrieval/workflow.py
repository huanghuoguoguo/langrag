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
        rewriter=None, # BaseRewriter
        vector_store_cls=None
    ):
        self.router = router
        self.reranker = reranker
        self.rewriter = rewriter
        self.vector_store_cls = vector_store_cls
        self.post_processor = PostProcessor()
        # Callbacks (lazy init or passed)
        self.callback_manager = None 

    def set_callback_manager(self, manager):
        self.callback_manager = manager

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
        
        # 0. Callback Start
        run_id = None
        if self.callback_manager:
            run_id = self.callback_manager.on_retrieve_start(query=query)
            
        try:
            # 0.5. Query Rewrite
            original_query = query
            if self.rewriter:
                try:
                    query = self.rewriter.rewrite(query)
                    logger.info(f"Query rewritten: '{original_query}' -> '{query}'")
                except Exception as e:
                    logger.error(f"Query rewrite failed: {e}")

            # 1. Routing (Agentic Step)
            # If router is present and we have multiple datasets, asking router which ones to query.
            selected_datasets = datasets
            if self.router and len(datasets) > 1:
                try:
                    selected_datasets = self.router.route(query, datasets)
                    logger.info(f"Router selected {len(selected_datasets)} datasets: {[d.name for d in selected_datasets]}")
                except Exception as e:
                    # If routing fails, we should arguably check if we should fallback to all or empty?
                    # Current logic falls back to all which is safe.
                    logger.warning(f"Router failed, falling back to all datasets: {e}")

            if not selected_datasets:
                if self.callback_manager:
                    self.callback_manager.on_retrieve_end([], run_id=run_id)
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

            # 2.5. QA Indexing Handling (Swap Question -> Answer)
            # If we retrieved a "Question" document, we want to return the "Answer" (original text)
            # and use the original document ID for deduplication.
            for doc in all_documents:
                if doc.metadata.get("is_qa"):
                    # Swap content
                    question_text = doc.page_content
                    # Store question in metadata for potential debugging or UI display
                    doc.metadata["matched_question"] = question_text
                    
                    # Restore original answer as content
                    if "answer" in doc.metadata:
                        doc.page_content = doc.metadata["answer"]
                    
                    # Restore original doc ID for correct deduplication
                    if "original_doc_id" in doc.metadata:
                        doc.id = doc.metadata["original_doc_id"]
                        # Also update metadata doc_id to match (if downstream relies on it)
                        doc.metadata["document_id"] = doc.metadata["original_doc_id"]

            # 3. Reranking (Optimization)
            if self.reranker and all_documents:
                if self.callback_manager:
                    self.callback_manager.on_rerank_start(query=query, documents=all_documents, run_id=run_id)
                    
                try:
                    # Convert Document objects to SearchResult
                    from langrag.entities.search_result import SearchResult
                    
                    search_results = [
                        SearchResult(chunk=doc, score=doc.metadata.get('score', 0.0))
                        for doc in all_documents
                    ]
                    
                    reranked_results = self.reranker.rerank(
                        query, 
                        search_results, 
                        top_k=rerank_top_k or top_k
                    )
                    
                    # Update all_documents with reranked documents and their new scores
                    all_documents = []
                    for res in reranked_results:
                        doc = res.chunk
                        # Update score in metadata (important for PostProcessor)
                        doc.metadata['score'] = res.score
                        all_documents.append(doc)
                        
                except Exception as e:
                    logger.error(f"Reranking failed: {e}")
                    if self.callback_manager:
                        self.callback_manager.on_error(e, run_id=run_id)
                
                if self.callback_manager:
                    self.callback_manager.on_rerank_end(documents=all_documents, run_id=run_id)

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
            final_results = results[:final_k]
            
            if self.callback_manager:
                self.callback_manager.on_retrieve_end(final_results, run_id=run_id)
                
            return final_results
            
        except Exception as e:
            if self.callback_manager:
                self.callback_manager.on_error(e, run_id=run_id)
            raise e
