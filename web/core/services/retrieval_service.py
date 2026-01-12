"""
Retrieval service for the Web layer.

This module acts as a facade around the standard LangRAG RetrievalWorkflow.
It simplifies the orchestration of RAG retrieval.
"""

import logging
from typing import Any

from langrag import BaseEmbedder, BaseVector, Dataset
from langrag import Document as LangRAGDocument
from langrag.cache import SemanticCache
from langrag.datasource.kv.base import BaseKVStore
from langrag.retrieval.rerank.base import BaseReranker
from langrag.retrieval.rewriter.base import BaseRewriter
from langrag.retrieval.workflow import RetrievalWorkflow
from langrag.retrieval.config import WorkflowConfig

logger = logging.getLogger(__name__)

class RetrievalService:
    """
    Facade for the standard LangRAG RetrievalWorkflow.
    """

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        reranker: BaseReranker | None = None,
        rewriter: BaseRewriter | None = None,
        kv_store: BaseKVStore | None = None,
        cache: SemanticCache | None = None,
        vector_manager: Any = None
    ):
        """
        Initialize the retrieval service (facade).
        """
        self.embedder = embedder
        self.reranker = reranker
        self.rewriter = rewriter
        self.kv_store = kv_store
        self.cache = cache
        self.vector_manager = vector_manager
        
        # Initialize the underlying workflow
        self.workflow = RetrievalWorkflow(
            router=None, # Router is not passed in init currently in web layer? wait, checks rag_kernel.py
            embedder=embedder,
            reranker=reranker,
            rewriter=rewriter,
            vector_manager=vector_manager,
            cache=cache,
            config=WorkflowConfig(
                max_workers=5
            )
        )

    def search(
        self,
        store: BaseVector,
        query: str,
        top_k: int = 5,
        rewrite: bool = True,
        search_mode: str | None = None,
        use_rerank: bool | None = None,
        score_threshold: float = 0.0,
        rerank_top_k: int | None = None
    ) -> tuple[list[LangRAGDocument], str, str | None]:
        """
        Execute single store search using workflow.
        
        Args:
            store: 向量存储实例
            query: 查询文本
            top_k: 返回结果数量
            rewrite: 是否启用查询重写
            search_mode: 搜索模式 ("hybrid", "vector", "keyword")
            use_rerank: 是否启用重排序
            score_threshold: 分数阈值
            rerank_top_k: 重排序后返回的数量
        """
        try:
           # The workflow is designed for multiple datasets, but works for one.
           # However, the workflow interface returns list[RetrievalContext].
           # We need to map it back to what existing web callers expect:
           # (results list[LangRAGDocument], search type string, rewritten query or None)
           
           # 确定实际的 rerank_top_k
           effective_rerank_top_k = rerank_top_k if rerank_top_k else top_k
           
           results_context = self.workflow.retrieve(
               query=query,
               datasets=[store.dataset],
               top_k=top_k,
               score_threshold=score_threshold,
               rerank_top_k=effective_rerank_top_k
           )
           
           # Convert back to Documents
           docs = []
           for ctx in results_context:
               doc = LangRAGDocument(
                   page_content=ctx.content,
                   metadata=ctx.metadata
               )
               # Ensure ID consistency
               if hasattr(ctx, 'document_id'):
                    doc.metadata['document_id'] = ctx.document_id
                    
               doc.metadata['score'] = ctx.score
               docs.append(doc)
           
           # 确定搜索类型
           search_type = search_mode or "hybrid"
           if self.reranker and use_rerank:
               search_type = f"{search_type}+rerank"
               
           return docs, search_type, None
           
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return [], "error", None

    def multi_search(
        self,
        stores: dict[str, BaseVector],
        query: str,
        top_k: int = 5,
        rewrite: bool = True,
        use_cache: bool = True
    ) -> tuple[list[LangRAGDocument], str]:
        """
        Execute multi store search using workflow.
        """
        try:
           datasets = [s.dataset for s in stores.values()]
           
           results_context = self.workflow.retrieve(
               query=query,
               datasets=datasets,
               top_k=top_k
           )
           
           # Convert back to Documents
           docs = []
           for ctx in results_context:
               doc = LangRAGDocument(
                   page_content=ctx.content,
                   metadata=ctx.metadata
               )
               if hasattr(ctx, 'document_id'):
                    doc.metadata['document_id'] = ctx.document_id
               doc.metadata['score'] = ctx.score
               docs.append(doc)
               
           return docs, "semantic_search"
           
        except Exception as e:
            logger.error(f"Multi-search failed: {e}")
            return [], "error"
