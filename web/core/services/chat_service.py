"""
Chat service for the Web layer.

This module acts as a facade around the standard LangRAG ChatPipeline.
It simplifies the orchestration of RAG-powered chat.
"""

import logging
from typing import Any, AsyncGenerator

from langrag import BaseVector
from langrag.llm.base import BaseLLM
from langrag.pipeline.chat import ChatPipeline
from langrag.retrieval.workflow import RetrievalWorkflow
from langrag.retrieval.router.base import BaseRouter
from web.core.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

class ChatService:
    """
    Facade for the standard LangRAG ChatPipeline.
    """

    def __init__(
        self,
        llm: BaseLLM,
        llm_config: dict[str, Any],
        retrieval_service: RetrievalService, # Kept for backward compat construction, but we need RetrievalWorkflow
        router: BaseRouter | None = None,
        kb_names: dict[str, str] | None = None
    ):
        """
        Initialize.
        
        Args:
           llm: Language model.
           llm_config: Config for LLM.
           retrieval_service: Legacy web service. We need the underlying workflow.
           router: Router instance.
           kb_names: KB names map.
        """
        self.llm = llm
        self.llm_config = llm_config
        self.kb_names = kb_names or {}

        # Re-construct retrieval workflow from the web retrieval service components
        self.retrieval_workflow = RetrievalWorkflow(
            router=router,
            reranker=retrieval_service.reranker,
            rewriter=retrieval_service.rewriter,
            vector_manager=retrieval_service.vector_manager,
            vector_store_cls=None
        )
        
        # Initialize pipeline
        self.pipeline = ChatPipeline(
            llm=self.llm,
            retrieval_workflow=self.retrieval_workflow,
            debug=True
        )

    async def chat(
        self,
        kb_stores: dict[str, BaseVector],
        query: str,
        history: list[dict[str, str]] | None = None,
        stream: bool = False
    ) -> dict[str, Any] | AsyncGenerator[str, None]:
        """
        Execute chat using pipeline.
        """
        return await self.pipeline.run(
            query=query,
            kb_stores=kb_stores,
            history=history,
            stream=stream,
            top_k=5
        )
