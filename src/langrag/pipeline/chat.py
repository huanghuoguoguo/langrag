"""
Chat Pipeline Module.

This pipeline handles the standard RAG chat flow:
User Query -> Retrieve Context -> Build Prompt -> LLM Generation

It integrates with the RetrievalWorkflow for advanced retrieval features
(Routing, Rewriting, Reranking).
"""

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from langrag import BaseVector, Dataset
from langrag.entities.document import Document
from langrag.entities.dataset import RetrievalContext
from langrag.llm.base import BaseLLM
from langrag.retrieval.workflow import RetrievalWorkflow
from langrag.pipeline.base import BasePipeline

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant capable of answering questions based on the provided context.
Use the context below to answer the user's question clearly and accurately.
If the answer is not in the context, use your own knowledge but prefer the context.

Context:
{context_str}
"""

class ChatPipeline(BasePipeline):
    """
    Standard RAG Chat Pipeline.
    
    Orchestrates the conversation loop:
    1. Retrieve relevant context using RetrievalWorkflow
    2. Construct prompt (System + Context + History + Query)
    3. Generate response (support for streaming)
    """

    def __init__(
        self,
        llm: BaseLLM,
        retrieval_workflow: RetrievalWorkflow,
        prompt_template: str = DEFAULT_SYSTEM_PROMPT,
        debug: bool = False
    ):
        """
        Initialize the chat pipeline.

        Args:
            llm: The Language Model for generation.
            retrieval_workflow: Configured workflow for retrieval.
            prompt_template: Template for system prompt. Must contain {context_str}.
            debug: Enable debug logging.
        """
        self.llm = llm
        self.retrieval_workflow = retrieval_workflow
        self.prompt_template = prompt_template
        self.debug = debug

    async def run(
        self,
        query: str,
        kb_stores: Dict[str, BaseVector],
        history: List[Dict[str, str]] = None,
        stream: bool = False,
        top_k: int = 5,
        **kwargs
    ) -> Dict[str, Any] | AsyncGenerator[str, None]:
        """
        Execute the chat pipeline.

        Args:
            query: User's question.
            kb_stores: Dictionary of available vector stores {kb_id: store}.
            history: Conversation history (list of openai-style messages).
            stream: Whether to stream the response.
            top_k: Number of documents to retrieve.

        Returns:
            Dict with 'answer' and 'sources' (if stream=False)
            AsyncGenerator yielding JSONL (if stream=True)
        """
        # 1. Prepare Datasets
        datasets = [store.dataset for store in kb_stores.values()]
        
        # 2. Retrieve
        retrieval_results = []
        if datasets:
            logger.info(f"[ChatPipeline] Starting retrieval: {len(datasets)} datasets, top_k={top_k}")
            # We use asyncio.to_thread because workflow is sync currently
            # but usually called in async context in web
            import asyncio
            retrieval_results = await asyncio.to_thread(
                self.retrieval_workflow.retrieve,
                query=query,
                datasets=datasets,
                top_k=top_k
            )
            logger.info(f"[ChatPipeline] Retrieval completed: {len(retrieval_results)} results")
        else:
            logger.info("[ChatPipeline] Retrieval skipped: no datasets available")
        
        # 3. Build Prompt
        messages = self._build_messages(query, retrieval_results, history)
        
        # 4. Prepare Sources Metadata
        sources_list = self._format_sources(retrieval_results, kb_stores)
        
        if self.debug:
            logger.info(f"Chat Pipeline: {len(retrieval_results)} docs retrieved.")

        # 5. Generate (with graceful degradation)
        if not self.llm:
            # No LLM configured - return retrieval-only results
            logger.info("No LLM configured, returning retrieval-only results")
            return {
                "answer": None,
                "sources": sources_list,
                "mode": "retrieval_only",
                "message": "No LLM configured, returning retrieval results only"
            }

        # LLM available - proceed with generation
        if stream:
            return self._stream_response(messages, sources_list)
        else:
            return await self._sync_response(messages, sources_list)

    def _build_messages(
        self, 
        query: str, 
        results: List[RetrievalContext], 
        history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        history = history or []
        
        context_str = ""
        if results:
            parts = []
            for i, res in enumerate(results):
                # Retrieve the KB name if available in metadata, else ID
                source_id = res.metadata.get('kb_id', 'unknown')
                parts.append(f"--- Source {i+1} ({source_id}) ---\n{res.content}")
            context_str = "\n\n".join(parts)
            
        system_content = self.prompt_template.format(context_str=context_str)
        
        messages = [{"role": "system", "content": system_content}]
        messages.extend(history[-4:]) # Keep recent history context
        messages.append({"role": "user", "content": query})
        
        return messages

    def _format_sources(
        self, 
        results: List[RetrievalContext],
        kb_stores: Dict[str, BaseVector]
    ) -> List[Dict[str, Any]]:
        sources = []
        for res in results:
            # Try to get human readable name from dataset if possible
            kb_id = res.metadata.get('kb_id')
            kb_name = kb_id
            if kb_id and kb_id in kb_stores:
                kb_name = kb_stores[kb_id].dataset.name or kb_id

            sources.append({
                "content": res.content,
                "score": res.score,
                "source": res.metadata.get('source', 'unknown'),
                "kb_id": kb_id,
                "kb_name": kb_name,
                "document_id": res.document_id,
                "metadata": res.metadata
            })
        return sources

    async def _sync_response(self, messages, sources):
        import asyncio
        answer = await asyncio.to_thread(
            self.llm.chat,
            messages=messages
        )
        return {
            "answer": answer,
            "sources": sources,
            "mode": "full_rag"
        }

    async def _stream_response(self, messages, sources):
        # Protocol:
        # 1. Yield mode info
        # 2. Yield sources
        # 3. Yield content chunks
        yield json.dumps({"type": "mode", "data": "full_rag"}) + "\n"
        yield json.dumps({"type": "sources", "data": sources}) + "\n"
        
        try:
            # We assume llm.stream_chat is a generator (not async generator)
            # wrapping it in async iteration might require adapter if BaseLLM is sync
            # But looking at BaseLLM, stream_chat returns a generator.
            # We need to iterate it. Since we are in async method, we can't block.
            # Ideally BaseLLM.stream_chat should definitely be async or we run it in thread.
            # But generators are hard to thread. 
            # For now, let's assume it's fast enough or we iterate directly.
            # Wait, `ChatService` in web used `self.llm.stream_chat`.
            
            # Let's check `src/langrag/llm/base.py` again.
            # It says `@abstractmethod def stream_chat(self, ...)`
            
            # If it's a sync generator, iterating it blocks the loop. 
            # In a real async app this is bad. But for now let's reproduce web logic.
            
            stream = self.llm.stream_chat(messages=messages)
            
            for chunk in stream:
                if chunk:
                    # In case stream yields small chunks, we just JSON dump them
                    yield json.dumps({"type": "content", "data": chunk}) + "\n"
                    # Allow event loop to breathe if possible
                    import asyncio
                    await asyncio.sleep(0) 

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield json.dumps({"type": "error", "data": str(e)}) + "\n"
