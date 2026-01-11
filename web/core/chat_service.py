"""
Chat service for the Web layer.

This module handles RAG-powered chat functionality including:
- Context retrieval from knowledge bases
- Prompt construction with retrieved context
- LLM response generation (sync and streaming)
- Agentic routing for multi-KB selection

Design Decisions:
-----------------
1. **Retrieval-Augmented Generation (RAG)**: The chat service retrieves relevant
   documents from knowledge bases and includes them as context for the LLM.
   This grounds the LLM's responses in your specific data.

2. **Streaming Support**: Both synchronous and streaming responses are supported.
   Streaming provides better UX for long responses by showing incremental output.

3. **Agentic Routing**: When multiple KBs are available, the Router component
   can intelligently select which KBs are most relevant for a given query.

4. **Source Attribution**: Every response includes the source documents used,
   enabling users to verify information and explore further.

Example Usage:
--------------
    service = ChatService(
        llm_client=async_openai_client,
        llm_config={"model": "gpt-4", "temperature": 0.7},
        retrieval_service=retrieval_service,
        router=router
    )

    # Synchronous chat
    response = await service.chat(
        kb_stores={"kb1": store1},
        query="What is machine learning?",
        history=[]
    )
    print(response["answer"])

    # Streaming chat
    async for chunk in await service.chat(..., stream=True):
        print(chunk, end="")
"""

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from langrag import BaseVector, Dataset
from langrag import Document as LangRAGDocument
from langrag.retrieval.router.base import BaseRouter
from web.core.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)


class ChatService:
    """
    Service for RAG-powered conversational AI.

    This service orchestrates the complete RAG chat pipeline:
    1. Route query to relevant knowledge bases (optional)
    2. Retrieve relevant documents
    3. Construct prompt with context
    4. Generate response using LLM

    The service supports both synchronous and streaming responses,
    making it suitable for various UI requirements.

    Attributes:
        llm_client: AsyncOpenAI client for LLM calls
        llm_config: Configuration dict (model, temperature, max_tokens)
        retrieval_service: Service for document retrieval
        router: Optional router for multi-KB selection
        kb_names: Mapping of KB IDs to human-readable names
    """

    def __init__(
        self,
        llm_client: Any,  # AsyncOpenAI
        llm_config: dict[str, Any],
        retrieval_service: RetrievalService,
        router: BaseRouter | None = None,
        kb_names: dict[str, str] | None = None
    ):
        """
        Initialize the chat service.

        Args:
            llm_client: AsyncOpenAI client for making LLM API calls.
                       Must support chat.completions.create().
            llm_config: LLM configuration containing:
                       - model: Model identifier (e.g., "gpt-4")
                       - temperature: Sampling temperature (0.0-2.0)
                       - max_tokens: Maximum response tokens
            retrieval_service: Service for retrieving documents from KBs.
            router: Optional Agentic router for KB selection.
                   When provided, intelligently filters KBs for each query.
            kb_names: Optional mapping of KB IDs to display names.
                     Used for source attribution in responses.
        """
        self.llm_client = llm_client
        self.llm_config = llm_config
        self.retrieval_service = retrieval_service
        self.router = router
        self.kb_names = kb_names or {}

    def _route_kbs(
        self,
        query: str,
        kb_stores: dict[str, BaseVector]
    ) -> list[str]:
        """
        Use the Agentic Router to select relevant KBs for a query.

        The router uses an LLM to analyze the query and determine which
        knowledge bases are most likely to contain relevant information.

        This is particularly useful when:
        - You have multiple specialized KBs (e.g., HR, Engineering, Sales)
        - Some KBs are for general knowledge, others for specific domains
        - You want to reduce irrelevant context in the prompt

        Args:
            query: The user's query
            kb_stores: All available KB stores

        Returns:
            List of selected KB IDs (may be subset of input)
        """
        if not self.router or len(kb_stores) <= 1:
            return list(kb_stores.keys())

        try:
            # Build candidate datasets with descriptions
            candidate_datasets = []
            for kb_id, store in kb_stores.items():
                # Generate description based on store type
                store_type = store.__class__.__name__
                if store_type == "WebVector":
                    description = (
                        "Internet Search Engine. Use for current events, "
                        "news, public information, or release dates."
                    )
                elif "SeekDB" in store_type or "Chroma" in store_type:
                    description = (
                        f"Local Private Knowledge Base ({kb_id}). Use for "
                        "internal documents, company policy, or domain knowledge."
                    )
                else:
                    description = f"Knowledge base: {kb_id}"

                candidate_datasets.append(
                    Dataset(
                        name=kb_id,
                        collection_name=kb_id,
                        description=description
                    )
                )

            # Route to select relevant KBs
            selected_datasets = self.router.route(query, candidate_datasets)
            selected_ids = [d.name for d in selected_datasets]

            if len(selected_ids) < len(kb_stores):
                logger.info(
                    f"[Agentic RAG] Router filtered KBs: "
                    f"{list(kb_stores.keys())} -> {selected_ids}"
                )
            
            return selected_ids

        except Exception as e:
            logger.error(f"Router failed: {e}")

        # Fallback to all KBs
        return list(kb_stores.keys())

    def _build_prompt(
        self,
        query: str,
        results: list[LangRAGDocument],
        history: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """
        Construct the chat messages with retrieved context.

        The prompt structure:
        1. System message with context (if results available)
        2. Recent conversation history (last 4 messages)
        3. Current user query

        Context formatting:
        - Each source is numbered for reference
        - KB ID is included for source attribution
        - Clear separation between sources

        Args:
            query: The current user query
            results: Retrieved documents for context
            history: Previous conversation messages

        Returns:
            List of message dicts for the LLM API
        """
        messages = []

        # Build system prompt
        if results:
            # Format context from retrieved documents
            context_parts = []
            for i, doc in enumerate(results):
                kb_id = doc.metadata.get('kb_id', '?')
                context_parts.append(
                    f"--- Source {i+1} (KB: {kb_id}) ---\n{doc.page_content}"
                )
            context_text = "\n\n".join(context_parts)

            system_prompt = f"""You are a helpful AI assistant capable of answering questions based on the provided context.
Use the context below to answer the user's question clearly and accurately.
If the answer is not in the context, use your own knowledge but prefer the context.

Context:
{context_text}
"""
        else:
            system_prompt = "You are a helpful AI assistant."

        messages.append({"role": "system", "content": system_prompt})

        # Add recent history (limit to last 4 to avoid token overflow)
        if history:
            messages.extend(history[-4:])

        # Add current query
        messages.append({"role": "user", "content": query})

        return messages

    def _build_sources_list(
        self,
        results: list[LangRAGDocument]
    ) -> list[dict[str, Any]]:
        """
        Build the sources list for response attribution.

        Each source includes:
        - content: The retrieved text
        - score: Relevance score from retrieval
        - source: Original file/document source
        - kb_id: Knowledge base identifier
        - kb_name: Human-readable KB name
        - Additional metadata (title, link, type)

        Args:
            results: Retrieved documents

        Returns:
            List of source dicts for the response
        """
        return [
            {
                "content": doc.page_content,
                "score": doc.metadata.get('score', 0),
                "source": doc.metadata.get('source', 'unknown'),
                "kb_id": doc.metadata.get('kb_id', 'unknown'),
                "kb_name": self.kb_names.get(
                    doc.metadata.get('kb_id', ''),
                    'Unknown'
                ),
                "title": doc.metadata.get('title'),
                "link": doc.metadata.get('link'),
                "type": doc.metadata.get('type')
            }
            for doc in results
        ]

    async def _generate_streaming(
        self,
        messages: list[dict[str, str]],
        sources_list: list[dict[str, Any]]
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response from LLM.

        The streaming format uses JSON Lines (JSONL):
        1. First line: {"type": "sources", "data": [...]}
        2. Content lines: {"type": "content", "data": "..."}
        3. Error line (if any): {"type": "error", "data": "..."}

        This format allows the frontend to:
        - Display sources immediately
        - Stream content as it arrives
        - Handle errors gracefully

        Args:
            messages: Chat messages for the LLM
            sources_list: Retrieved sources to include

        Yields:
            JSONL strings for each chunk
        """
        # Yield sources first
        yield json.dumps({"type": "sources", "data": sources_list}) + "\n"

        try:
            stream_resp = await self.llm_client.chat.completions.create(
                model=self.llm_config["model"],
                messages=messages,
                temperature=self.llm_config["temperature"],
                max_tokens=self.llm_config["max_tokens"],
                stream=True
            )

            async for chunk in stream_resp:
                content = chunk.choices[0].delta.content
                if content:
                    yield json.dumps({"type": "content", "data": content}) + "\n"

        except Exception as e:
            logger.error(f"LLM streaming failed: {e}")
            yield json.dumps({"type": "error", "data": str(e)}) + "\n"

    async def chat(
        self,
        kb_stores: dict[str, BaseVector],
        query: str,
        history: list[dict[str, str]] | None = None,
        stream: bool = False
    ) -> dict[str, Any] | AsyncGenerator[str, None]:
        """
        Execute RAG chat with optional streaming.

        This is the main entry point for RAG-powered chat. It:
        1. Routes to relevant KBs (if router configured)
        2. Retrieves context from selected KBs
        3. Builds the prompt with context
        4. Generates LLM response

        Args:
            kb_stores: Dict mapping KB IDs to vector stores.
                      Pass empty dict for chat without RAG context.
            query: The user's question or message.
            history: Previous conversation messages.
                    Each message: {"role": "user"|"assistant", "content": "..."}
            stream: If True, returns an async generator yielding chunks.
                   If False, returns complete response dict.

        Returns:
            If stream=False:
                {"answer": "...", "sources": [...]}
            If stream=True:
                AsyncGenerator yielding JSONL strings

        Raises:
            ValueError: If LLM client is not configured

        Example:
            # Synchronous usage
            response = await service.chat(stores, "What is X?")
            print(response["answer"])

            # Streaming usage
            async for chunk in await service.chat(stores, "What is X?", stream=True):
                data = json.loads(chunk)
                if data["type"] == "content":
                    print(data["data"], end="")
        """
        if not self.llm_client:
            raise ValueError("LLM is not configured")

        history = history or []

        # Step 1: Retrieve context
        if not kb_stores:
            # Chat without RAG context
            results = []
            search_type = "none"
        else:
            # Route to select relevant KBs
            selected_kb_ids = self._route_kbs(query, kb_stores)
            selected_stores = {
                kb_id: store
                for kb_id, store in kb_stores.items()
                if kb_id in selected_kb_ids
            }

            # Retrieve from selected KBs
            # Retrieve from selected KBs
            import asyncio
            results, search_type = await asyncio.to_thread(
                self.retrieval_service.multi_search,
                stores=selected_stores,
                query=query,
                top_k=5,
                rewrite=True  # Apply query rewriting
            )

        # Step 2: Build prompt
        messages = self._build_prompt(query, results, history)

        # Step 3: Build sources list
        sources_list = self._build_sources_list(results)

        logger.info(
            f"Chat: model={self.llm_config['model']}, "
            f"sources={len(sources_list)}, stream={stream}"
        )

        # Step 4: Generate response
        if stream:
            return self._generate_streaming(messages, sources_list)
        else:
            try:
                response = await self.llm_client.chat.completions.create(
                    model=self.llm_config["model"],
                    messages=messages,
                    temperature=self.llm_config["temperature"],
                    max_tokens=self.llm_config["max_tokens"]
                )

                answer = response.choices[0].message.content

                return {
                    "answer": answer,
                    "sources": sources_list
                }

            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                raise
