"""
Tree Agent Retriever - Agentic retrieval over PageIndex structures.

This module implements the retrieval logic for the PageIndex strategy.
It uses an LLM acting as an agent to navigate the document tree:
1. Starts with root/high-level summaries.
2. Decides which nodes to expand based on the query.
3. Drills down until sufficient information is found or leaf nodes are reached.
"""

import json
import logging
from typing import Any, List, Set

from langrag.datasource.vdb.base import BaseVector
from langrag.entities.dataset import Dataset
from langrag.entities.document import Document
from langrag.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class TreeAgentRetriever:
    """
    Agentic retriever that navigates a PageIndex tree structure.
    """

    def __init__(self, llm: BaseLLM, vector_store: BaseVector, max_steps: int = 3):
        self.llm = llm
        self.vector_store = vector_store
        self.max_steps = max_steps

    async def retrieve(self, query: str, top_k: int = 4) -> List[Document]:
        """
        Perform agentic retrieval.
        
        Args:
            query: User query.
            top_k: Number of final text chunks to return.
            
        Returns:
            List of relevant leaf documents (chunks).
        """
        # 1. Initial Search: Find entry points based on semantic similarity of summaries
        # We search for "summary" nodes specifically.
        # Note: This assumes the vector store supports filtering.
        # If not, we search everything and filter in memory.
        
        # Determine query embedding for vector search
        query_vector = await self.llm.embed_query_async(query)
        
        initial_results = await self.vector_store.search_async(
            query=query, 
            query_vector=query_vector, 
            top_k=5,
            search_type="similarity",
            # filter={"is_summary": True}  # Ideally we filter for summaries only
        )
        
        # Filter for summaries (if DB didn't do it)
        candidate_nodes = [doc for doc in initial_results if doc.metadata.get("is_summary")]
        
        # If no summaries found (maybe only raw chunks indexed?), fallback to raw search
        if not candidate_nodes:
            logger.info("No summary nodes found, falling back to standard search")
            return await self.vector_store.search_async(
                query=query, query_vector=query_vector, top_k=top_k
            )

        # 2. Agent Logic: Navigate the tree
        # We maintain a "frontier" of nodes to explore.
        # For simplicity in this first version, we use a scoring-based approach 
        # rather than a full multi-turn chat loop to reduce latency.
        
        # Strategy:
        # - Analyze candidate summaries.
        # - If a summary looks promising but serves as a parent to other nodes, fetch its children.
        # - If a node is a leaf, collect it.
        
        relevant_leaves: List[Document] = []
        visited_ids: Set[str] = set()
        
        # Queue of nodes to process: (doc, depth)
        queue = [(doc, 0) for doc in candidate_nodes]
        
        steps = 0
        while queue and steps < self.max_steps * 5: # Limit total expansions
            current_doc, depth = queue.pop(0)
            
            if current_doc.metadata.get("document_id", "") in visited_ids:
                continue
            visited_ids.add(current_doc.metadata.get("document_id", "")) # Use unique node ID logic
            
            # Check relevance using LLM (Reasoning Step)
            # "Is this section relevant to answering '{query}'?"
            is_relevant = await self._check_relevance(query, current_doc)
            if not is_relevant:
                continue
                
            # If relevant:
            if current_doc.metadata.get("is_leaf"):
                relevant_leaves.append(current_doc)
            else:
                # Expand children
                children_ids = current_doc.metadata.get("children_ids", [])
                if children_ids:
                    # In a real implementation, we need efficient "get_by_ids"
                    # For now, we simulate fetching children.
                    # This is the bottleneck: Random Access in Vector DB is slow/hard without primary key lookup.
                    # We assume vector_store has a retrieve_by_ids method.
                    if hasattr(self.vector_store, "get_by_ids"):
                         children_docs = await self.vector_store.get_by_ids(children_ids)
                         for child in children_docs:
                             queue.append((child, depth + 1))
                    else:
                        logger.warning("Vector store does not support get_by_ids, cannot expand children.")
            
            steps += 1
            if len(relevant_leaves) >= top_k:
                break
                
        # If we found leaves via navigation, return them.
        if relevant_leaves:
            return relevant_leaves[:top_k]
            
        # Fallback
        return initial_results[:top_k]

    async def _check_relevance(self, query: str, doc: Document) -> bool:
        """Use LLM to check if a document summary is relevant."""
        # Bilingual prompt for relevance checking
        prompt = f"""Determine if the section below is relevant to answering the query.
判断下面的章节摘要是否与查询相关。

Query / 查询: {query}

Section Title: {doc.metadata.get('title', 'Unknown')}
Section Summary: {doc.page_content[:500]}

Is this section likely to contain information needed to answer the query?
该章节是否可能包含回答查询所需的信息？

Reply with only: YES or NO
只需回复: YES 或 NO"""

        try:
            response = await self.llm.chat_async([{"role": "user", "content": prompt}])
            is_relevant = "YES" in response.upper()
            logger.debug(f"Relevance check for '{doc.metadata.get('title', 'N/A')}': {is_relevant}")
            return is_relevant
        except Exception as e:
            logger.warning(f"Relevance check failed: {e}, assuming relevant")
            return True  # Conservative fallback

