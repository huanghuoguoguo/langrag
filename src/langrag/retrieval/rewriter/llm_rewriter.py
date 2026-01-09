from loguru import logger

from langrag.llm.base import BaseLLM

from .base import BaseRewriter

REWRITE_PROMPT = """
You are a search query optimizer. Rewrite the following user query to be more effective for semantic search retrieval. 
Do not change the intent. Only return the rewritten query text.

Query: {query}
Rewritten:
"""

class LLMRewriter(BaseRewriter):
    """
    Rewriter using LLM.
    """

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def rewrite(self, query: str) -> str:
        prompt = REWRITE_PROMPT.format(query=query)
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            rewritten = response.strip()
            # Safety check: if empty or too different, maybe keep original?
            # For now, just return it.
            return rewritten
        except Exception as e:
            logger.error(f"Query rewrite failed: {e}")
            return query
