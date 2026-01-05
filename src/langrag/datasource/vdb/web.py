import logging
import os
from typing import List, Optional

from langrag.datasource.vdb.base import BaseVector
from langrag.entities.document import Document
from langrag.entities.dataset import Dataset
from langrag.datasource.web.base import BaseWebSearchProvider
from langrag.datasource.web.providers.google import GoogleSearchProvider
from langrag.datasource.web.providers.ddg import DuckDuckGoSearchProvider

logger = logging.getLogger(__name__)

class WebVector(BaseVector):
    """
    Adapter that exposes Web Search as a Vector Store.
    This allows the RAG kernel to treat the internet as just another knowledge base.
    """
    
    def __init__(self, dataset: Dataset, **kwargs):
        super().__init__(dataset)
        self.provider = self._init_provider()
        
    def _init_provider(self) -> BaseWebSearchProvider:
        """Initialize the search provider based on environment variables."""
        # TODO: Move this configuration to Settings or pass via kwargs
        api_key = os.getenv("GOOGLE_API_KEY")
        cse_id = os.getenv("GOOGLE_CSE_ID")
        
        if api_key and cse_id:
            logger.info("Initializing GoogleSearchProvider for WebVector")
            return GoogleSearchProvider(api_key=api_key, cse_id=cse_id)
        
        # Fallback to Bing (China accessible)
        try:
            logger.info("Initializing BingSearchProvider for WebVector")
            from langrag.datasource.web.providers.bing import BingSearchProvider
            return BingSearchProvider()
        except ImportError:
            # Fallback to DuckDuckGo if Bing fails to load (unlikely but safe)
            try:
                logger.info("Bing provider failed, falling back to DuckDuckGo")
                return DuckDuckGoSearchProvider()
            except ImportError:
                logger.error("Web search providers failed to load.")
                raise ImportError("No valid web search provider available.")



    def create(self, texts: list[Document], **kwargs) -> None:
        """Web Search is read-only."""
        pass

    def add_texts(self, texts: list[Document], **kwargs) -> None:
        """Web Search is read-only."""
        pass

    def search(
        self, 
        query: str, 
        query_vector: list[float] | None, 
        top_k: int = 4, 
        **kwargs
    ) -> list[Document]:
        """
        Perform web search and convert results to Documents.
        
        Args:
            query: Search query
            query_vector: Ignored for web search (unless we did hybrid visual search later)
            top_k: Number of results
        """
        logger.info(f"WebVector searching for: '{query}' (top_k={top_k})")
        
        # Determine actual query to use (keyword vs semantic intent)
        # For now, just use the raw query string. 
        # Ideally, we might want to extract keywords if the query is very long.
        
        results = self.provider.search(query, top_k=top_k)
        
        documents = []
        for res in results:
            # Convert WebSearchResult to Document
            doc = Document(
                page_content=f"Title: {res.title}\nSource: {res.source}\nLink: {res.link}\n\n{res.snippet}",
                metadata={
                    "source": res.link,
                    "title": res.title,
                    "type": "web_search",
                    "link": res.link,
                    "kb_id": self.dataset.id,
                    "score": res.score or 0.5 # Default score if not provided
                }
            )
            documents.append(doc)
            
        return documents

    def delete_by_ids(self, ids: list[str]) -> None:
        pass
    
    def delete(self) -> None:
        pass
