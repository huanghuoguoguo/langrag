import logging
from typing import List
from langrag.datasource.web.base import BaseWebSearchProvider, WebSearchResult

logger = logging.getLogger(__name__)

class DuckDuckGoSearchProvider(BaseWebSearchProvider):
    """
    DuckDuckGo Search Provider.
    Does NOT require API Keys.
    """
    
    def __init__(self, region: str = "us-en", safesearch: str = "moderate", max_retries: int = 3):
        self.region = region
        self.safesearch = safesearch
        self.max_retries = max_retries
        
    def search(self, query: str, top_k: int = 5, **kwargs) -> List[WebSearchResult]:
        try:
            from ddgs import DDGS
        except ImportError:
            logger.error("ddgs not installed. Please install with `uv add ddgs`")
            return []
            
        results = []
        try:
            with DDGS() as ddgs:
                # DDGS keywords defaults to text search
                ddgs_gen = ddgs.text(
                    query,
                    region=self.region,
                    safesearch=self.safesearch,
                    max_results=top_k
                )
                
                for r in ddgs_gen:
                    results.append(WebSearchResult(
                        title=r.get("title", ""),
                        link=r.get("href", ""),
                        snippet=r.get("body", ""),
                        source="duckduckgo"
                    ))
                    
            return results
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []
