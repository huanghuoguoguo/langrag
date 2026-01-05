import httpx
import logging
from typing import List, Optional
from langrag.datasource.web.base import BaseWebSearchProvider, WebSearchResult

logger = logging.getLogger(__name__)

class GoogleSearchProvider(BaseWebSearchProvider):
    """
    Google Custom Search JSON API Provider.
    Requires GOOGLE_API_KEY and GOOGLE_CSE_ID.
    """
    
    def __init__(self, api_key: str, cse_id: str, timeout: float = 10.0):
        self.api_key = api_key
        self.cse_id = cse_id
        self.timeout = timeout
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
    def search(self, query: str, top_k: int = 5, **kwargs) -> List[WebSearchResult]:
        if not self.api_key or not self.cse_id:
             logger.warning("Google Search credentials missing. Returning empty.")
             return []

        params = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": query,
            "num": min(top_k, 10) # Google API max is 10
        }
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
            items = data.get("items", [])
            results = []
            
            for item in items:
                results.append(WebSearchResult(
                    title=item.get("title", ""),
                    link=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source="google"
                ))
            
            # If user requested more than 10 (rare), we'd need paging, 
            # but for RAG context, top 10 is usually sufficient.
            return results[:top_k]
            
        except httpx.HTTPError as e:
            logger.error(f"Google Search API error: {e}") 
            return []
        except Exception as e:
            logger.error(f"Google Search failed: {e}")
            return []
