
import logging
import urllib.parse

import httpx
from bs4 import BeautifulSoup

from langrag.datasource.web.base import BaseWebSearchProvider, WebSearchResult

logger = logging.getLogger(__name__)

class BingSearchProvider(BaseWebSearchProvider):
    """
    Bing Web Search Provider (Scraping).
    Does NOT require API Keys.
    Works in China.
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"
        }

    def search(self, query: str, top_k: int = 5, **kwargs) -> list[WebSearchResult]:
        """
        Search Bing and parse HTML results.
        """
        results = []

        # Bing search URL
        url = f"https://cn.bing.com/search?q={urllib.parse.quote(query)}"

        try:
            with httpx.Client(headers=self.headers, follow_redirects=True, timeout=10.0) as client:
                response = client.get(url)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')

                # Bing results are usually in <li class="b_algo">
                search_results = soup.find_all('li', class_='b_algo')

                for i, item in enumerate(search_results):
                    if i >= top_k:
                        break

                    try:
                        # Extract Title and Link
                        title_tag = item.find('h2')
                        if not title_tag:
                            continue

                        link_tag = title_tag.find('a')
                        if not link_tag:
                            continue

                        title = link_tag.get_text()
                        link = link_tag.get('href')

                        # Extract Snippet
                        snippet = ""
                        caption = item.find('div', class_='b_caption')
                        if caption:
                            snippet_tag = caption.find('p')
                            if snippet_tag:
                                snippet = snippet_tag.get_text()

                        if not snippet:
                            # Fallback snippet extraction
                            snippet_tag = item.find('div', class_='b_snippet')
                            if snippet_tag:
                                snippet = snippet_tag.get_text()

                        if title and link:
                            results.append(WebSearchResult(
                                title=title,
                                link=link,
                                snippet=snippet or "No snippet available",
                                source="bing",
                                score=1.0 - (i * 0.1) # Degrading score based on rank
                            ))

                    except Exception as e:
                        logger.warning(f"Error parsing Bing result {i}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Bing search failed: {e}")

        return results
