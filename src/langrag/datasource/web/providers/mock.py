
from langrag.datasource.web.base import BaseWebSearchProvider, WebSearchResult


class MockSearchProvider(BaseWebSearchProvider):
    """
    Mock Search Provider for testing/demo purposes.
    Returns static results based on query.
    """

    def search(self, query: str, top_k: int = 5, **kwargs) -> list[WebSearchResult]:
        # Simple keywords detection to return semi-relevant mock data
        base_results = []

        if "iphone" in query.lower():
            base_results.append(WebSearchResult(
                title="iPhone 16 - Apple",
                link="https://www.apple.com/iphone-16",
                snippet="The new iPhone 16 features the A18 Bionic chip, improved battery life, and a new camera system.",
                source="mock"
            ))
            base_results.append(WebSearchResult(
                title="iPhone 16 Review: The Best Yet",
                link="https://www.theverge.com/reviews/iphone-16",
                snippet="Apple's latest flagship, the iPhone 16, refines the formula with subtle but meaningful updates.",
                source="mock"
            ))
        elif "python" in query.lower():
             base_results.append(WebSearchResult(
                title="Python 3.13 Release Notes",
                link="https://docs.python.org/3.13/whatsnew/3.13.html",
                snippet="Python 3.13 introduces the JIT compiler, removal of the GIL (experimental), and better error messages.",
                source="mock"
            ))
        else:
             base_results.append(WebSearchResult(
                title=f"Search Results for {query}",
                link="https://example.com/search",
                snippet=f"This is a mock search result for the query '{query}'. Real web search requires API keys.",
                source="mock"
            ))

        # Pad with generic results if needed to reach top_k? No, just return what we have.
        return base_results[:top_k]
