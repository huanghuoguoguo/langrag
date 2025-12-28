"""SearchResult entity representing a retrieval result."""

from pydantic import BaseModel, Field

from .document import Document


class SearchResult(BaseModel):
    """Represents a search result with relevance score.

    Attributes:
        chunk: The retrieved chunk
        score: Relevance score in range [0, 1]
    """

    chunk: Document
    score: float = Field(..., ge=0.0, le=1.0)

    model_config = {
        "frozen": True,  # Results are immutable
    }

    def __lt__(self, other: "SearchResult") -> bool:
        """Enable sorting by score (descending)."""
        return self.score > other.score
