"""Query entity representing a search query."""

from pydantic import BaseModel, Field


class Query(BaseModel):
    """Represents a search query.

    Attributes:
        text: The query text
        vector: Optional pre-computed embedding vector
    """

    text: str = Field(..., min_length=1)
    vector: list[float] | None = None

    model_config = {
        "str_strip_whitespace": True,
        "frozen": True,  # Queries are immutable
    }
