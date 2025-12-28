"""Test utilities - Test assertions helpers."""

from langrag.entities.document import Document
from langrag.entities.search_result import SearchResult

def assert_search_results_valid(results, min_results=0, max_results=float("inf")):
    assert isinstance(results, list), "Results must be a list"
    assert min_results <= len(results) <= max_results

    for result in results:
        # SearchResult might wrap a Document or Chunk
        # In new architecture, SearchResult.chunk is a Document
        if hasattr(result, "chunk"):
            assert isinstance(result.chunk, Document)
        
        # Or if result IS a document (some services return list[Document])
        if isinstance(result, Document):
             pass
        elif isinstance(result, SearchResult):
             assert 0.0 <= result.score <= 1.0


def assert_scores_descending(results):
    # Depending on result type, extract score
    scores = []
    for r in results:
        if isinstance(r, SearchResult):
            scores.append(r.score)
        elif isinstance(r, Document):
            # Try to get score from metadata
            scores.append(r.metadata.get("score", 0))
            
    assert scores == sorted(scores, reverse=True)


def assert_file_exists(path):
    from pathlib import Path
    path = Path(path)
    assert path.exists()
    assert path.is_file()
