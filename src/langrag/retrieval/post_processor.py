from typing import List, Set
from langrag.entities.document import Document

class PostProcessor:
    """
    Handles post-retrieval processing steps:
    1. Deduplication
    2. Score Filtering
    """

    def __init__(self):
        pass

    def run(self, documents: List[Document], score_threshold: float = 0.0) -> List[Document]:
        """
        Apply all post-processing steps.
        
        Args:
            documents: List of retrieved documents.
            score_threshold: Minimum score required for a document to be kept.
            
        Returns:
            Processed list of documents.
        """
        # Deduplicate first to avoid checking score for duplicates
        documents = self._deduplicate(documents)
        
        # Filter by score
        documents = self._filter_by_score(documents, score_threshold)
        
        return documents

    def _filter_by_score(self, documents: List[Document], score_threshold: float) -> List[Document]:
        """
        Filter out documents with score less than threshold.
        Assumes doc.metadata['score'] exists.
        """
        if score_threshold <= 0:
            return documents
            
        filtered = []
        for doc in documents:
            score = doc.metadata.get('score', 0.0)
            if score >= score_threshold:
                filtered.append(doc)
        return filtered

    def _deduplicate(self, documents: List[Document]) -> List[Document]:
        """
        Remove duplicate documents.
        Strategy: Use document_id (from metadata) if available, otherwise hash content.
        
        Note: When duplicates are found, the first occurrence is kept.
        If we wanted to keep the one with higher score, we should sort by score first.
        """
        seen: Set[str] = set()
        unique_docs = []
        
        # We assume documents might not be sorted by score yet, or mixed from multiple sources.
        # But usually retrieval returns them sorted. 
        # If we merged multiple sources, we should rely on the order passed in (which assumes sorted-ish) 
        # or we might want to sort here. For now, we respect input order.
        
        for doc in documents:
            # Primary key: document_id (distinct from the chunk's UUID if we are talking about unique content)
            # Actually, we likely want to deduplicate CHUNKS.
            # doc.id is the chunk ID.
            key = doc.id
            
            # If for some reason doc.id is not unique enough or we want to verify content
            if not key:
                 key = str(hash(doc.page_content))
            
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)
                
        return unique_docs
