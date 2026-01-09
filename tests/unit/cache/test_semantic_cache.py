"""
Tests for Semantic Cache functionality.

These tests verify:
1. Basic cache operations (get, set, delete, clear)
2. Semantic similarity matching
3. TTL expiration
4. LRU eviction
5. Thread safety
6. Cache statistics
"""

import time
import threading

import pytest

from langrag.cache import SemanticCache, CacheEntry, cosine_similarity
from langrag.entities.document import Document


class TestCosinesSimilarity:
    """Tests for the cosine_similarity function."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity of 1.0."""
        vec = [0.1, 0.2, 0.3, 0.4]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity of 0.0."""
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        assert cosine_similarity(vec1, vec2) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity of -1.0."""
        vec1 = [1.0, 0.0]
        vec2 = [-1.0, 0.0]
        assert cosine_similarity(vec1, vec2) == pytest.approx(-1.0)

    def test_similar_vectors(self):
        """Similar vectors should have high similarity."""
        vec1 = [0.1, 0.2, 0.3, 0.4]
        vec2 = [0.11, 0.21, 0.31, 0.41]
        sim = cosine_similarity(vec1, vec2)
        assert sim > 0.99

    def test_different_lengths(self):
        """Vectors of different lengths should return 0.0."""
        vec1 = [0.1, 0.2, 0.3]
        vec2 = [0.1, 0.2, 0.3, 0.4]
        assert cosine_similarity(vec1, vec2) == 0.0

    def test_zero_vector(self):
        """Zero vector should return 0.0 similarity."""
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [0.1, 0.2, 0.3]
        assert cosine_similarity(vec1, vec2) == 0.0


class TestSemanticCache:
    """Tests for SemanticCache class."""

    @pytest.fixture
    def cache(self):
        """Create a basic semantic cache."""
        return SemanticCache(
            similarity_threshold=0.95,
            ttl_seconds=3600,
            max_size=100
        )

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for caching."""
        return [
            Document(id="doc1", page_content="Machine learning basics"),
            Document(id="doc2", page_content="Neural networks explained")
        ]

    @pytest.fixture
    def sample_embedding(self):
        """Create a sample embedding vector."""
        return [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_set_and_get_exact_match(self, cache, sample_documents, sample_embedding):
        """Test storing and retrieving with exact key match."""
        cache.set_with_embedding(
            query="What is ML?",
            embedding=sample_embedding,
            results=sample_documents,
            metadata={"search_type": "vector"}
        )

        entry = cache.get("What is ML?")
        assert entry is not None
        assert entry.query == "What is ML?"
        assert len(entry.results) == 2
        assert entry.metadata["search_type"] == "vector"

    def test_get_by_similarity_hit(self, cache, sample_documents):
        """Test semantic similarity lookup with a similar query."""
        embedding1 = [0.1, 0.2, 0.3, 0.4, 0.5]
        embedding2 = [0.101, 0.201, 0.301, 0.401, 0.501]  # Very similar

        cache.set_with_embedding(
            query="What is machine learning?",
            embedding=embedding1,
            results=sample_documents,
            metadata={"search_type": "vector"}
        )

        hit = cache.get_by_similarity(embedding2)
        assert hit is not None
        assert hit.query == "What is machine learning?"

    def test_get_by_similarity_miss(self, cache, sample_documents):
        """Test semantic similarity lookup with a different query."""
        embedding1 = [0.1, 0.2, 0.3, 0.4, 0.5]
        embedding2 = [0.9, 0.8, 0.7, 0.6, 0.5]  # Very different

        cache.set_with_embedding(
            query="What is machine learning?",
            embedding=embedding1,
            results=sample_documents,
            metadata={"search_type": "vector"}
        )

        hit = cache.get_by_similarity(embedding2)
        assert hit is None

    def test_context_key_filtering(self, cache, sample_documents):
        """Test that context_key properly scopes cache lookups."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Cache with context_key
        cache.set_with_embedding(
            query="What is ML?",
            embedding=embedding,
            results=sample_documents,
            metadata={"search_type": "vector", "context_key": "kb1"}
        )

        # Should find with correct context
        hit = cache.get_by_similarity(embedding, context_key="kb1")
        assert hit is not None

        # Should not find with different context
        hit = cache.get_by_similarity(embedding, context_key="kb2")
        assert hit is None

    def test_delete(self, cache, sample_documents, sample_embedding):
        """Test deleting a cache entry."""
        cache.set_with_embedding(
            query="Test query",
            embedding=sample_embedding,
            results=sample_documents,
            metadata={}
        )

        assert cache.get("Test query") is not None
        cache.delete("Test query")
        assert cache.get("Test query") is None

    def test_clear(self, cache, sample_documents, sample_embedding):
        """Test clearing all cache entries."""
        for i in range(5):
            cache.set_with_embedding(
                query=f"Query {i}",
                embedding=[0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i, 0.5 * i],
                results=sample_documents,
                metadata={}
            )

        assert cache.size == 5
        cache.clear()
        assert cache.size == 0

    def test_size_property(self, cache, sample_documents, sample_embedding):
        """Test the size property."""
        assert cache.size == 0

        cache.set_with_embedding(
            query="Query 1",
            embedding=sample_embedding,
            results=sample_documents,
            metadata={}
        )
        assert cache.size == 1

    def test_stats(self, cache, sample_documents):
        """Test cache statistics."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Store entry
        cache.set_with_embedding(
            query="Test query",
            embedding=embedding,
            results=sample_documents,
            metadata={}
        )

        # Hit
        cache.get_by_similarity(embedding)

        # Miss
        cache.get_by_similarity([0.9, 0.8, 0.7, 0.6, 0.5])

        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(0.5)
        assert stats["size"] == 1


class TestSemanticCacheTTL:
    """Tests for TTL (time-to-live) functionality."""

    def test_ttl_expiration(self):
        """Test that entries expire after TTL."""
        cache = SemanticCache(
            similarity_threshold=0.95,
            ttl_seconds=1,  # 1 second TTL
            max_size=100
        )

        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        cache.set_with_embedding(
            query="Test query",
            embedding=embedding,
            results=[Document(id="doc1", page_content="test")],
            metadata={}
        )

        # Should be present immediately
        assert cache.get("Test query") is not None

        # Wait for TTL
        time.sleep(1.5)

        # Should be expired
        assert cache.get("Test query") is None

    def test_no_expiration_with_zero_ttl(self):
        """Test that zero TTL means no expiration."""
        cache = SemanticCache(
            similarity_threshold=0.95,
            ttl_seconds=0,  # No expiration
            max_size=100
        )

        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        cache.set_with_embedding(
            query="Test query",
            embedding=embedding,
            results=[Document(id="doc1", page_content="test")],
            metadata={}
        )

        # Should still be present after some time
        time.sleep(0.5)
        assert cache.get("Test query") is not None


class TestSemanticCacheLRU:
    """Tests for LRU (Least Recently Used) eviction."""

    def test_lru_eviction(self):
        """Test that oldest entries are evicted when max_size is reached."""
        cache = SemanticCache(
            similarity_threshold=0.95,
            ttl_seconds=3600,
            max_size=3
        )

        for i in range(5):
            cache.set_with_embedding(
                query=f"Query {i}",
                embedding=[0.1 * (i + 1)] * 5,
                results=[Document(id=f"doc{i}", page_content=f"content{i}")],
                metadata={}
            )

        # Should only have 3 entries
        assert cache.size == 3

        # First two should be evicted
        assert cache.get("Query 0") is None
        assert cache.get("Query 1") is None

        # Last three should be present
        assert cache.get("Query 2") is not None
        assert cache.get("Query 3") is not None
        assert cache.get("Query 4") is not None


class TestSemanticCacheThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_writes(self):
        """Test concurrent write operations."""
        cache = SemanticCache(
            similarity_threshold=0.95,
            ttl_seconds=3600,
            max_size=1000
        )

        def write_entries(start: int, count: int):
            for i in range(start, start + count):
                cache.set_with_embedding(
                    query=f"Query {i}",
                    embedding=[0.1 * i] * 5,
                    results=[Document(id=f"doc{i}", page_content=f"content{i}")],
                    metadata={}
                )

        threads = [
            threading.Thread(target=write_entries, args=(i * 100, 100))
            for i in range(4)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have 400 entries
        assert cache.size == 400

    def test_concurrent_reads_and_writes(self):
        """Test concurrent read and write operations."""
        cache = SemanticCache(
            similarity_threshold=0.95,
            ttl_seconds=3600,
            max_size=1000
        )

        # Pre-populate some entries
        for i in range(100):
            cache.set_with_embedding(
                query=f"Query {i}",
                embedding=[0.1 * i] * 5,
                results=[Document(id=f"doc{i}", page_content=f"content{i}")],
                metadata={}
            )

        read_count = [0]
        write_count = [0]

        def read_entries():
            for i in range(100):
                if cache.get(f"Query {i}"):
                    read_count[0] += 1

        def write_entries():
            for i in range(100, 200):
                cache.set_with_embedding(
                    query=f"Query {i}",
                    embedding=[0.1 * i] * 5,
                    results=[Document(id=f"doc{i}", page_content=f"content{i}")],
                    metadata={}
                )
                write_count[0] += 1

        threads = [
            threading.Thread(target=read_entries),
            threading.Thread(target=write_entries)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All reads and writes should complete
        assert read_count[0] == 100
        assert write_count[0] == 100


class TestSemanticCacheValidation:
    """Tests for input validation."""

    def test_invalid_similarity_threshold_high(self):
        """Test that similarity_threshold > 1.0 raises ValueError."""
        with pytest.raises(ValueError):
            SemanticCache(similarity_threshold=1.5)

    def test_invalid_similarity_threshold_low(self):
        """Test that similarity_threshold < 0.0 raises ValueError."""
        with pytest.raises(ValueError):
            SemanticCache(similarity_threshold=-0.1)

    def test_valid_similarity_threshold_boundary(self):
        """Test that boundary values are accepted."""
        cache1 = SemanticCache(similarity_threshold=0.0)
        cache2 = SemanticCache(similarity_threshold=1.0)
        assert cache1.similarity_threshold == 0.0
        assert cache2.similarity_threshold == 1.0
