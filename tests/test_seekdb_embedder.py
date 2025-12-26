"""Test SeekDB embedder."""

import pytest
from langrag import EmbedderFactory


class TestSeekDBEmbedder:
    """Test SeekDB embedder functionality."""

    @staticmethod
    def seekdb_available():
        """Check if pyseekdb is available."""
        try:
            import pyseekdb
            return True
        except ImportError:
            return False

    def test_seekdb_embedder_registration(self):
        """SeekDB embedder should be registered if pyseekdb is available."""
        available = EmbedderFactory.list_types()

        # Mock should always be available
        assert "mock" in available

        # SeekDB may or may not be available
        if self.seekdb_available():
            assert "seekdb" in available
        else:
            assert "seekdb" not in available

    @pytest.mark.skipif(
        not seekdb_available.__func__(),
        reason="SeekDB embedder not available (pyseekdb not installed)"
    )
    def test_seekdb_embedder_creation(self):
        """Test creating SeekDB embedder."""
        embedder = EmbedderFactory.create("seekdb")

        # Check dimension
        assert embedder.dimension == 384  # all-MiniLM-L6-v2

    @pytest.mark.skipif(
        not seekdb_available.__func__(),
        reason="SeekDB embedder not available (pyseekdb not installed)"
    )
    def test_seekdb_embedder_embed(self):
        """Test embedding generation with SeekDB."""
        embedder = EmbedderFactory.create("seekdb")

        # Generate embeddings
        texts = [
            "Hello world",
            "Machine learning is fascinating",
            "Python programming language"
        ]

        embeddings = embedder.embed(texts)

        # Verify results
        assert len(embeddings) == len(texts)
        assert all(len(emb) == 384 for emb in embeddings)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(all(isinstance(v, float) for v in emb) for emb in embeddings)

        # Embeddings should be different for different texts
        assert embeddings[0] != embeddings[1]
        assert embeddings[1] != embeddings[2]

    @pytest.mark.skipif(
        not seekdb_available.__func__(),
        reason="SeekDB embedder not available (pyseekdb not installed)"
    )
    def test_seekdb_embedder_deterministic(self):
        """Test that SeekDB embeddings are deterministic."""
        embedder = EmbedderFactory.create("seekdb")

        text = ["Test text for determinism"]

        # Generate twice
        embedding1 = embedder.embed(text)
        embedding2 = embedder.embed(text)

        # Should be identical
        assert embedding1 == embedding2

    @pytest.mark.skipif(
        not seekdb_available.__func__(),
        reason="SeekDB embedder not available (pyseekdb not installed)"
    )
    def test_seekdb_embedder_empty_input(self):
        """Test that empty input raises ValueError."""
        embedder = EmbedderFactory.create("seekdb")

        with pytest.raises(ValueError, match="texts cannot be empty"):
            embedder.embed([])

    @pytest.mark.skipif(
        not seekdb_available.__func__(),
        reason="SeekDB embedder not available (pyseekdb not installed)"
    )
    def test_seekdb_embedder_batch(self):
        """Test batch embedding generation."""
        embedder = EmbedderFactory.create("seekdb")

        # Large batch
        texts = [f"Text number {i}" for i in range(100)]
        embeddings = embedder.embed(texts)

        assert len(embeddings) == 100
        assert all(len(emb) == 384 for emb in embeddings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
