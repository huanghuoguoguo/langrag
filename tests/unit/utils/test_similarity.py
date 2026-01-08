"""Tests for similarity calculation utilities."""

import pytest

from langrag.utils.similarity import cosine_similarity


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors(self):
        """Identical vectors have maximum similarity."""
        vec = [1.0, 0.0, 0.0]
        result = cosine_similarity(vec, vec)
        assert result == 1.0

    def test_opposite_vectors(self):
        """Opposite vectors have minimum similarity."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        result = cosine_similarity(vec1, vec2)
        assert result == 0.0  # Normalized from -1 to [0, 1]

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have middle similarity."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        result = cosine_similarity(vec1, vec2)
        assert result == 0.5  # Normalized from 0 to [0, 1]

    def test_dimension_mismatch(self):
        """Raises error for vectors with different dimensions."""
        vec1 = [1.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        with pytest.raises(ValueError, match="dimension mismatch"):
            cosine_similarity(vec1, vec2)

    def test_empty_vectors(self):
        """Raises error for empty vectors."""
        with pytest.raises(ValueError, match="cannot be empty"):
            cosine_similarity([], [])

    def test_normalized_output_range(self):
        """Output is always in [0, 1] range."""
        test_cases = [
            ([1.0, 0.0], [1.0, 0.0]),
            ([1.0, 0.0], [-1.0, 0.0]),
            ([1.0, 0.0], [0.0, 1.0]),
            ([0.5, 0.5], [0.5, -0.5]),
        ]
        for vec1, vec2 in test_cases:
            result = cosine_similarity(vec1, vec2)
            assert 0.0 <= result <= 1.0, f"Result {result} out of range for {vec1}, {vec2}"

    def test_floating_point_precision(self):
        """Handles floating point precision correctly."""
        # Very small vectors
        vec1 = [1e-10, 1e-10]
        vec2 = [1e-10, 1e-10]
        result = cosine_similarity(vec1, vec2)
        assert 0.0 <= result <= 1.0

    def test_high_dimensional_vectors(self):
        """Works with high-dimensional vectors."""
        dim = 1536  # Common embedding dimension
        vec1 = [1.0 / dim] * dim
        vec2 = [1.0 / dim] * dim
        result = cosine_similarity(vec1, vec2)
        assert 0.0 <= result <= 1.0
