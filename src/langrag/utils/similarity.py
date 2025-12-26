"""Vector similarity calculation utilities."""


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Similarity score in [0, 1] range (normalized from [-1, 1])

    Raises:
        ValueError: If vectors have different dimensions
    """
    if len(vec1) != len(vec2):
        raise ValueError(
            f"Vector dimension mismatch: {len(vec1)} != {len(vec2)}"
        )

    if not vec1:
        raise ValueError("Vectors cannot be empty")

    # Compute dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))

    # Clamp to [-1, 1] to handle floating point errors
    dot_product = max(-1.0, min(1.0, dot_product))

    # Convert from [-1, 1] to [0, 1]
    return (dot_product + 1.0) / 2.0
