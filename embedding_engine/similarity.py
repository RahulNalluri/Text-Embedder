import numpy as np


def calculate_cosine_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:
    """
    Measure the similarity between two embedding vectors.
    """

    if vector_a.ndim != 1 or vector_b.ndim != 1:
        raise ValueError(
            "Both embeddings must be one-dimensional vectors."
        )

    if vector_a.shape != vector_b.shape:
        raise ValueError(
            "Both embeddings must have the same dimensions."
        )

    length_a = np.linalg.norm(vector_a)
    length_b = np.linalg.norm(vector_b)

    if length_a == 0 or length_b == 0:
        return 0.0

    similarity = np.dot(
        vector_a,
        vector_b,
    ) / (length_a * length_b)

    return float(
        np.clip(similarity, -1.0, 1.0)
    )


def build_similarity_matrix(
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    Calculate cosine similarity between every pair of embeddings.
    """

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a two-dimensional matrix."
        )

    vector_lengths = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    normalized_embeddings = np.divide(
        embeddings,
        vector_lengths,
        out=np.zeros_like(embeddings, dtype=float),
        where=vector_lengths != 0,
    )

    similarity_matrix = (
        normalized_embeddings
        @ normalized_embeddings.T
    )

    return np.clip(
        similarity_matrix,
        -1.0,
        1.0,
    )