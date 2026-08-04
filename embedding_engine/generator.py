import numpy as np


def generate_word_embeddings(
    ppmi_matrix: np.ndarray,
    embedding_dimension: int,
) -> np.ndarray:
    """
    Reduce the PPMI matrix into dense word embeddings using
    Singular Value Decomposition (SVD).
    """

    if ppmi_matrix.ndim != 2:
        raise ValueError(
            "The PPMI matrix must be two-dimensional."
        )

    if embedding_dimension < 1:
        raise ValueError(
            "The embedding dimension must be at least 1."
        )

    vocabulary_size = ppmi_matrix.shape[0]

    if vocabulary_size == 0:
        return np.empty((0, 0), dtype=float)

    actual_dimension = min(
        embedding_dimension,
        ppmi_matrix.shape[0],
        ppmi_matrix.shape[1],
    )

    left_vectors, singular_values, _ = np.linalg.svd(
        ppmi_matrix,
        full_matrices=False,
    )

    embeddings = (
        left_vectors[:, :actual_dimension]
        * np.sqrt(singular_values[:actual_dimension])
    )

    vector_lengths = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    nonzero_vectors = vector_lengths[:, 0] > 0

    embeddings[nonzero_vectors] = (
        embeddings[nonzero_vectors]
        / vector_lengths[nonzero_vectors]
    )

    return embeddings


def generate_sentence_embedding(
    tokens: list[str],
    vocabulary: list[str],
    word_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Create one sentence embedding by averaging the embeddings
    of all known words in the sentence.
    """

    if word_embeddings.ndim != 2:
        raise ValueError(
            "Word embeddings must be a two-dimensional matrix."
        )

    if len(vocabulary) != word_embeddings.shape[0]:
        raise ValueError(
            "The vocabulary and word embeddings do not match."
        )

    word_to_index = {
        word: index
        for index, word in enumerate(vocabulary)
    }

    sentence_word_vectors = []

    for token in tokens:
        if token in word_to_index:
            word_index = word_to_index[token]
            word_vector = word_embeddings[word_index]

            sentence_word_vectors.append(word_vector)

    if not sentence_word_vectors:
        raise ValueError(
            "The sentence does not contain any words "
            "from the trained vocabulary."
        )

    sentence_embedding = np.mean(
        sentence_word_vectors,
        axis=0,
    )

    vector_length = np.linalg.norm(
        sentence_embedding
    )

    if vector_length > 0:
        sentence_embedding = (
            sentence_embedding / vector_length
        )

    return sentence_embedding