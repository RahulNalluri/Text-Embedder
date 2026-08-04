import numpy as np


def build_cooccurrence_matrix(
    tokens: list[str],
    vocabulary: list[str],
    window_size: int = 2,
) -> np.ndarray:
    """
    Build a weighted word co-occurrence matrix.

    Words that are closer together receive a higher weight.
    """

    if window_size < 1:
        raise ValueError(
            "Window size must be at least 1."
        )

    if len(vocabulary) != len(set(vocabulary)):
        raise ValueError(
            "The vocabulary cannot contain duplicate words."
        )

    vocabulary_size = len(vocabulary)

    if vocabulary_size == 0:
        return np.empty((0, 0), dtype=float)

    word_to_index = {
        word: index
        for index, word in enumerate(vocabulary)
    }

    matrix = np.zeros(
        shape=(vocabulary_size, vocabulary_size),
        dtype=float,
    )

    for center_position, center_word in enumerate(tokens):
        if center_word not in word_to_index:
            continue

        center_index = word_to_index[center_word]

        window_start = max(
            0,
            center_position - window_size,
        )

        window_end = min(
            len(tokens),
            center_position + window_size + 1,
        )

        for context_position in range(
            window_start,
            window_end,
        ):
            if context_position == center_position:
                continue

            context_word = tokens[context_position]

            if context_word not in word_to_index:
                continue

            context_index = word_to_index[context_word]

            distance = abs(
                center_position - context_position
            )

            weight = 1.0 / distance

            matrix[
                center_index,
                context_index,
            ] += weight

    return matrix