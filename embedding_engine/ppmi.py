import numpy as np


def calculate_ppmi(
    cooccurrence_matrix: np.ndarray,
) -> np.ndarray:
    """
    Convert a co-occurrence matrix into a Positive Pointwise
    Mutual Information (PPMI) matrix.
    """

    if cooccurrence_matrix.ndim != 2:
        raise ValueError(
            "The co-occurrence matrix must be two-dimensional."
        )

    if np.any(cooccurrence_matrix < 0):
        raise ValueError(
            "The co-occurrence matrix cannot contain negative values."
        )

    total_count = cooccurrence_matrix.sum()

    if total_count == 0:
        return np.zeros_like(
            cooccurrence_matrix,
            dtype=float,
        )

    row_totals = cooccurrence_matrix.sum(
        axis=1,
        keepdims=True,
    )

    column_totals = cooccurrence_matrix.sum(
        axis=0,
        keepdims=True,
    )

    expected_counts = (
        row_totals @ column_totals
    ) / total_count

    valid_positions = (
        (cooccurrence_matrix > 0)
        & (expected_counts > 0)
    )

    ppmi_matrix = np.zeros_like(
        cooccurrence_matrix,
        dtype=float,
    )

    ppmi_matrix[valid_positions] = np.maximum(
        np.log2(
            cooccurrence_matrix[valid_positions]
            / expected_counts[valid_positions]
        ),
        0,
    )

    return ppmi_matrix