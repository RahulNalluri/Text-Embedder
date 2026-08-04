import numpy as np
import pandas as pd
import streamlit as st


def display_labeled_matrix(
    title: str,
    matrix: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
) -> None:
    """
    Display a NumPy matrix as a labeled Streamlit table.
    """

    st.subheader(title)

    dataframe = pd.DataFrame(
        matrix,
        index=row_labels,
        columns=column_labels,
    )

    st.dataframe(
        dataframe.style.format("{:.3f}"),
        width="stretch",
    )


def display_embedding(
    title: str,
    embedding: np.ndarray,
) -> None:
    """
    Display a complete embedding as raw values, statistics,
    a table, and a bar chart.
    """

    st.subheader(title)

    raw_embedding = np.array2string(
        embedding,
        precision=6,
        separator=", ",
        threshold=np.inf,
        max_line_width=120,
    )

    st.code(
        raw_embedding,
        language="text",
    )

    column1, column2, column3 = st.columns(3)

    column1.metric(
        "Minimum",
        f"{embedding.min():.6f}",
    )

    column2.metric(
        "Maximum",
        f"{embedding.max():.6f}",
    )

    column3.metric(
        "Vector length",
        f"{np.linalg.norm(embedding):.6f}",
    )

    embedding_dataframe = pd.DataFrame(
        {
            "Dimension": np.arange(
                embedding.shape[0]
            ),
            "Value": embedding,
        }
    )

    left_column, right_column = st.columns(2)

    with left_column:
        st.markdown("#### Dimension values")

        st.dataframe(
            embedding_dataframe.style.format(
                {"Value": "{:.6f}"}
            ),
            width="stretch",
            hide_index=True,
        )

    with right_column:
        st.markdown("#### Dimension chart")

        st.bar_chart(
            embedding_dataframe,
            x="Dimension",
            y="Value",
            height=400,
        )
