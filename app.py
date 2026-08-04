import numpy as np
import pandas as pd
import streamlit as st

from embedding_engine import (
    build_cooccurrence_matrix,
    build_similarity_matrix,
    build_vocabulary,
    calculate_ppmi,
    generate_sentence_embedding,
    generate_word_embeddings,
    tokenize,
)
from ui import (
    display_embedding,
    display_labeled_matrix,
)


CONTEXT_WINDOW_SIZE = 3
EMBEDDING_DIMENSION = 5


DEFAULT_CORPUS = """
Python is a programming language.
Developers use Python to build applications.
Java is also a programming language.
Developers use Java to build software.
Machine learning uses data and algorithms.
Python is popular for machine learning and data science.
Programmers write code to solve problems.
Software developers build useful applications with code.
""".strip()


st.set_page_config(
    page_title="Vector Forge",
    page_icon="🧠",
    layout="wide",
)


st.markdown(
    """
    <style>
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 12px;
            padding: 12px;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 10px;
            font-weight: 600;
        }

        div.stDownloadButton > button {
            width: 100%;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("🧠 Vector Forge")

st.markdown(
    """
    **Create and explore text embeddings from scratch.**

    This application uses word co-occurrence, PPMI, and SVD.
    It does not use a pretrained embedding model or an external API.
    """
)


with st.sidebar:
    st.header("🧪 Project information")

    st.markdown(
        """
        ### How it works

        1. Clean and tokenize the corpus
        2. Build a vocabulary
        3. Count nearby words
        4. Calculate PPMI
        5. Apply SVD
        6. Display the embeddings
        """
    )

    st.warning(
        "A larger and more varied corpus produces more "
        "meaningful embeddings."
    )


with st.form("embedding_form"):
    corpus = st.text_area(
        "📚 Training corpus",
        value=DEFAULT_CORPUS,
        height=220,
        help=(
            "The application learns word relationships "
            "only from this text."
        ),
    )

    sentences_input = st.text_area(
        "✍️ Sentences to embed",
        value=(
            "I love Python\n"
            "Developers build software\n"
            "Machine learning uses data"
        ),
        height=130,
        help="Enter one sentence per line.",
    )

    generate_button = st.form_submit_button(
        "⚡ Generate embeddings",
        type="primary",
    )


if generate_button:
    st.session_state.pop(
        "embedding_results",
        None,
    )

    try:
        corpus_tokens = tokenize(corpus)

        if len(corpus_tokens) < 3:
            raise ValueError(
                "The training corpus must contain at least "
                "three words."
            )

        vocabulary = build_vocabulary(
            corpus_tokens
        )

        if len(vocabulary) < EMBEDDING_DIMENSION:
            raise ValueError(
                f"The corpus must contain at least "
                f"{EMBEDDING_DIMENSION} unique words."
            )

        cooccurrence_matrix = build_cooccurrence_matrix(
            tokens=corpus_tokens,
            vocabulary=vocabulary,
            window_size=CONTEXT_WINDOW_SIZE,
        )

        ppmi_matrix = calculate_ppmi(
            cooccurrence_matrix
        )

        word_embeddings = generate_word_embeddings(
            ppmi_matrix=ppmi_matrix,
            embedding_dimension=EMBEDDING_DIMENSION,
        )

        sentence_texts = [
            sentence.strip()
            for sentence in sentences_input.splitlines()
            if sentence.strip()
        ]

        if not sentence_texts:
            raise ValueError(
                "Enter at least one sentence to embed."
            )

        vocabulary_set = set(vocabulary)
        sentence_results = []

        for sentence in sentence_texts:
            sentence_tokens = tokenize(sentence)

            known_tokens = [
                token
                for token in sentence_tokens
                if token in vocabulary_set
            ]

            unknown_tokens = [
                token
                for token in sentence_tokens
                if token not in vocabulary_set
            ]

            if not known_tokens:
                continue

            sentence_embedding = (
                generate_sentence_embedding(
                    tokens=sentence_tokens,
                    vocabulary=vocabulary,
                    word_embeddings=word_embeddings,
                )
            )

            sentence_results.append(
                {
                    "text": sentence,
                    "embedding": sentence_embedding,
                    "known_tokens": known_tokens,
                    "unknown_tokens": unknown_tokens,
                }
            )

        if not sentence_results:
            raise ValueError(
                "None of the entered sentences contain words "
                "from the training vocabulary."
            )

        st.session_state.embedding_results = {
            "tokens": corpus_tokens,
            "vocabulary": vocabulary,
            "cooccurrence_matrix": cooccurrence_matrix,
            "ppmi_matrix": ppmi_matrix,
            "word_embeddings": word_embeddings,
            "sentence_results": sentence_results,
        }

    except (TypeError, ValueError) as error:
        st.error(str(error))


if "embedding_results" not in st.session_state:
    st.info(
        "👆 Enter a corpus and click **Generate embeddings** "
        "to begin exploring."
    )

    st.stop()


results = st.session_state.embedding_results

tokens = results["tokens"]
vocabulary = results["vocabulary"]
cooccurrence_matrix = results["cooccurrence_matrix"]
ppmi_matrix = results["ppmi_matrix"]
word_embeddings = results["word_embeddings"]
sentence_results = results["sentence_results"]


st.success(
    "Embeddings successfully generated from the supplied corpus."
)


metric1, metric2, metric3 = st.columns(3)

metric1.metric(
    "Corpus tokens",
    len(tokens),
)

metric2.metric(
    "Vocabulary words",
    len(vocabulary),
)

metric3.metric(
    "Embedded sentences",
    len(sentence_results),
)


pipeline_tab, words_tab, sentences_tab, similarity_tab = st.tabs(
    [
        "🔬 Learning process",
        "🔤 Word embeddings",
        "📝 Sentence embeddings",
        "📐 Similarity",
    ]
)


with pipeline_tab:
    st.markdown(
        """
        ### From text to numbers

        This section shows how the supplied text is converted
        into embeddings.
        """
    )

    with st.expander(
        "View tokens and vocabulary",
        expanded=True,
    ):
        st.markdown("#### Tokens")

        st.write(tokens)

        st.markdown("#### Vocabulary")

        vocabulary_dataframe = pd.DataFrame(
            {
                "Index": range(len(vocabulary)),
                "Word": vocabulary,
            }
        )

        st.dataframe(
            vocabulary_dataframe,
            width="stretch",
            hide_index=True,
        )

    with st.expander(
        "View co-occurrence matrix",
        expanded=False,
    ):
        display_labeled_matrix(
            title="Weighted co-occurrence matrix",
            matrix=cooccurrence_matrix,
            row_labels=vocabulary,
            column_labels=vocabulary,
        )

        st.caption(
            "Larger values mean that two words appeared "
            "more frequently or more closely together."
        )

    with st.expander(
        "View PPMI matrix",
        expanded=False,
    ):
        display_labeled_matrix(
            title="PPMI relationship matrix",
            matrix=ppmi_matrix,
            row_labels=vocabulary,
            column_labels=vocabulary,
        )

        st.caption(
            "PPMI highlights relationships that occur more "
            "often than expected by chance."
        )


with words_tab:
    st.markdown(
        """
        ### Inspect an individual word

        Select a vocabulary word to view the dense embedding
        created for it by SVD.
        """
    )

    selected_word = st.selectbox(
        "Choose a word",
        options=vocabulary,
    )

    selected_word_index = vocabulary.index(
        selected_word
    )

    selected_word_embedding = word_embeddings[
        selected_word_index
    ]

    display_embedding(
        title=f'Embedding for “{selected_word}”',
        embedding=selected_word_embedding,
    )

    st.divider()

    word_dimension_labels = [
        f"Dimension {index}"
        for index in range(word_embeddings.shape[1])
    ]

    with st.expander(
        "View every word embedding",
        expanded=False,
    ):
        display_labeled_matrix(
            title="Complete word-embedding matrix",
            matrix=word_embeddings,
            row_labels=vocabulary,
            column_labels=word_dimension_labels,
        )

    word_embeddings_dataframe = pd.DataFrame(
        word_embeddings,
        index=vocabulary,
        columns=word_dimension_labels,
    )

    st.download_button(
        "⬇️ Download word embeddings as CSV",
        data=word_embeddings_dataframe.to_csv(),
        file_name="word_embeddings.csv",
        mime="text/csv",
    )


with sentences_tab:
    st.markdown(
        """
        ### Inspect an entire sentence

        A sentence embedding is created by averaging its known
        word embeddings and normalizing the result.
        """
    )

    selected_sentence_index = st.selectbox(
        "Choose a sentence",
        options=range(len(sentence_results)),
        format_func=lambda index: sentence_results[index]["text"],
    )

    selected_sentence = sentence_results[
        selected_sentence_index
    ]

    known_text = ", ".join(
        selected_sentence["known_tokens"]
    )

    st.write(
        f"**Words used:** {known_text}"
    )

    if selected_sentence["unknown_tokens"]:
        unknown_text = ", ".join(
            selected_sentence["unknown_tokens"]
        )

        st.warning(
            "These words were not present in the training "
            f"corpus and were ignored: {unknown_text}"
        )

    display_embedding(
        title="Actual sentence embedding",
        embedding=selected_sentence["embedding"],
    )

    sentence_dimension_labels = [
        f"Dimension {index}"
        for index in range(
            sentence_results[0]["embedding"].shape[0]
        )
    ]

    sentence_embeddings_dataframe = pd.DataFrame(
        [
            result["embedding"]
            for result in sentence_results
        ],
        index=[
            result["text"]
            for result in sentence_results
        ],
        columns=sentence_dimension_labels,
    )

    st.download_button(
        "⬇️ Download sentence embeddings as CSV",
        data=sentence_embeddings_dataframe.to_csv(),
        file_name="sentence_embeddings.csv",
        mime="text/csv",
    )


with similarity_tab:
    st.markdown(
        """
        ### Compare sentence directions

        Cosine similarity measures whether sentence embeddings
        point in similar directions.
        """
    )

    if len(sentence_results) < 2:
        st.info(
            "Enter at least two valid sentences to generate "
            "a similarity matrix."
        )

    else:
        sentence_embeddings = np.vstack(
            [
                result["embedding"]
                for result in sentence_results
            ]
        )

        similarity_matrix = build_similarity_matrix(
            sentence_embeddings
        )

        sentence_labels = [
            f"Sentence {index + 1}"
            for index in range(len(sentence_results))
        ]

        display_labeled_matrix(
            title="Cosine-similarity matrix",
            matrix=similarity_matrix,
            row_labels=sentence_labels,
            column_labels=sentence_labels,
        )

        sentence_key = pd.DataFrame(
            {
                "Label": sentence_labels,
                "Sentence": [
                    result["text"]
                    for result in sentence_results
                ],
            }
        )

        st.markdown("#### Sentence key")

        st.dataframe(
            sentence_key,
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "A score close to 1 means the vectors point in "
            "similar directions. Results depend entirely on "
            "the supplied training corpus."
        )


st.divider()

st.caption(
    "🧪 Vector Forge creates corpus-specific statistical "
    "embeddings without pretrained model weights."
)
