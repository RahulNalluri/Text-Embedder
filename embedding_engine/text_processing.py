import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    """
    Convert text into a cleaned list of lowercase words.
    """

    if not isinstance(text, str):
        raise TypeError(
            "The provided text must be a string."
        )

    text = text.lower()

    return re.findall(
        pattern=r"\b[a-z0-9']+\b",
        string=text,
    )


def build_vocabulary(
    tokens: list[str],
) -> list[str]:
    """
    Create a sorted vocabulary containing every unique token.
    """

    word_counts = Counter(tokens)

    vocabulary = sorted(
        word_counts.keys()
    )

    return vocabulary