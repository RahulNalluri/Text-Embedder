"""Clean extracted annual-report text for Word2Vec training."""

import re
import unicodedata


LIGATURE_ARTIFACT_PATTERN = re.compile(r"/([a-z])_([a-z])\.liga", re.IGNORECASE)
RUPEE_ARTIFACT_PATTERN = re.compile(r"/uni20b9", re.IGNORECASE)
BROKEN_WORD_PATTERN = re.compile(r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)")
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|₹|%|\d+(?:[,.]\d+)*", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"\d+(?:[,.]\d+)*")


def normalize_pdf_text(text: str) -> str:
    """Repair common PDF artifacts and normalize whitespace.

    This stage deliberately preserves sentence punctuation and financial words.
    It does not remove domain terms such as ``crore``, ``revenue``, or ``tax``.
    """

    if not isinstance(text, str):
        raise TypeError("Extracted PDF text must be a string.")

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\x00", "").replace("\u00ad", "")
    normalized = RUPEE_ARTIFACT_PATTERN.sub("₹", normalized)
    normalized = LIGATURE_ARTIFACT_PATTERN.sub(r"\1\2", normalized)
    normalized = BROKEN_WORD_PATTERN.sub("", normalized)

    paragraphs = []
    for paragraph in re.split(r"\n\s*\n+", normalized):
        paragraph = re.sub(r"[ \t]*\n[ \t]*", " ", paragraph)
        paragraph = re.sub(r"[ \t]+", " ", paragraph).strip()
        if paragraph:
            paragraphs.append(paragraph)

    return "\n\n".join(paragraphs)


def split_sentences(text: str) -> list[str]:
    """Split normalized text into non-empty sentence-like units."""

    if not isinstance(text, str):
        raise TypeError("Text to split must be a string.")

    sentences: list[str] = []
    for paragraph in re.split(r"\n\s*\n+", text):
        sentences.extend(
            sentence.strip()
            for sentence in SENTENCE_BOUNDARY_PATTERN.split(paragraph.strip())
            if sentence.strip()
        )
    return sentences


def tokenize_financial_sentence(sentence: str) -> list[str]:
    """Convert one sentence into consistent Word2Vec tokens.

    Exact numeric values are mapped to ``<number>`` to prevent thousands of
    one-use numbers from crowding the vocabulary. Currency and percentage
    symbols become readable tokens while words such as ``crore`` remain intact.
    """

    if not isinstance(sentence, str):
        raise TypeError("Sentence must be a string.")

    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(sentence.lower()):
        token = match.group()
        if NUMBER_PATTERN.fullmatch(token):
            tokens.append("<number>")
        elif token == "₹":
            tokens.append("rupee")
        elif token == "%":
            tokens.append("percent")
        else:
            tokens.append(token)
    return tokens


def preprocess_text(text: str, minimum_tokens: int = 3) -> list[list[str]]:
    """Return cleaned, tokenized sentences ready for Word2Vec training."""

    if minimum_tokens < 1:
        raise ValueError("minimum_tokens must be at least 1.")

    normalized = normalize_pdf_text(text)
    tokenized_sentences = (
        tokenize_financial_sentence(sentence)
        for sentence in split_sentences(normalized)
    )
    return [
        tokens
        for tokens in tokenized_sentences
        if len(tokens) >= minimum_tokens
    ]
