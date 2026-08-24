"""Build a reproducible Word2Vec corpus from validated annual-report PDFs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .config import PROCESSED_DATA_DIR
from .pdf_extraction import PDFExtractionResult, extract_pdf_text
from .preprocessing import preprocess_text


Extractor = Callable[[str | Path], PDFExtractionResult]
Preprocessor = Callable[[str, int], list[list[str]]]


@dataclass(frozen=True, slots=True)
class ProcessedDocument:
    """Corpus statistics for one successfully processed PDF."""

    source_path: Path
    total_pages: int
    text_pages: int
    possible_scanned_pages: tuple[int, ...]
    sentence_count: int
    token_count: int

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe document statistics."""

        return {
            "source_filename": self.source_path.name,
            "total_pages": self.total_pages,
            "text_pages": self.text_pages,
            "possible_scanned_pages": list(self.possible_scanned_pages),
            "sentence_count": self.sentence_count,
            "token_count": self.token_count,
        }


@dataclass(frozen=True, slots=True)
class SkippedDocument:
    """A PDF deliberately omitted because it duplicates earlier content."""

    source_path: Path
    duplicate_of: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "source_filename": self.source_path.name,
            "reason": "duplicate_file_content",
            "duplicate_of": self.duplicate_of.name,
        }


@dataclass(frozen=True, slots=True)
class FailedDocument:
    """A PDF that could not be processed without stopping the full build."""

    source_path: Path
    error: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source_filename": self.source_path.name,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class CorpusBuildResult:
    """Combined tokenized corpus and its processing audit trail."""

    sentences: tuple[tuple[str, ...], ...]
    documents: tuple[ProcessedDocument, ...]
    skipped_documents: tuple[SkippedDocument, ...]
    failed_documents: tuple[FailedDocument, ...]

    @property
    def token_count(self) -> int:
        return sum(len(sentence) for sentence in self.sentences)

    @property
    def vocabulary_counts(self) -> Counter[str]:
        return Counter(token for sentence in self.sentences for token in sentence)

    def to_metadata_dict(self) -> dict[str, object]:
        """Return a JSON-safe corpus quality report."""

        vocabulary_counts = self.vocabulary_counts
        return {
            "documents_processed": len(self.documents),
            "documents_skipped": len(self.skipped_documents),
            "documents_failed": len(self.failed_documents),
            "pages_processed": sum(document.total_pages for document in self.documents),
            "text_pages": sum(document.text_pages for document in self.documents),
            "possible_scanned_pages": sum(
                len(document.possible_scanned_pages) for document in self.documents
            ),
            "sentence_count": len(self.sentences),
            "token_count": self.token_count,
            "vocabulary_size": len(vocabulary_counts),
            "documents": [document.to_dict() for document in self.documents],
            "skipped_documents": [document.to_dict() for document in self.skipped_documents],
            "failed_documents": [document.to_dict() for document in self.failed_documents],
        }


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_corpus(
    pdf_paths: Iterable[str | Path],
    minimum_tokens: int = 3,
    extractor: Extractor = extract_pdf_text,
    preprocessor: Preprocessor = preprocess_text,
) -> CorpusBuildResult:
    """Extract and preprocess selected PDFs into one tokenized corpus.

    The caller supplies paths explicitly. This avoids silently training on a
    partial annual report or a quarterly-results PDF left in the raw-data folder.
    A failed document is recorded, while other valid documents continue.
    """

    paths = tuple(Path(path) for path in pdf_paths)
    if not paths:
        raise ValueError("At least one PDF path is required to build a corpus.")
    if minimum_tokens < 1:
        raise ValueError("minimum_tokens must be at least 1.")

    sentences: list[tuple[str, ...]] = []
    documents: list[ProcessedDocument] = []
    skipped_documents: list[SkippedDocument] = []
    failed_documents: list[FailedDocument] = []
    known_hashes: dict[str, Path] = {}

    for path in paths:
        resolved_path = path.resolve()
        try:
            content_hash = file_sha256(resolved_path)
        except OSError as error:
            failed_documents.append(FailedDocument(resolved_path, str(error)))
            continue

        original_path = known_hashes.get(content_hash)
        if original_path is not None:
            skipped_documents.append(SkippedDocument(resolved_path, original_path))
            continue
        known_hashes[content_hash] = resolved_path

        try:
            extraction = extractor(resolved_path)
            tokenized_sentences = preprocessor(
                extraction.combined_text,
                minimum_tokens=minimum_tokens,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            failed_documents.append(FailedDocument(resolved_path, str(error)))
            continue

        normalized_sentences = tuple(
            tuple(tokens) for tokens in tokenized_sentences
        )
        sentences.extend(normalized_sentences)
        documents.append(
            ProcessedDocument(
                source_path=resolved_path,
                total_pages=extraction.total_pages,
                text_pages=extraction.text_pages,
                possible_scanned_pages=extraction.possible_scanned_pages,
                sentence_count=len(normalized_sentences),
                token_count=sum(len(sentence) for sentence in normalized_sentences),
            )
        )

    return CorpusBuildResult(
        sentences=tuple(sentences),
        documents=tuple(documents),
        skipped_documents=tuple(skipped_documents),
        failed_documents=tuple(failed_documents),
    )


def save_corpus(
    result: CorpusBuildResult,
    output_directory: str | Path = PROCESSED_DATA_DIR,
) -> dict[str, Path]:
    """Save tokenized sentences, vocabulary counts, and build metadata locally."""

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)

    sentences_path = directory / "sentences.txt"
    vocabulary_path = directory / "vocabulary_counts.csv"
    report_path = directory / "corpus_report.json"

    with sentences_path.open("w", encoding="utf-8", newline="\n") as file:
        for sentence in result.sentences:
            file.write(" ".join(sentence))
            file.write("\n")

    with vocabulary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["token", "count"])
        for token, count in result.vocabulary_counts.most_common():
            writer.writerow([token, count])

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(result.to_metadata_dict(), file, indent=2)

    return {
        "sentences": sentences_path,
        "vocabulary": vocabulary_path,
        "report": report_path,
    }


def parse_arguments() -> argparse.Namespace:
    """Parse explicit input PDFs for local corpus construction."""

    parser = argparse.ArgumentParser(
        description="Build a Word2Vec corpus from selected annual-report PDFs."
    )
    parser.add_argument(
        "--pdf",
        action="append",
        required=True,
        help="Path to one validated complete annual-report PDF. Repeat per PDF.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROCESSED_DATA_DIR),
        help="Local directory for processed corpus files.",
    )
    parser.add_argument(
        "--minimum-tokens",
        type=int,
        default=3,
        help="Discard sentences shorter than this number of tokens.",
    )
    return parser.parse_args()


def main() -> None:
    """Build and save a corpus from explicitly selected PDFs."""

    arguments = parse_arguments()
    result = build_corpus(arguments.pdf, minimum_tokens=arguments.minimum_tokens)
    output_paths = save_corpus(result, arguments.output_dir)

    print(json.dumps(result.to_metadata_dict(), indent=2))
    print("Saved corpus files:")
    for label, path in output_paths.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
