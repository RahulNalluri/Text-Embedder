"""Build a reproducible Word2Vec corpus from validated annual-report PDFs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

from .config import PROCESSED_DATA_DIR
from .pdf_extraction import PDFExtractionResult, extract_pdf_text
from .preprocessing import preprocess_text


Extractor = Callable[[str | Path], PDFExtractionResult]
Preprocessor = Callable[[str, int], list[list[str]]]


@dataclass(frozen=True, slots=True)
class ProcessedDocument:
    """Corpus statistics for one successfully processed PDF."""

    source_path: Path
    content_sha256: str
    total_pages: int
    text_pages: int
    possible_scanned_pages: tuple[int, ...]
    sentence_count: int
    raw_sentence_count: int
    duplicate_sentences_removed: int
    token_count: int
    sentence_start: int
    sentence_end: int
    repeated_edge_lines_removed: int = 0
    company_name: str = ""
    sector: str = ""
    financial_year: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe document statistics."""

        return {
            "source_filename": self.source_path.name,
            "content_sha256": self.content_sha256,
            "total_pages": self.total_pages,
            "text_pages": self.text_pages,
            "possible_scanned_pages": list(self.possible_scanned_pages),
            "sentence_count": self.sentence_count,
            "raw_sentence_count": self.raw_sentence_count,
            "duplicate_sentences_removed": self.duplicate_sentences_removed,
            "token_count": self.token_count,
            "sentence_start": self.sentence_start,
            "sentence_end": self.sentence_end,
            "repeated_edge_lines_removed": self.repeated_edge_lines_removed,
            "company_name": self.company_name,
            "sector": self.sector,
            "financial_year": self.financial_year,
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
            "repeated_edge_lines_removed": sum(
                document.repeated_edge_lines_removed for document in self.documents
            ),
            "raw_sentence_count": sum(
                document.raw_sentence_count for document in self.documents
            ),
            "sentence_count": len(self.sentences),
            "token_count": self.token_count,
            "vocabulary_size": len(vocabulary_counts),
            "duplicate_sentences_removed": sum(
                document.duplicate_sentences_removed
                for document in self.documents
            ),
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


def remove_repeated_page_edges(
    extraction: PDFExtractionResult,
    edge_line_count: int = 2,
    minimum_repetitions: int = 3,
) -> tuple[str, int]:
    """Remove recurring short header/footer lines from extracted page text.

    Only lines in the first or last ``edge_line_count`` non-empty lines of a
    page are considered. A candidate must occur on at least 10% of pages and
    at least ``minimum_repetitions`` times, which avoids removing ordinary
    headings that happen to appear once near a page boundary.
    """

    if edge_line_count < 1:
        raise ValueError("edge_line_count must be at least 1.")
    if minimum_repetitions < 2:
        raise ValueError("minimum_repetitions must be at least 2.")

    page_lines: list[list[str]] = []
    edge_counts: Counter[str] = Counter()
    for page in extraction.pages:
        lines = [line.strip() for line in page.text.splitlines() if line.strip()]
        page_lines.append(lines)
        edge_lines = lines[:edge_line_count] + lines[-edge_line_count:]
        edge_counts.update(
            line.casefold()
            for line in set(edge_lines)
            if 2 <= len(line) <= 120
        )

    repetition_threshold = max(
        minimum_repetitions,
        max(1, int(extraction.total_pages * 0.10)),
    )
    repeated_lines = {
        line for line, count in edge_counts.items() if count >= repetition_threshold
    }

    cleaned_pages: list[str] = []
    removed_count = 0
    for lines in page_lines:
        edge_indexes = set(range(min(edge_line_count, len(lines))))
        edge_indexes.update(
            range(max(0, len(lines) - edge_line_count), len(lines))
        )
        kept_lines = []
        for index, line in enumerate(lines):
            if index in edge_indexes and line.casefold() in repeated_lines:
                removed_count += 1
            else:
                kept_lines.append(line)
        if kept_lines:
            cleaned_pages.append("\n".join(kept_lines))

    return "\n\n".join(cleaned_pages), removed_count


def build_corpus(
    pdf_paths: Iterable[str | Path],
    minimum_tokens: int = 3,
    extractor: Extractor = extract_pdf_text,
    preprocessor: Preprocessor = preprocess_text,
    document_metadata: Mapping[str, Mapping[str, str]] | None = None,
    remove_repeated_edges: bool = True,
    deduplicate_sentences: bool = True,
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
    seen_sentences: set[tuple[str, ...]] = set()

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
            if remove_repeated_edges:
                cleaned_text, repeated_edge_lines_removed = remove_repeated_page_edges(
                    extraction
                )
            else:
                cleaned_text = extraction.combined_text
                repeated_edge_lines_removed = 0
            tokenized_sentences = preprocessor(
                cleaned_text,
                minimum_tokens=minimum_tokens,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            failed_documents.append(FailedDocument(resolved_path, str(error)))
            continue

        raw_sentences = tuple(
            tuple(tokens) for tokens in tokenized_sentences
        )
        if deduplicate_sentences:
            retained_sentences = []
            for sentence in raw_sentences:
                if sentence in seen_sentences:
                    continue
                seen_sentences.add(sentence)
                retained_sentences.append(sentence)
            normalized_sentences = tuple(retained_sentences)
        else:
            normalized_sentences = raw_sentences
        duplicate_sentences_removed = len(raw_sentences) - len(normalized_sentences)
        sentence_start = len(sentences)
        sentences.extend(normalized_sentences)
        sentence_end = len(sentences)
        metadata = (document_metadata or {}).get(resolved_path.name, {})
        documents.append(
            ProcessedDocument(
                source_path=resolved_path,
                content_sha256=content_hash,
                total_pages=extraction.total_pages,
                text_pages=extraction.text_pages,
                possible_scanned_pages=extraction.possible_scanned_pages,
                sentence_count=len(normalized_sentences),
                raw_sentence_count=len(raw_sentences),
                duplicate_sentences_removed=duplicate_sentences_removed,
                token_count=sum(len(sentence) for sentence in normalized_sentences),
                sentence_start=sentence_start,
                sentence_end=sentence_end,
                repeated_edge_lines_removed=repeated_edge_lines_removed,
                company_name=metadata.get("company_name", ""),
                sector=metadata.get("sector", ""),
                financial_year=metadata.get("financial_year", ""),
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
    document_directory = directory / "documents"
    document_directory.mkdir(parents=True, exist_ok=True)

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

    for document in result.documents:
        document_path = document_directory / f"{document.source_path.stem}.txt"
        with document_path.open("w", encoding="utf-8", newline="\n") as file:
            for sentence in result.sentences[
                document.sentence_start : document.sentence_end
            ]:
                file.write(" ".join(sentence))
                file.write("\n")

    return {
        "sentences": sentences_path,
        "vocabulary": vocabulary_path,
        "report": report_path,
        "documents": document_directory,
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
