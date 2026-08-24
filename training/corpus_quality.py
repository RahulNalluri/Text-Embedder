"""Evaluate corpus balance and financial-term coverage before Word2Vec training."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .config import METADATA_DIR, PROCESSED_DATA_DIR
from .preprocessing import tokenize_financial_sentence


@dataclass(frozen=True, slots=True)
class TargetTerm:
    """One financial term used to assess vocabulary coverage."""

    term: str
    category: str
    priority: str

    @property
    def component_tokens(self) -> tuple[str, ...]:
        """Return the token components needed for vocabulary-level coverage."""

        return tuple(tokenize_financial_sentence(self.term))


def load_tokenized_sentences(path: str | Path) -> list[list[str]]:
    """Load one space-separated tokenized sentence from each corpus line."""

    with Path(path).open(encoding="utf-8") as file:
        return [line.split() for line in file if line.strip()]


def load_target_terms(path: str | Path) -> list[TargetTerm]:
    """Load the project-owned financial terminology taxonomy."""

    with Path(path).open(encoding="utf-8", newline="") as file:
        return [
            TargetTerm(
                term=row["term"].strip(),
                category=row["category"].strip(),
                priority=row["priority"].strip(),
            )
            for row in csv.DictReader(file)
        ]


def load_corpus_metadata(path: str | Path) -> dict[str, object]:
    """Load the JSON audit report produced by the corpus builder."""

    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def evaluate_corpus_quality(
    sentences: list[list[str]],
    target_terms: list[TargetTerm],
    corpus_metadata: dict[str, object],
    top_token_limit: int = 20,
) -> dict[str, object]:
    """Return reproducible corpus-quality measures and review warnings.

    Multi-word targets are assessed using *component-token coverage*: every word
    in ``cash flow`` must be in the vocabulary. It does not claim that the phrase
    itself has been learned as one Word2Vec token; phrase detection comes later.
    """

    if not sentences:
        raise ValueError("Cannot evaluate an empty corpus.")
    if not target_terms:
        raise ValueError("At least one target term is required.")
    if top_token_limit < 1:
        raise ValueError("top_token_limit must be at least 1.")

    token_counts = Counter(token for sentence in sentences for token in sentence)
    vocabulary = set(token_counts)
    total_tokens = sum(token_counts.values())
    average_sentence_length = total_tokens / len(sentences)

    covered_terms = [
        target
        for target in target_terms
        if target.component_tokens and all(token in vocabulary for token in target.component_tokens)
    ]
    missing_terms = [
        target.term
        for target in target_terms
        if target not in covered_terms
    ]

    priority_totals = Counter(target.priority for target in target_terms)
    priority_covered = Counter(target.priority for target in covered_terms)
    category_totals = Counter(target.category for target in target_terms)
    category_covered = Counter(target.category for target in covered_terms)

    documents = corpus_metadata.get("documents", [])
    document_shares = []
    if isinstance(documents, list):
        for document in documents:
            if not isinstance(document, dict):
                continue
            token_count = int(document.get("token_count", 0))
            document_shares.append(
                {
                    "source_filename": document.get("source_filename", "unknown"),
                    "token_count": token_count,
                    "share_of_corpus": round(token_count / total_tokens, 4),
                }
            )

    warnings: list[str] = []
    largest_document_share = max(
        (document["share_of_corpus"] for document in document_shares), default=0.0
    )
    core_coverage = priority_covered["core"] / max(priority_totals["core"], 1)
    if int(corpus_metadata.get("documents_failed", 0)):
        warnings.append("One or more selected documents failed during corpus construction.")
    if largest_document_share > 0.30:
        warnings.append("One document contributes more than 30% of all corpus tokens.")
    if core_coverage < 0.70:
        warnings.append("Fewer than 70% of core target terms have component-token coverage.")
    if token_counts["<number>"] / total_tokens > 0.10:
        warnings.append("The <number> token exceeds 10% of the corpus; review numeric-table noise.")

    return {
        "documents_processed": int(corpus_metadata.get("documents_processed", 0)),
        "sentence_count": len(sentences),
        "token_count": total_tokens,
        "vocabulary_size": len(vocabulary),
        "average_sentence_length": round(average_sentence_length, 2),
        "most_common_tokens": [
            {"token": token, "count": count}
            for token, count in token_counts.most_common(top_token_limit)
        ],
        "dominant_tokens": [
            {
                "token": token,
                "count": count,
                "share_of_corpus": round(count / total_tokens, 4),
            }
            for token, count in token_counts.most_common()
            if count / total_tokens >= 0.02
        ],
        "document_token_shares": sorted(
            document_shares,
            key=lambda document: document["share_of_corpus"],
            reverse=True,
        ),
        "target_term_component_coverage": {
            "covered_terms": len(covered_terms),
            "total_terms": len(target_terms),
            "coverage_rate": round(len(covered_terms) / len(target_terms), 4),
            "missing_terms": missing_terms,
            "by_priority": {
                priority: {
                    "covered_terms": priority_covered[priority],
                    "total_terms": total,
                    "coverage_rate": round(priority_covered[priority] / total, 4),
                }
                for priority, total in sorted(priority_totals.items())
            },
            "by_category": {
                category: {
                    "covered_terms": category_covered[category],
                    "total_terms": total,
                    "coverage_rate": round(category_covered[category] / total, 4),
                }
                for category, total in sorted(category_totals.items())
            },
            "note": (
                "Coverage means every token component exists in the vocabulary. "
                "It is not phrase-token coverage."
            ),
        },
        "warnings": warnings,
        "recommended_next_step": (
            "Review high-frequency tokens and missing core terms, then add phrase "
            "detection before training Word2Vec."
        ),
    }


def evaluate_saved_corpus(
    corpus_directory: str | Path = PROCESSED_DATA_DIR,
    target_terms_path: str | Path = METADATA_DIR / "target_terms.csv",
) -> dict[str, object]:
    """Evaluate the corpus files created by :mod:`training.corpus_builder`."""

    directory = Path(corpus_directory)
    return evaluate_corpus_quality(
        sentences=load_tokenized_sentences(directory / "sentences.txt"),
        target_terms=load_target_terms(target_terms_path),
        corpus_metadata=load_corpus_metadata(directory / "corpus_report.json"),
    )


def save_quality_report(
    report: dict[str, object],
    corpus_directory: str | Path = PROCESSED_DATA_DIR,
) -> Path:
    """Save the quality report next to the local processed corpus files."""

    output_path = Path(corpus_directory) / "quality_report.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    return output_path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options for local corpus evaluation."""

    parser = argparse.ArgumentParser(
        description="Evaluate the quality of a saved financial embedding corpus."
    )
    parser.add_argument(
        "--corpus-dir",
        default=str(PROCESSED_DATA_DIR),
        help="Directory containing sentences.txt and corpus_report.json.",
    )
    parser.add_argument(
        "--target-terms",
        default=str(METADATA_DIR / "target_terms.csv"),
        help="Path to the financial terminology taxonomy CSV.",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate and save the local corpus-quality report."""

    arguments = parse_arguments()
    report = evaluate_saved_corpus(arguments.corpus_dir, arguments.target_terms)
    output_path = save_quality_report(report, arguments.corpus_dir)
    print(json.dumps(report, indent=2))
    print(f"Saved quality report: {output_path}")


if __name__ == "__main__":
    main()
