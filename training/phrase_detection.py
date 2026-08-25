"""Detect and join recurring financial phrases in the tokenized corpus."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .config import METADATA_DIR, PROCESSED_DATA_DIR
from .corpus_quality import load_target_terms, load_tokenized_sentences


@dataclass(frozen=True, slots=True)
class PhraseCandidate:
    """One approved multi-word financial term that may become one token."""

    term: str
    tokens: tuple[str, ...]
    category: str
    priority: str

    @property
    def joined_token(self) -> str:
        """Return the Word2Vec token used to represent this phrase."""

        return "_".join(self.tokens)


@dataclass(frozen=True, slots=True)
class PhraseDetectionResult:
    """Phrase-aware sentences and the audit data used to create them."""

    sentences: tuple[tuple[str, ...], ...]
    phrase_counts: Counter[str]
    candidates: tuple[PhraseCandidate, ...]
    detected_candidates: tuple[PhraseCandidate, ...]
    minimum_count: int
    input_token_count: int

    @property
    def output_token_count(self) -> int:
        return sum(len(sentence) for sentence in self.sentences)

    @property
    def vocabulary_size(self) -> int:
        return len({token for sentence in self.sentences for token in sentence})

    @property
    def joined_phrase_counts(self) -> Counter[str]:
        """Count phrase tokens actually written after longest-match replacement."""

        detected_tokens = {candidate.joined_token for candidate in self.detected_candidates}
        return Counter(
            token
            for sentence in self.sentences
            for token in sentence
            if token in detected_tokens
        )

    def to_report_dict(self) -> dict[str, object]:
        """Return a JSON-safe phrase-detection report."""

        detected_tokens = {candidate.joined_token for candidate in self.detected_candidates}
        joined_counts = self.joined_phrase_counts
        return {
            "sentence_count": len(self.sentences),
            "input_token_count": self.input_token_count,
            "output_token_count": self.output_token_count,
            "tokens_combined": self.input_token_count - self.output_token_count,
            "output_vocabulary_size": self.vocabulary_size,
            "minimum_phrase_count": self.minimum_count,
            "candidate_phrase_count": len(self.candidates),
            "detected_phrase_count": len(self.detected_candidates),
            "total_candidate_matches": sum(self.phrase_counts.values()),
            "total_joined_phrase_occurrences": sum(joined_counts.values()),
            "detected_phrases": [
                {
                    "term": candidate.term,
                    "token": candidate.joined_token,
                    "candidate_count": self.phrase_counts[candidate.joined_token],
                    "joined_count": joined_counts[candidate.joined_token],
                    "category": candidate.category,
                    "priority": candidate.priority,
                }
                for candidate in sorted(
                    self.detected_candidates,
                    key=lambda item: (-joined_counts[item.joined_token], item.term),
                )
            ],
            "undetected_phrases": [
                candidate.term
                for candidate in self.candidates
                if candidate.joined_token not in detected_tokens
            ],
            "method": (
                "Approved multi-word terms from target_terms.csv are counted as "
                "contiguous token sequences. Terms meeting the minimum count are "
                "joined with underscores using longest-match-first replacement."
            ),
        }


def build_phrase_candidates(target_terms_path: str | Path) -> list[PhraseCandidate]:
    """Load unique, tokenizable multi-word terms from the financial taxonomy."""

    candidates: list[PhraseCandidate] = []
    seen_tokens: set[tuple[str, ...]] = set()
    for target in load_target_terms(target_terms_path):
        tokens = target.component_tokens
        # Exact digits are normalized to <number> in the base corpus. Excluding
        # these candidates prevents "scope 1" and "scope 2" from collapsing
        # into the same, incorrectly labelled phrase token.
        if len(tokens) < 2 or "<number>" in tokens or tokens in seen_tokens:
            continue
        seen_tokens.add(tokens)
        candidates.append(
            PhraseCandidate(
                term=target.term,
                tokens=tokens,
                category=target.category,
                priority=target.priority,
            )
        )
    return candidates


def count_candidate_phrases(
    sentences: Iterable[Sequence[str]],
    candidates: Iterable[PhraseCandidate],
) -> Counter[str]:
    """Count every contiguous occurrence of each candidate phrase."""

    candidates_by_first_token: dict[str, list[PhraseCandidate]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_first_token[candidate.tokens[0]].append(candidate)

    counts: Counter[str] = Counter()
    for sentence in sentences:
        for start, token in enumerate(sentence):
            for candidate in candidates_by_first_token.get(token, ()):
                end = start + len(candidate.tokens)
                if tuple(sentence[start:end]) == candidate.tokens:
                    counts[candidate.joined_token] += 1
    return counts


def join_detected_phrases(
    sentence: Sequence[str],
    detected_candidates: Iterable[PhraseCandidate],
) -> list[str]:
    """Join detected phrases, preferring the longest match at each position."""

    candidates_by_first_token: dict[str, list[PhraseCandidate]] = defaultdict(list)
    for candidate in detected_candidates:
        candidates_by_first_token[candidate.tokens[0]].append(candidate)
    for candidates in candidates_by_first_token.values():
        candidates.sort(key=lambda candidate: len(candidate.tokens), reverse=True)

    output: list[str] = []
    position = 0
    while position < len(sentence):
        match = next(
            (
                candidate
                for candidate in candidates_by_first_token.get(sentence[position], ())
                if tuple(sentence[position : position + len(candidate.tokens)])
                == candidate.tokens
            ),
            None,
        )
        if match is None:
            output.append(sentence[position])
            position += 1
        else:
            output.append(match.joined_token)
            position += len(match.tokens)
    return output


def detect_phrases(
    sentences: Iterable[Sequence[str]],
    candidates: Iterable[PhraseCandidate],
    minimum_count: int = 5,
) -> PhraseDetectionResult:
    """Select recurring candidates and produce a phrase-aware corpus."""

    if minimum_count < 1:
        raise ValueError("minimum_count must be at least 1.")

    normalized_sentences = tuple(tuple(sentence) for sentence in sentences)
    if not normalized_sentences:
        raise ValueError("Cannot detect phrases in an empty corpus.")

    normalized_candidates = tuple(candidates)
    phrase_counts = count_candidate_phrases(normalized_sentences, normalized_candidates)
    detected_candidates = tuple(
        candidate
        for candidate in normalized_candidates
        if phrase_counts[candidate.joined_token] >= minimum_count
    )
    transformed_sentences = tuple(
        tuple(join_detected_phrases(sentence, detected_candidates))
        for sentence in normalized_sentences
    )

    return PhraseDetectionResult(
        sentences=transformed_sentences,
        phrase_counts=phrase_counts,
        candidates=normalized_candidates,
        detected_candidates=detected_candidates,
        minimum_count=minimum_count,
        input_token_count=sum(len(sentence) for sentence in normalized_sentences),
    )


def build_phrase_corpus(
    corpus_directory: str | Path = PROCESSED_DATA_DIR,
    target_terms_path: str | Path = METADATA_DIR / "target_terms.csv",
    minimum_count: int = 5,
) -> PhraseDetectionResult:
    """Build phrase-aware sentences from the saved base corpus."""

    directory = Path(corpus_directory)
    sentences = load_tokenized_sentences(directory / "sentences.txt")
    candidates = build_phrase_candidates(target_terms_path)
    return detect_phrases(sentences, candidates, minimum_count)


def save_phrase_corpus(
    result: PhraseDetectionResult,
    output_directory: str | Path = PROCESSED_DATA_DIR,
) -> dict[str, Path]:
    """Save phrase-aware sentences, counts, and an audit report."""

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    sentences_path = directory / "sentences_phrased.txt"
    counts_path = directory / "phrase_counts.csv"
    report_path = directory / "phrase_report.json"

    with sentences_path.open("w", encoding="utf-8", newline="\n") as file:
        for sentence in result.sentences:
            file.write(" ".join(sentence))
            file.write("\n")

    candidate_by_token = {
        candidate.joined_token: candidate for candidate in result.candidates
    }
    with counts_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "term",
                "phrase_token",
                "candidate_count",
                "joined_count",
                "detected",
                "category",
                "priority",
            ]
        )
        joined_counts = result.joined_phrase_counts
        for token, count in result.phrase_counts.most_common():
            candidate = candidate_by_token[token]
            writer.writerow(
                [
                    candidate.term,
                    token,
                    count,
                    joined_counts[token],
                    count >= result.minimum_count,
                    candidate.category,
                    candidate.priority,
                ]
            )

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(result.to_report_dict(), file, indent=2)

    return {"sentences": sentences_path, "counts": counts_path, "report": report_path}


def parse_arguments() -> argparse.Namespace:
    """Parse phrase-corpus command-line options."""

    parser = argparse.ArgumentParser(
        description="Join recurring approved financial phrases in a saved corpus."
    )
    parser.add_argument(
        "--corpus-dir",
        default=str(PROCESSED_DATA_DIR),
        help="Directory containing sentences.txt.",
    )
    parser.add_argument(
        "--target-terms",
        default=str(METADATA_DIR / "target_terms.csv"),
        help="Path to the approved financial terminology taxonomy.",
    )
    parser.add_argument(
        "--minimum-count",
        type=int,
        default=5,
        help="Minimum corpus occurrences required to join a phrase.",
    )
    return parser.parse_args()


def main() -> None:
    """Create and save the phrase-aware training corpus."""

    arguments = parse_arguments()
    result = build_phrase_corpus(
        corpus_directory=arguments.corpus_dir,
        target_terms_path=arguments.target_terms,
        minimum_count=arguments.minimum_count,
    )
    paths = save_phrase_corpus(result, arguments.corpus_dir)
    print(json.dumps(result.to_report_dict(), indent=2))
    print("Saved phrase corpus files:")
    for label, path in paths.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
