"""Evaluate technical and semantic quality of saved financial embeddings."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np
from gensim.models import KeyedVectors

from .config import ARTIFACTS_DIR, METADATA_DIR
from .corpus_quality import TargetTerm, load_target_terms


DEFAULT_NEIGHBOUR_TERMS = (
    "revenue", "profit", "liquidity", "dividend", "net_profit",
    "credit_risk", "working_capital", "fair_value",
    "cash_and_cash_equivalents", "operating_profit",
)


@dataclass(frozen=True, slots=True)
class EvaluationPair:
    """A model-token pair with an expected semantic relationship."""

    first: str
    second: str
    relationship: str

    def __post_init__(self) -> None:
        if self.relationship not in {"related", "unrelated"}:
            raise ValueError("relationship must be 'related' or 'unrelated'.")


DEFAULT_EVALUATION_PAIRS = (
    EvaluationPair("net_profit", "profit_after_tax", "related"),
    EvaluationPair("credit_risk", "market_risk", "related"),
    EvaluationPair("cash", "liquidity", "related"),
    EvaluationPair("dividend", "shareholder", "related"),
    EvaluationPair("revenue", "profit", "related"),
    EvaluationPair("working_capital", "liquidity", "related"),
    EvaluationPair("fair_value", "amortised_cost", "related"),
    EvaluationPair("dividend", "cybersecurity", "unrelated"),
    EvaluationPair("loan", "employee", "unrelated"),
    EvaluationPair("revenue", "water_consumption", "unrelated"),
    EvaluationPair("credit_risk", "dividend", "unrelated"),
    EvaluationPair("fair_value", "employee", "unrelated"),
    EvaluationPair("net_profit", "cybersecurity", "unrelated"),
)


def target_model_token(target: TargetTerm) -> str | None:
    """Map taxonomy text to its expected phrase-aware model token."""

    components = target.component_tokens
    if not components or "<number>" in components:
        return None
    return components[0] if len(components) == 1 else "_".join(components)


def evaluate_technical_validity(
    vectors: KeyedVectors,
    training_report: dict[str, object],
) -> dict[str, object]:
    """Compare vector properties with the recorded training metadata."""

    configuration = training_report.get("configuration", {})
    training = training_report.get("training", {})
    expected_dimension = int(configuration.get("embedding_dimension", 0)) if isinstance(configuration, dict) else 0
    expected_vocabulary = int(training.get("retained_vocabulary_size", 0)) if isinstance(training, dict) else 0
    checks = {
        "vocabulary_non_empty": len(vectors) > 0,
        "values_finite": bool(np.isfinite(vectors.vectors).all()),
        "dimension_matches_training_report": vectors.vector_size == expected_dimension,
        "vocabulary_matches_training_report": len(vectors) == expected_vocabulary,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "actual_vocabulary_size": len(vectors),
        "actual_vector_size": vectors.vector_size,
        "embedding_matrix_shape": list(vectors.vectors.shape),
    }


def _coverage_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    evaluable = [row for row in rows if row["model_token"] is not None]
    covered = [row for row in evaluable if row["present"]]
    return {
        "covered_terms": len(covered),
        "evaluable_terms": len(evaluable),
        "coverage_rate": round(len(covered) / len(evaluable), 4) if evaluable else 0.0,
    }


def evaluate_target_coverage(
    vectors: KeyedVectors,
    target_terms: list[TargetTerm],
) -> dict[str, object]:
    """Measure exact single-token and phrase-token vocabulary coverage."""

    rows: list[dict[str, object]] = []
    for target in target_terms:
        model_token = target_model_token(target)
        rows.append({
            "term": target.term,
            "model_token": model_token,
            "category": target.category,
            "priority": target.priority,
            "is_phrase": len(target.component_tokens) > 1,
            "present": model_token in vectors if model_token is not None else False,
        })

    evaluable = [row for row in rows if row["model_token"] is not None]
    priorities = sorted({str(row["priority"]) for row in evaluable})
    categories = sorted({str(row["category"]) for row in evaluable})
    return {
        "overall": _coverage_summary(rows),
        "single_terms": _coverage_summary([row for row in rows if not row["is_phrase"]]),
        "phrase_terms": _coverage_summary([row for row in rows if row["is_phrase"]]),
        "by_priority": {name: _coverage_summary([row for row in rows if row["priority"] == name]) for name in priorities},
        "by_category": {name: _coverage_summary([row for row in rows if row["category"] == name]) for name in categories},
        "missing_terms": [row for row in evaluable if not row["present"]],
        "not_evaluable_terms": [row["term"] for row in rows if row["model_token"] is None],
        "note": (
            "Multi-word terms require their exact underscore-joined token. "
            "Numbered terms are excluded because exact numbers were normalized."
        ),
    }


def evaluate_similarity_pairs(
    vectors: KeyedVectors,
    pairs: tuple[EvaluationPair, ...] = DEFAULT_EVALUATION_PAIRS,
) -> dict[str, object]:
    """Compare curated related similarities with unrelated controls."""

    evaluated: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    for pair in pairs:
        missing = [token for token in (pair.first, pair.second) if token not in vectors]
        if missing:
            skipped.append({
                "first": pair.first,
                "second": pair.second,
                "relationship": pair.relationship,
                "reason": f"missing vocabulary: {', '.join(missing)}",
            })
            continue
        evaluated.append({
            "first": pair.first,
            "second": pair.second,
            "relationship": pair.relationship,
            "cosine_similarity": round(float(vectors.similarity(pair.first, pair.second)), 4),
        })

    related = [float(row["cosine_similarity"]) for row in evaluated if row["relationship"] == "related"]
    unrelated = [float(row["cosine_similarity"]) for row in evaluated if row["relationship"] == "unrelated"]
    related_average = mean(related) if related else None
    unrelated_average = mean(unrelated) if unrelated else None
    margin = related_average - unrelated_average if related_average is not None and unrelated_average is not None else None
    comparisons = [positive > negative for positive in related for negative in unrelated]
    accuracy = sum(comparisons) / len(comparisons) if comparisons else None
    return {
        "evaluated_pairs": evaluated,
        "skipped_pairs": skipped,
        "related_average": round(related_average, 4) if related_average is not None else None,
        "unrelated_average": round(unrelated_average, 4) if unrelated_average is not None else None,
        "average_margin": round(margin, 4) if margin is not None else None,
        "pairwise_separation_accuracy": round(accuracy, 4) if accuracy is not None else None,
        "note": "This small diagnostic benchmark is not a universal measure of financial correctness.",
    }


def evaluate_nearest_neighbours(
    vectors: KeyedVectors,
    terms: tuple[str, ...] = DEFAULT_NEIGHBOUR_TERMS,
    topn: int = 8,
) -> dict[str, object]:
    """Record nearest neighbours for stable review terms."""

    if topn < 1:
        raise ValueError("topn must be at least 1.")
    available_topn = min(topn, max(len(vectors) - 1, 0))
    results: dict[str, object] = {}
    for term in terms:
        if term not in vectors:
            results[term] = {"present": False, "neighbours": []}
            continue
        neighbours = vectors.most_similar(term, topn=available_topn) if available_topn else []
        results[term] = {
            "present": True,
            "neighbours": [
                {"token": token, "cosine_similarity": round(float(score), 4)}
                for token, score in neighbours
            ],
        }
    return results


def find_suspicious_tokens(
    vectors: KeyedVectors,
    target_terms: list[TargetTerm],
    limit: int = 100,
) -> list[dict[str, str]]:
    """Find malformed tokens and one-character PDF truncations."""

    if limit < 1:
        raise ValueError("limit must be at least 1.")
    vocabulary = set(vectors.index_to_key)
    atomic_targets = {
        token for target in target_terms for token in target.component_tokens
        if token.isalpha() and len(token) >= 6
    }
    suspicious: dict[str, str] = {}
    allowed = re.compile(r"[a-z0-9_<>']+")
    for token in vocabulary:
        if not allowed.fullmatch(token):
            suspicious[token] = "unexpected character"
        elif token.startswith("_") or token.endswith("_") or "__" in token:
            suspicious[token] = "malformed phrase separator"
        elif len(token) > 80:
            suspicious[token] = "unusually long token"
    for target in atomic_targets:
        truncated = target[1:]
        if truncated in vocabulary and truncated not in atomic_targets:
            suspicious[truncated] = f"possible missing first character from '{target}'"
    return [{"token": token, "reason": suspicious[token]} for token in sorted(suspicious)[:limit]]


def evaluate_model(
    vectors: KeyedVectors,
    target_terms: list[TargetTerm],
    training_report: dict[str, object],
    pairs: tuple[EvaluationPair, ...] = DEFAULT_EVALUATION_PAIRS,
    neighbour_terms: tuple[str, ...] = DEFAULT_NEIGHBOUR_TERMS,
) -> dict[str, object]:
    """Return the complete baseline model evaluation report."""

    if not target_terms:
        raise ValueError("At least one target term is required for evaluation.")
    technical = evaluate_technical_validity(vectors, training_report)
    coverage = evaluate_target_coverage(vectors, target_terms)
    benchmark = evaluate_similarity_pairs(vectors, pairs)
    suspicious = find_suspicious_tokens(vectors, target_terms)
    warnings: list[str] = []
    if not technical["passed"]:
        warnings.append("Technical model metadata or vector validation failed.")
    if float(coverage["overall"]["coverage_rate"]) < 0.80:
        warnings.append("Exact target-token coverage is below 80%.")
    margin = benchmark["average_margin"]
    accuracy = benchmark["pairwise_separation_accuracy"]
    if margin is None:
        warnings.append("Too few benchmark pairs are available to calculate a margin.")
    elif float(margin) < 0.10:
        warnings.append("Related pairs average less than 0.10 above unrelated pairs.")
    if accuracy is not None and float(accuracy) < 0.70:
        warnings.append("Fewer than 70% of pair comparisons separate correctly.")
    if suspicious:
        warnings.append("Potential PDF extraction artifacts remain in the vocabulary.")
    return {
        "model_name": training_report.get("model_name", "unknown"),
        "technical_validation": technical,
        "target_coverage": coverage,
        "similarity_benchmark": benchmark,
        "nearest_neighbours": evaluate_nearest_neighbours(vectors, neighbour_terms),
        "suspicious_tokens": suspicious,
        "warnings": warnings,
        "manual_review_required": True,
        "recommended_next_step": (
            "Review missing targets, suspicious tokens, and neighbour quality; "
            "then improve preprocessing before adding reports or retraining."
        ),
    }


def load_training_report(path: str | Path) -> dict[str, object]:
    """Load saved training metadata."""

    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def evaluate_saved_model(
    artifact_directory: str | Path = ARTIFACTS_DIR,
    target_terms_path: str | Path = METADATA_DIR / "target_terms.csv",
) -> dict[str, object]:
    """Load and evaluate the locally saved baseline model."""

    directory = Path(artifact_directory)
    vectors = KeyedVectors.load(str(directory / "vectors.kv"), mmap="r")
    return evaluate_model(
        vectors,
        load_target_terms(target_terms_path),
        load_training_report(directory / "training_report.json"),
    )


def save_evaluation_report(
    report: dict[str, object],
    artifact_directory: str | Path = ARTIFACTS_DIR,
) -> Path:
    """Atomically save the evaluation beside local model artifacts."""

    directory = Path(artifact_directory)
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / "evaluation_report.json"
    temporary_path = directory / "evaluation_report.json.tmp"
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the saved Indian financial Word2Vec model.")
    parser.add_argument("--artifact-dir", default=str(ARTIFACTS_DIR))
    parser.add_argument("--target-terms", default=str(METADATA_DIR / "target_terms.csv"))
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    report = evaluate_saved_model(arguments.artifact_dir, arguments.target_terms)
    output_path = save_evaluation_report(report, arguments.artifact_dir)
    print(output_path.read_text(encoding="utf-8"))
    print(f"Saved evaluation report: {output_path}")


if __name__ == "__main__":
    main()
