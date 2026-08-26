"""Train and save the project-owned Indian financial Word2Vec model."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import gensim
import numpy as np
from gensim.models import KeyedVectors, Word2Vec

from .config import ARTIFACTS_DIR, DEFAULT_CONFIG, PROCESSED_DATA_DIR, TrainingConfig


FINAL_LEARNING_RATE = 0.0001


class SentenceCorpus:
    """Re-iterable, memory-efficient reader for the phrase-aware corpus."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __iter__(self) -> Iterator[list[str]]:
        with self.path.open(encoding="utf-8") as file:
            for line in file:
                tokens = line.split()
                if tokens:
                    yield tokens


@dataclass(frozen=True, slots=True)
class CorpusStatistics:
    """Basic statistics collected before training begins."""

    sentence_count: int
    token_count: int
    raw_vocabulary_size: int

    def to_dict(self) -> dict[str, int]:
        return {
            "sentence_count": self.sentence_count,
            "token_count": self.token_count,
            "raw_vocabulary_size": self.raw_vocabulary_size,
        }


@dataclass(slots=True)
class TrainingResult:
    """Trained model plus the information needed for its audit report."""

    model: Word2Vec
    config: TrainingConfig
    corpus_path: Path
    corpus_statistics: CorpusStatistics
    training_seconds: float
    effective_training_words: int
    total_training_words: int

    def to_report_dict(self) -> dict[str, object]:
        validation = validate_model(self.model, self.config.embedding_dimension)
        return {
            "model_name": "indian_financial_word2vec",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "library": {"gensim": gensim.__version__, "numpy": np.__version__},
            "configuration": self.config.to_dict(),
            "corpus": {
                "source_filename": self.corpus_path.name,
                "sha256": file_sha256(self.corpus_path),
                **self.corpus_statistics.to_dict(),
            },
            "training": {
                "training_seconds": round(self.training_seconds, 3),
                "effective_training_words": self.effective_training_words,
                "total_training_words": self.total_training_words,
                "retained_vocabulary_size": len(self.model.wv),
                "discarded_vocabulary_size": (
                    self.corpus_statistics.raw_vocabulary_size - len(self.model.wv)
                ),
                "final_learning_rate": FINAL_LEARNING_RATE,
            },
            "validation": validation,
            "reproducibility": {
                "single_worker": self.config.workers == 1,
                "stable_token_hash": True,
                "random_seed": self.config.random_seed,
            },
        }


def stable_token_hash(token: str) -> int:
    """Return a stable integer hash for reproducible vector initialization."""

    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def file_sha256(path: str | Path) -> str:
    """Calculate a file digest without loading the complete file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_corpus(path: str | Path) -> CorpusStatistics:
    """Validate a tokenized corpus and return its basic statistics."""

    corpus_path = Path(path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Training corpus does not exist: {corpus_path}")
    if not corpus_path.is_file():
        raise ValueError(f"Training corpus path is not a file: {corpus_path}")

    sentence_count = 0
    token_count = 0
    vocabulary: set[str] = set()
    for sentence in SentenceCorpus(corpus_path):
        sentence_count += 1
        token_count += len(sentence)
        vocabulary.update(sentence)

    if sentence_count == 0:
        raise ValueError("Training corpus contains no tokenized sentences.")

    return CorpusStatistics(sentence_count, token_count, len(vocabulary))


def validate_model(model: Word2Vec, expected_dimension: int) -> dict[str, object]:
    """Fail if the trained vocabulary or embedding matrix is invalid."""

    vocabulary_size = len(model.wv)
    if vocabulary_size == 0:
        raise ValueError("The trained model has an empty vocabulary.")

    expected_shape = (vocabulary_size, expected_dimension)
    if model.wv.vectors.shape != expected_shape:
        raise ValueError(
            f"Expected embedding matrix shape {expected_shape}, "
            f"received {model.wv.vectors.shape}."
        )
    if not np.isfinite(model.wv.vectors).all():
        raise ValueError("The embedding matrix contains NaN or infinite values.")

    vector_norms = np.linalg.norm(model.wv.vectors, axis=1)
    if np.any(vector_norms == 0):
        raise ValueError("One or more learned embedding vectors have zero length.")

    return {
        "passed": True,
        "vocabulary_non_empty": True,
        "embedding_matrix_shape": list(model.wv.vectors.shape),
        "all_values_finite": True,
        "all_vectors_nonzero": True,
    }


def train_word2vec(
    corpus_path: str | Path,
    config: TrainingConfig = DEFAULT_CONFIG,
) -> TrainingResult:
    """Train a Word2Vec model from a phrase-aware tokenized corpus."""

    resolved_path = Path(corpus_path).resolve()
    statistics = inspect_corpus(resolved_path)
    corpus = SentenceCorpus(resolved_path)
    architecture_flag = 1 if config.architecture == "skip_gram" else 0

    model = Word2Vec(
        vector_size=config.embedding_dimension,
        window=config.context_window,
        min_count=config.minimum_word_frequency,
        sg=architecture_flag,
        hs=0,
        negative=config.negative_samples,
        alpha=config.learning_rate,
        min_alpha=FINAL_LEARNING_RATE,
        seed=config.random_seed,
        workers=config.workers,
        hashfxn=stable_token_hash,
    )
    model.build_vocab(corpus_iterable=corpus)
    if len(model.wv) == 0:
        raise ValueError(
            "No tokens met the configured minimum_word_frequency of "
            f"{config.minimum_word_frequency}."
        )

    started_at = time.perf_counter()
    effective_words, total_words = model.train(
        corpus_iterable=corpus,
        total_examples=model.corpus_count,
        epochs=config.epochs,
        start_alpha=config.learning_rate,
        end_alpha=FINAL_LEARNING_RATE,
    )
    elapsed = time.perf_counter() - started_at
    validate_model(model, config.embedding_dimension)

    return TrainingResult(
        model=model,
        config=config,
        corpus_path=resolved_path,
        corpus_statistics=statistics,
        training_seconds=elapsed,
        effective_training_words=effective_words,
        total_training_words=total_words,
    )


def save_training_artifacts(
    result: TrainingResult,
    output_directory: str | Path = ARTIFACTS_DIR,
) -> dict[str, Path]:
    """Atomically save the full model, query vectors, and training report."""

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "word2vec.model"
    vectors_path = directory / "vectors.kv"
    report_path = directory / "training_report.json"
    temporary_model_path = directory / "word2vec.model.tmp"
    temporary_vectors_path = directory / "vectors.kv.tmp"
    temporary_report_path = directory / "training_report.json.tmp"

    try:
        result.model.save(str(temporary_model_path))
        result.model.wv.save(str(temporary_vectors_path))

        reloaded_model = Word2Vec.load(str(temporary_model_path))
        reloaded_vectors = KeyedVectors.load(str(temporary_vectors_path), mmap="r")
        validate_model(reloaded_model, result.config.embedding_dimension)
        if reloaded_vectors.vectors.shape != result.model.wv.vectors.shape:
            raise ValueError("Reloaded query vectors have an unexpected shape.")
        if not np.isfinite(reloaded_vectors.vectors).all():
            raise ValueError("Reloaded query vectors contain invalid values.")

        report = result.to_report_dict()
        report["artifacts"] = {
            "word2vec_model": model_path.name,
            "query_vectors": vectors_path.name,
            "training_report": report_path.name,
        }
        with temporary_report_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)

        temporary_model_path.replace(model_path)
        temporary_vectors_path.replace(vectors_path)
        temporary_report_path.replace(report_path)
    finally:
        temporary_model_path.unlink(missing_ok=True)
        temporary_vectors_path.unlink(missing_ok=True)
        temporary_report_path.unlink(missing_ok=True)

    return {"model": model_path, "vectors": vectors_path, "report": report_path}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line paths for the baseline Word2Vec training run."""

    parser = argparse.ArgumentParser(
        description="Train the Indian financial Word2Vec model."
    )
    parser.add_argument(
        "--corpus",
        default=str(PROCESSED_DATA_DIR / "sentences_phrased.txt"),
        help="Path to the phrase-aware tokenized corpus.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ARTIFACTS_DIR),
        help="Directory for local trained-model artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    """Train, validate, and save the baseline model."""

    arguments = parse_arguments()
    result = train_word2vec(arguments.corpus)
    paths = save_training_artifacts(result, arguments.output_dir)
    print(paths["report"].read_text(encoding="utf-8"))
    print("Saved training artifacts:")
    for label, path in paths.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
