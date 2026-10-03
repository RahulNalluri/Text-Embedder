"""Train the frozen Version 3 financial Word2Vec model reproducibly."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import (
    DEFAULT_CONFIG,
    VERSION_3_ARTIFACTS_DIR,
    VERSION_3_PROCESSED_DIR,
    TrainingConfig,
)
from .word2vec_trainer import (
    file_sha256,
    save_training_artifacts,
    train_word2vec,
)


AUDIT_FILENAMES = (
    "dataset_inventory.json",
    "corpus_report.json",
    "quality_report.json",
    "phrase_report.json",
    "version3_build_summary.json",
)
TRAINING_CORPUS_FILENAME = "sentences_phrased.txt"


@dataclass(frozen=True, slots=True)
class Version3Provenance:
    """Verified identity of the dataset and phrase corpus used for training."""

    dataset_sha256: str
    report_count: int
    corpus_sha256: str
    build_summary_sha256: str
    verified_output_sha256: dict[str, str]
    corpus_quality_warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_sha256": self.dataset_sha256,
            "report_count": self.report_count,
            "phrase_corpus_sha256": self.corpus_sha256,
            "build_summary_sha256": self.build_summary_sha256,
            "verified_output_sha256": self.verified_output_sha256,
            "corpus_quality_warnings": list(self.corpus_quality_warnings),
        }


def _load_json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required Version 3 file does not exist: {path}")
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def validate_version3_training_inputs(
    corpus_directory: str | Path = VERSION_3_PROCESSED_DIR,
    expected_report_count: int = 20,
) -> Version3Provenance:
    """Verify the frozen build summary and every recorded output checksum."""

    directory = Path(corpus_directory)
    summary_path = directory / "version3_build_summary.json"
    summary = _load_json_object(summary_path)

    report_count = int(summary.get("report_count", 0))
    if report_count != expected_report_count:
        raise ValueError(
            f"Expected {expected_report_count} Version 3 reports, found {report_count}."
        )

    dataset_sha256 = str(summary.get("dataset_sha256", ""))
    if len(dataset_sha256) != 64:
        raise ValueError("Version 3 summary contains an invalid dataset SHA-256.")

    recorded_hashes = summary.get("output_sha256")
    if not isinstance(recorded_hashes, dict) or not recorded_hashes:
        raise ValueError("Version 3 summary has no recorded output checksums.")
    if TRAINING_CORPUS_FILENAME not in recorded_hashes:
        raise ValueError("Version 3 summary does not identify the phrase corpus.")

    verified_hashes: dict[str, str] = {}
    for filename, expected_hash in sorted(recorded_hashes.items()):
        if not isinstance(filename, str) or not isinstance(expected_hash, str):
            raise ValueError("Version 3 output checksum entries must be strings.")
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"Recorded Version 3 output is missing: {path}")
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Checksum mismatch for {filename}: expected {expected_hash}, "
                f"calculated {actual_hash}."
            )
        verified_hashes[filename] = actual_hash

    warnings = summary.get("quality_warnings", [])
    if not isinstance(warnings, list) or not all(
        isinstance(warning, str) for warning in warnings
    ):
        raise ValueError("Version 3 quality warnings must be a list of strings.")

    return Version3Provenance(
        dataset_sha256=dataset_sha256,
        report_count=report_count,
        corpus_sha256=verified_hashes[TRAINING_CORPUS_FILENAME],
        build_summary_sha256=file_sha256(summary_path),
        verified_output_sha256=verified_hashes,
        corpus_quality_warnings=tuple(warnings),
    )


def run_version3_training(
    corpus_directory: str | Path = VERSION_3_PROCESSED_DIR,
    output_directory: str | Path = VERSION_3_ARTIFACTS_DIR,
    config: TrainingConfig = DEFAULT_CONFIG,
    expected_report_count: int = 20,
) -> Path:
    """Validate, train and publish a self-contained Version 3 model package."""

    corpus_path = Path(corpus_directory)
    output_path = Path(output_directory)
    if output_path.exists():
        raise FileExistsError(
            f"Version 3 artifacts already exist; preserve or rename them first: "
            f"{output_path}"
        )

    provenance = validate_version3_training_inputs(
        corpus_path,
        expected_report_count=expected_report_count,
    )
    for filename in AUDIT_FILENAMES:
        if not (corpus_path / filename).is_file():
            raise FileNotFoundError(
                f"Required Version 3 audit file does not exist: "
                f"{corpus_path / filename}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="version_3_training_",
        dir=output_path.parent,
    ) as temporary_directory:
        staging_path = Path(temporary_directory)
        result = train_word2vec(corpus_path / TRAINING_CORPUS_FILENAME, config)
        artifact_paths = save_training_artifacts(result, staging_path)

        for filename in AUDIT_FILENAMES:
            shutil.copy2(corpus_path / filename, staging_path / filename)

        report_path = artifact_paths["report"]
        report = _load_json_object(report_path)
        report["model_name"] = "indian_financial_word2vec_v3"
        report["model_version"] = 3
        report["dataset_provenance"] = provenance.to_dict()
        report["source_audit_files"] = list(AUDIT_FILENAMES)
        report["artifact_sha256"] = {
            artifact_paths["model"].name: file_sha256(artifact_paths["model"]),
            artifact_paths["vectors"].name: file_sha256(artifact_paths["vectors"]),
        }
        with report_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)

        staging_path.replace(output_path)

    return output_path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the frozen Version 3 Indian financial Word2Vec model."
    )
    parser.add_argument("--corpus-dir", default=str(VERSION_3_PROCESSED_DIR))
    parser.add_argument("--output-dir", default=str(VERSION_3_ARTIFACTS_DIR))
    parser.add_argument("--expected-count", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    output_path = run_version3_training(
        corpus_directory=arguments.corpus_dir,
        output_directory=arguments.output_dir,
        expected_report_count=arguments.expected_count,
    )
    report_path = output_path / "training_report.json"
    print(report_path.read_text(encoding="utf-8"))
    print(f"Saved Version 3 training package: {output_path}")


if __name__ == "__main__":
    main()
