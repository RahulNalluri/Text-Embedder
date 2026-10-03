"""Build the complete, versioned corpus package from approved Version 3 PDFs."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .config import (
    METADATA_DIR,
    VERSION_3_APPROVED_DIR,
    VERSION_3_PROCESSED_DIR,
)
from .corpus_builder import build_corpus, file_sha256, save_corpus
from .corpus_quality import evaluate_saved_corpus, save_quality_report
from .dataset_inventory import build_approved_inventory, save_inventory
from .phrase_detection import build_phrase_corpus, save_phrase_corpus


VERSION_3_OUTPUT_FILES = (
    "dataset_inventory.json",
    "sentences.txt",
    "vocabulary_counts.csv",
    "corpus_report.json",
    "quality_report.json",
    "sentences_phrased.txt",
    "phrase_counts.csv",
    "phrase_report.json",
)


def write_version3_build_summary(directory: str | Path) -> Path:
    """Recompute the auditable summary and output hashes for a built corpus."""

    path = Path(directory)
    with (path / "dataset_inventory.json").open(encoding="utf-8") as file:
        inventory = json.load(file)
    with (path / "corpus_report.json").open(encoding="utf-8") as file:
        corpus_report = json.load(file)
    with (path / "quality_report.json").open(encoding="utf-8") as file:
        quality_report = json.load(file)
    with (path / "phrase_report.json").open(encoding="utf-8") as file:
        phrase_report = json.load(file)

    summary = {
        "schema_version": 1,
        "dataset_sha256": inventory["dataset_sha256"],
        "report_count": inventory["report_count"],
        "base_sentence_count": corpus_report["sentence_count"],
        "base_token_count": corpus_report["token_count"],
        "phrase_token_count": phrase_report["output_token_count"],
        "base_vocabulary_size": corpus_report["vocabulary_size"],
        "phrase_vocabulary_size": phrase_report["output_vocabulary_size"],
        "quality_warnings": quality_report["warnings"],
        "output_sha256": {
            filename: file_sha256(path / filename)
            for filename in VERSION_3_OUTPUT_FILES
        },
    }
    summary_path = path / "version3_build_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary_path


def run_version3_pipeline(
    manifest_path: str | Path = METADATA_DIR / "companies.csv",
    approved_directory: str | Path = VERSION_3_APPROVED_DIR,
    output_directory: str | Path = VERSION_3_PROCESSED_DIR,
    expected_report_count: int = 20,
    minimum_tokens: int = 3,
    minimum_phrase_count: int = 5,
) -> Path:
    """Build inventory, base corpus, quality report and phrase corpus atomically."""

    output_path = Path(output_directory)
    if output_path.exists():
        raise FileExistsError(
            f"Version 3 output already exists; preserve or remove it first: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    inventory = build_approved_inventory(
        manifest_path=manifest_path,
        approved_directory=approved_directory,
        expected_report_count=expected_report_count,
    )
    metadata = {
        report.source_filename: {
            "company_name": report.company_name,
            "sector": report.sector,
            "financial_year": report.financial_year,
        }
        for report in inventory.reports
    }

    with tempfile.TemporaryDirectory(
        prefix="version_3_build_",
        dir=output_path.parent,
    ) as temporary_directory:
        staging_path = Path(temporary_directory)
        save_inventory(inventory, staging_path / "dataset_inventory.json")

        corpus = build_corpus(
            (report.source_path for report in inventory.reports),
            minimum_tokens=minimum_tokens,
            document_metadata=metadata,
        )
        if corpus.skipped_documents or corpus.failed_documents:
            raise RuntimeError(
                "Version 3 corpus build requires every frozen report to process "
                "successfully without duplicate-file skips."
            )
        save_corpus(corpus, staging_path)

        quality_report = evaluate_saved_corpus(staging_path)
        save_quality_report(quality_report, staging_path)

        phrase_result = build_phrase_corpus(
            corpus_directory=staging_path,
            minimum_count=minimum_phrase_count,
        )
        save_phrase_corpus(phrase_result, staging_path)

        write_version3_build_summary(staging_path)

        staging_path.replace(output_path)

    return output_path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the frozen Version 3 financial embedding corpus."
    )
    parser.add_argument(
        "--manifest",
        default=str(METADATA_DIR / "companies.csv"),
    )
    parser.add_argument(
        "--approved-dir",
        default=str(VERSION_3_APPROVED_DIR),
    )
    parser.add_argument(
        "--output-dir",
        default=str(VERSION_3_PROCESSED_DIR),
    )
    parser.add_argument("--expected-count", type=int, default=20)
    parser.add_argument("--minimum-tokens", type=int, default=3)
    parser.add_argument("--minimum-phrase-count", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    output_path = run_version3_pipeline(
        manifest_path=arguments.manifest,
        approved_directory=arguments.approved_dir,
        output_directory=arguments.output_dir,
        expected_report_count=arguments.expected_count,
        minimum_tokens=arguments.minimum_tokens,
        minimum_phrase_count=arguments.minimum_phrase_count,
    )
    summary_path = output_path / "version3_build_summary.json"
    print(summary_path.read_text(encoding="utf-8"))
    print(f"Saved Version 3 corpus package: {output_path}")


if __name__ == "__main__":
    main()
