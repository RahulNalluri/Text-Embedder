"""Freeze and validate the approved reports used by the Version 3 corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import (
    METADATA_DIR,
    VERSION_3_APPROVED_DIR,
    VERSION_3_PROCESSED_DIR,
)


FINANCIAL_YEAR_PATTERN = re.compile(r"FY(?P<start>\d{4})-(?P<end>\d{2})")
CHECKSUM_PATTERN = re.compile(r"SHA-256 (?P<digest>[0-9a-f]{64})")


def file_sha256(path: str | Path) -> str:
    """Return a file checksum without loading the whole report into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ApprovedReport:
    """One validated report selected for the Version 3 corpus."""

    company_name: str
    sector: str
    nse_symbol: str
    financial_year: str
    official_ir_url: str
    source_filename: str
    source_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class DatasetInventory:
    """A deterministic snapshot of the approved local training dataset."""

    reports: tuple[ApprovedReport, ...]
    dataset_sha256: str

    def to_dict(self) -> dict[str, object]:
        sector_counts = Counter(report.sector for report in self.reports)
        year_counts = Counter(report.financial_year for report in self.reports)
        return {
            "schema_version": 1,
            "report_count": len(self.reports),
            "sector_count": len(sector_counts),
            "financial_year_count": len(year_counts),
            "reports_by_sector": dict(sorted(sector_counts.items())),
            "reports_by_financial_year": dict(sorted(year_counts.items())),
            "dataset_sha256": self.dataset_sha256,
            "reports": [asdict(report) for report in self.reports],
        }


def _load_approved_manifest_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Company manifest does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as file:
        return [
            row
            for row in csv.DictReader(file)
            if row.get("source_status", "").strip() == "approved"
        ]


def _expected_year_suffix(financial_year: str) -> str:
    match = FINANCIAL_YEAR_PATTERN.fullmatch(financial_year)
    if match is None:
        raise ValueError(f"Invalid financial year: {financial_year}")
    return f"fy{match.group('start')}_{match.group('end')}"


def _dataset_digest(reports: list[ApprovedReport]) -> str:
    canonical_records = [
        {
            "company_name": report.company_name,
            "financial_year": report.financial_year,
            "sector": report.sector,
            "sha256": report.sha256,
            "source_filename": report.source_filename,
        }
        for report in reports
    ]
    serialized = json.dumps(
        canonical_records,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def build_approved_inventory(
    manifest_path: str | Path = METADATA_DIR / "companies.csv",
    approved_directory: str | Path = VERSION_3_APPROVED_DIR,
    expected_report_count: int | None = None,
) -> DatasetInventory:
    """Validate the private manifest against the approved PDF directory."""

    manifest = Path(manifest_path)
    approved_dir = Path(approved_directory)
    if not approved_dir.is_dir():
        raise FileNotFoundError(
            f"Approved report directory does not exist: {approved_dir}"
        )

    rows = _load_approved_manifest_rows(manifest)
    if not rows:
        raise ValueError("The manifest does not contain any approved reports.")
    if expected_report_count is not None and len(rows) != expected_report_count:
        raise ValueError(
            f"Expected {expected_report_count} approved reports but found {len(rows)}."
        )

    filenames = [row.get("local_filename", "").strip() for row in rows]
    if any(not filename for filename in filenames):
        raise ValueError("Every approved manifest row must have a local filename.")
    if len(filenames) != len(set(filenames)):
        raise ValueError("Approved manifest filenames must be unique.")

    manifest_files = set(filenames)
    directory_files = {path.name for path in approved_dir.glob("*.pdf")}
    missing = sorted(manifest_files - directory_files)
    unlisted = sorted(directory_files - manifest_files)
    if missing or unlisted:
        raise ValueError(
            "Approved manifest and directory differ: "
            f"missing={missing}, unlisted={unlisted}"
        )

    reports: list[ApprovedReport] = []
    for row in sorted(rows, key=lambda item: item["local_filename"]):
        filename = row["local_filename"].strip()
        financial_year = row["financial_year"].strip()
        expected_suffix = _expected_year_suffix(financial_year)
        if expected_suffix not in filename.lower():
            raise ValueError(
                f"Filename does not match financial year {financial_year}: {filename}"
            )

        path = (approved_dir / filename).resolve()
        checksum = file_sha256(path)
        checksum_match = CHECKSUM_PATTERN.search(row.get("notes", ""))
        if checksum_match is None:
            raise ValueError(f"Manifest checksum is missing for {filename}.")
        if checksum != checksum_match.group("digest"):
            raise ValueError(f"Manifest checksum does not match {filename}.")

        company_name = row["company_name"].strip()
        sector = row["sector"].strip()
        official_url = row["official_ir_url"].strip()
        if not company_name or not sector:
            raise ValueError(f"Company and sector are required for {filename}.")
        if not official_url.startswith("https://"):
            raise ValueError(f"An official HTTPS source is required for {filename}.")

        reports.append(
            ApprovedReport(
                company_name=company_name,
                sector=sector,
                nse_symbol=row["nse_symbol"].strip(),
                financial_year=financial_year,
                official_ir_url=official_url,
                source_filename=filename,
                source_path=str(path),
                size_bytes=path.stat().st_size,
                sha256=checksum,
            )
        )

    return DatasetInventory(
        reports=tuple(reports),
        dataset_sha256=_dataset_digest(reports),
    )


def save_inventory(
    inventory: DatasetInventory,
    output_path: str | Path = VERSION_3_PROCESSED_DIR / "dataset_inventory.json",
) -> Path:
    """Save the deterministic dataset snapshot used for corpus construction."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(inventory.to_dict(), file, indent=2)
    return path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and freeze the Version 3 approved-report inventory."
    )
    parser.add_argument(
        "--manifest",
        default=str(METADATA_DIR / "companies.csv"),
        help="Private company manifest containing approved report rows.",
    )
    parser.add_argument(
        "--approved-dir",
        default=str(VERSION_3_APPROVED_DIR),
        help="Directory containing only approved annual-report PDFs.",
    )
    parser.add_argument(
        "--output",
        default=str(VERSION_3_PROCESSED_DIR / "dataset_inventory.json"),
        help="Path for the generated inventory JSON.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=20,
        help="Required number of approved reports.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    inventory = build_approved_inventory(
        manifest_path=arguments.manifest,
        approved_directory=arguments.approved_dir,
        expected_report_count=arguments.expected_count,
    )
    output_path = save_inventory(inventory, arguments.output)
    print(json.dumps(inventory.to_dict(), indent=2))
    print(f"Saved dataset inventory: {output_path}")


if __name__ == "__main__":
    main()
