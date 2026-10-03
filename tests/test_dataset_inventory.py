import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from training.dataset_inventory import build_approved_inventory, save_inventory


class TestDatasetInventory(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.approved_directory = self.directory / "approved"
        self.approved_directory.mkdir()
        self.report_path = self.approved_directory / "example_fy2024_25.pdf"
        self.report_path.write_bytes(b"validated annual report")
        self.checksum = hashlib.sha256(self.report_path.read_bytes()).hexdigest()
        self.manifest_path = self.directory / "companies.csv"
        self.fieldnames = [
            "company_name",
            "sector",
            "nse_symbol",
            "bse_code",
            "cin",
            "financial_year",
            "official_ir_url",
            "report_url",
            "local_filename",
            "source_status",
            "permission_status",
            "notes",
        ]
        self._write_manifest()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write_manifest(self, **overrides):
        row = {
            "company_name": "Example Limited",
            "sector": "example_sector",
            "nse_symbol": "EXAMPLE",
            "bse_code": "",
            "cin": "",
            "financial_year": "FY2024-25",
            "official_ir_url": "https://example.com/investors",
            "report_url": "",
            "local_filename": self.report_path.name,
            "source_status": "approved",
            "permission_status": "review_required",
            "notes": f"Version 3 approved - SHA-256 {self.checksum}",
        }
        row.update(overrides)
        with self.manifest_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerow(row)

    def test_builds_and_saves_deterministic_inventory(self):
        inventory = build_approved_inventory(
            self.manifest_path,
            self.approved_directory,
            expected_report_count=1,
        )

        self.assertEqual(len(inventory.reports), 1)
        self.assertEqual(inventory.reports[0].sha256, self.checksum)
        self.assertEqual(inventory.to_dict()["reports_by_sector"], {"example_sector": 1})

        output_path = save_inventory(inventory, self.directory / "inventory.json")
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["report_count"], 1)
        self.assertEqual(saved["dataset_sha256"], inventory.dataset_sha256)

    def test_rejects_checksum_mismatch(self):
        self._write_manifest(notes=f"Version 3 approved - SHA-256 {'0' * 64}")

        with self.assertRaisesRegex(ValueError, "checksum does not match"):
            build_approved_inventory(self.manifest_path, self.approved_directory)

    def test_rejects_unlisted_approved_pdf(self):
        (self.approved_directory / "other_fy2024_25.pdf").write_bytes(b"other")

        with self.assertRaisesRegex(ValueError, "unlisted"):
            build_approved_inventory(self.manifest_path, self.approved_directory)

    def test_rejects_filename_year_mismatch(self):
        self._write_manifest(financial_year="FY2025-26")

        with self.assertRaisesRegex(ValueError, "does not match financial year"):
            build_approved_inventory(self.manifest_path, self.approved_directory)


if __name__ == "__main__":
    unittest.main()
