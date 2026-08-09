import csv
import unittest
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"


def read_csv(filename: str) -> list[dict[str, str]]:
    with (METADATA_DIR / filename).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


class TestMetadataFiles(unittest.TestCase):
    def test_target_term_taxonomy_is_valid(self):
        rows = read_csv("target_terms.csv")
        terms = [row["term"].strip().lower() for row in rows]

        self.assertGreaterEqual(len(rows), 100)
        self.assertEqual(len(terms), len(set(terms)))
        self.assertTrue(all(row["category"].strip() for row in rows))
        self.assertTrue(
            all(row["priority"] in {"core", "sector", "supplemental"} for row in rows)
        )

    def test_public_company_manifest_template_has_expected_columns(self):
        rows = read_csv("companies.example.csv")
        expected_columns = {
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
        }

        self.assertEqual(set(rows[0]), expected_columns)
        self.assertIn("Example Company", rows[0]["company_name"])

    def test_local_company_manifest_has_balanced_pilot_coverage(self):
        if not (METADATA_DIR / "companies.csv").exists():
            return

        rows = read_csv("companies.csv")
        sector_counts = Counter(row["sector"] for row in rows)

        self.assertEqual(len(rows), 12)
        self.assertEqual(len(sector_counts), 6)
        self.assertTrue(all(count == 2 for count in sector_counts.values()))
        self.assertEqual(
            {row["financial_year"] for row in rows}, {"FY2023-24", "FY2024-25"}
        )
        self.assertTrue(all(row["official_ir_url"].startswith("https://") for row in rows))
        self.assertTrue(all(row["source_status"] == "official_page_verified" for row in rows))
        self.assertTrue(all(row["permission_status"] == "review_required" for row in rows))


if __name__ == "__main__":
    unittest.main()
