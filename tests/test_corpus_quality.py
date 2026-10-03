import csv
import json
import tempfile
import unittest
from pathlib import Path

from training.corpus_quality import (
    evaluate_saved_corpus,
    load_target_terms,
    save_quality_report,
)


class TestCorpusQuality(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        (self.directory / "sentences.txt").write_text(
            "revenue increased\ncash flow improved\nrevenue growth\n",
            encoding="utf-8",
        )
        (self.directory / "corpus_report.json").write_text(
            json.dumps(
                {
                    "documents_processed": 2,
                    "documents_failed": 0,
                    "documents": [
                        {
                            "source_filename": "first.pdf",
                            "token_count": 4,
                            "company_name": "First Company",
                            "sector": "banking",
                            "financial_year": "FY2024-25",
                        },
                        {
                            "source_filename": "second.pdf",
                            "token_count": 3,
                            "company_name": "Second Company",
                            "sector": "technology",
                            "financial_year": "FY2025-26",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.target_terms_path = self.directory / "target_terms.csv"
        with self.target_terms_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["term", "category", "priority"])
            writer.writeheader()
            writer.writerows(
                [
                    {"term": "revenue", "category": "performance", "priority": "core"},
                    {"term": "cash flow", "category": "liquidity", "priority": "core"},
                    {"term": "borrowings", "category": "liquidity", "priority": "supplemental"},
                ]
            )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_evaluates_saved_corpus_and_component_coverage(self):
        report = evaluate_saved_corpus(self.directory, self.target_terms_path)

        self.assertEqual(report["token_count"], 7)
        self.assertEqual(report["vocabulary_size"], 6)
        self.assertEqual(
            report["target_term_component_coverage"]["covered_terms"], 2
        )
        self.assertEqual(
            report["target_term_component_coverage"]["missing_terms"],
            ["borrowings"],
        )
        self.assertEqual(report["document_token_shares"][0]["source_filename"], "first.pdf")
        self.assertEqual(report["company_token_shares"][0]["name"], "First Company")
        self.assertEqual(report["sector_token_shares"][0]["name"], "banking")
        self.assertEqual(
            report["exact_duplicate_sentences"]["repeated_occurrences"], 0
        )

    def test_saves_quality_report(self):
        report = evaluate_saved_corpus(self.directory, self.target_terms_path)
        output_path = save_quality_report(report, self.directory)

        self.assertTrue(output_path.exists())
        saved_report = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(saved_report["sentence_count"], 3)

    def test_warns_when_exact_repeated_sentences_exceed_ten_percent(self):
        sentences = [["revenue", "increased"]] * 3 + [["cash", "improved"]]
        (self.directory / "sentences.txt").write_text(
            "\n".join(" ".join(sentence) for sentence in sentences) + "\n",
            encoding="utf-8",
        )

        report = evaluate_saved_corpus(self.directory, self.target_terms_path)

        self.assertIn(
            "Exact repeated sentences exceed 10% of the corpus; review "
            "annual-report boilerplate before training.",
            report["warnings"],
        )


if __name__ == "__main__":
    unittest.main()
