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
                        {"source_filename": "first.pdf", "token_count": 4},
                        {"source_filename": "second.pdf", "token_count": 3},
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

    def test_saves_quality_report(self):
        report = evaluate_saved_corpus(self.directory, self.target_terms_path)
        output_path = save_quality_report(report, self.directory)

        self.assertTrue(output_path.exists())
        saved_report = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(saved_report["sentence_count"], 3)


if __name__ == "__main__":
    unittest.main()
