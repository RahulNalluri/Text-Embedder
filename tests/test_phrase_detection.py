import csv
import json
import tempfile
import unittest
from pathlib import Path

from training.phrase_detection import (
    PhraseCandidate,
    build_phrase_corpus,
    detect_phrases,
    join_detected_phrases,
    save_phrase_corpus,
)


class TestPhraseDetection(unittest.TestCase):
    def test_detects_only_candidates_meeting_minimum_count(self):
        sentences = [
            ["net", "profit", "and", "cash", "flow"],
            ["net", "profit", "increased"],
        ]
        candidates = [
            PhraseCandidate("net profit", ("net", "profit"), "performance", "core"),
            PhraseCandidate("cash flow", ("cash", "flow"), "liquidity", "core"),
        ]

        result = detect_phrases(sentences, candidates, minimum_count=2)

        self.assertEqual(
            result.sentences,
            (
                ("net_profit", "and", "cash", "flow"),
                ("net_profit", "increased"),
            ),
        )
        self.assertEqual(result.phrase_counts["net_profit"], 2)
        self.assertEqual(result.phrase_counts["cash_flow"], 1)

    def test_prefers_longest_overlapping_phrase(self):
        candidates = [
            PhraseCandidate("net profit", ("net", "profit"), "performance", "core"),
            PhraseCandidate(
                "net profit margin",
                ("net", "profit", "margin"),
                "performance",
                "core",
            ),
        ]

        transformed = join_detected_phrases(
            ["the", "net", "profit", "margin", "improved"], candidates
        )

        self.assertEqual(transformed, ["the", "net_profit_margin", "improved"])

    def test_builds_and_saves_phrase_corpus(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "sentences.txt").write_text(
                "cash flow improved\ncash flow remained strong\n",
                encoding="utf-8",
            )
            terms_path = directory / "target_terms.csv"
            with terms_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(
                    file, fieldnames=["term", "category", "priority"]
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "term": "cash flow",
                            "category": "liquidity",
                            "priority": "core",
                        },
                        {
                            "term": "revenue",
                            "category": "performance",
                            "priority": "core",
                        },
                        {
                            "term": "scope 1 emissions",
                            "category": "sustainability",
                            "priority": "supplemental",
                        },
                    ]
                )

            result = build_phrase_corpus(directory, terms_path, minimum_count=2)
            paths = save_phrase_corpus(result, directory)

            self.assertEqual(
                paths["sentences"].read_text(encoding="utf-8"),
                "cash_flow improved\ncash_flow remained strong\n",
            )
            saved_report = json.loads(paths["report"].read_text(encoding="utf-8"))
            self.assertEqual(saved_report["detected_phrase_count"], 1)
            self.assertEqual(saved_report["candidate_phrase_count"], 1)
            self.assertEqual(saved_report["total_joined_phrase_occurrences"], 2)

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            detect_phrases([], [], minimum_count=1)
        with self.assertRaises(ValueError):
            detect_phrases([["cash", "flow"]], [], minimum_count=0)


if __name__ == "__main__":
    unittest.main()
