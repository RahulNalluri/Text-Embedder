import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors

from training.corpus_quality import load_target_terms
from training.model_evaluation import (
    EvaluationPair,
    evaluate_model,
    evaluate_saved_model,
    load_training_report,
    save_evaluation_report,
)


class TestModelEvaluation(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        vectors = KeyedVectors(vector_size=3)
        vectors.add_vectors(
            ["revenue", "profit", "net_profit", "profit_after_tax", "cybersecurity", "evenue"],
            np.asarray([
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.9, 0.1],
                [-1.0, 0.0, 0.0],
                [0.95, 0.05, 0.0],
            ], dtype=np.float32),
        )
        vectors.save(str(self.directory / "vectors.kv"))
        (self.directory / "training_report.json").write_text(
            json.dumps({
                "model_name": "test_financial_word2vec",
                "configuration": {"embedding_dimension": 3},
                "training": {"retained_vocabulary_size": 6},
            }),
            encoding="utf-8",
        )
        self.target_terms_path = self.directory / "target_terms.csv"
        with self.target_terms_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["term", "category", "priority"])
            writer.writeheader()
            writer.writerows([
                {"term": "revenue", "category": "performance", "priority": "core"},
                {"term": "net profit", "category": "performance", "priority": "core"},
                {"term": "credit risk", "category": "risk", "priority": "core"},
                {"term": "scope 1 emissions", "category": "sustainability", "priority": "supplemental"},
            ])

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_evaluates_coverage_similarity_and_noise(self):
        report = evaluate_saved_model(self.directory, self.target_terms_path)

        self.assertTrue(report["technical_validation"]["passed"])
        self.assertEqual(report["target_coverage"]["overall"]["covered_terms"], 2)
        self.assertEqual(report["target_coverage"]["overall"]["evaluable_terms"], 3)
        self.assertEqual(report["target_coverage"]["not_evaluable_terms"], ["scope 1 emissions"])
        self.assertIn("evenue", {row["token"] for row in report["suspicious_tokens"]})

    def test_custom_pairs_separate_related_terms(self):
        vectors = KeyedVectors.load(str(self.directory / "vectors.kv"), mmap="r")
        report = evaluate_model(
            vectors,
            load_target_terms(self.target_terms_path),
            load_training_report(self.directory / "training_report.json"),
            pairs=(
                EvaluationPair("revenue", "profit", "related"),
                EvaluationPair("revenue", "cybersecurity", "unrelated"),
            ),
            neighbour_terms=("revenue",),
        )

        benchmark = report["similarity_benchmark"]
        self.assertGreater(benchmark["average_margin"], 1.0)
        self.assertEqual(benchmark["pairwise_separation_accuracy"], 1.0)

    def test_saves_evaluation_report(self):
        report = evaluate_saved_model(self.directory, self.target_terms_path)
        output_path = save_evaluation_report(report, self.directory)

        saved = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["model_name"], "test_financial_word2vec")


if __name__ == "__main__":
    unittest.main()
