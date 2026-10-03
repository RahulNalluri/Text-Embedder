import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from training.config import TrainingConfig
from training.version3_trainer import (
    AUDIT_FILENAMES,
    run_version3_training,
    validate_version3_training_inputs,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestVersion3Trainer(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.corpus_directory = self.directory / "corpus"
        self.corpus_directory.mkdir()
        self.corpus_path = self.corpus_directory / "sentences_phrased.txt"
        self.corpus_path.write_text(
            "net_profit increased revenue growth\n"
            "operating_profit increased revenue growth\n"
            "net_profit improved financial performance\n"
            "operating_profit improved financial performance\n"
            "revenue growth improved performance\n",
            encoding="utf-8",
        )
        for filename in AUDIT_FILENAMES:
            if filename != "version3_build_summary.json":
                (self.corpus_directory / filename).write_text("{}", encoding="utf-8")
        self.summary_path = self.corpus_directory / "version3_build_summary.json"
        self._write_summary(sha256(self.corpus_path))
        self.config = TrainingConfig(
            embedding_dimension=12,
            context_window=2,
            minimum_word_frequency=1,
            negative_samples=2,
            epochs=2,
            workers=1,
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write_summary(self, corpus_hash: str) -> None:
        self.summary_path.write_text(
            json.dumps(
                {
                    "dataset_sha256": "a" * 64,
                    "report_count": 2,
                    "quality_warnings": ["review test warning"],
                    "output_sha256": {"sentences_phrased.txt": corpus_hash},
                }
            ),
            encoding="utf-8",
        )

    def test_validates_frozen_phrase_corpus_hash(self):
        provenance = validate_version3_training_inputs(
            self.corpus_directory,
            expected_report_count=2,
        )

        self.assertEqual(provenance.corpus_sha256, sha256(self.corpus_path))
        self.assertEqual(provenance.report_count, 2)
        self.assertEqual(provenance.corpus_quality_warnings, ("review test warning",))

        self._write_summary("0" * 64)
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            validate_version3_training_inputs(
                self.corpus_directory,
                expected_report_count=2,
            )

    def test_trains_versioned_package_without_overwriting(self):
        output_directory = self.directory / "version_3"

        result_path = run_version3_training(
            corpus_directory=self.corpus_directory,
            output_directory=output_directory,
            config=self.config,
            expected_report_count=2,
        )

        report = json.loads(
            (result_path / "training_report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["model_name"], "indian_financial_word2vec_v3")
        self.assertEqual(report["model_version"], 3)
        self.assertEqual(report["dataset_provenance"]["report_count"], 2)
        self.assertEqual(
            report["dataset_provenance"]["phrase_corpus_sha256"],
            sha256(self.corpus_path),
        )
        self.assertTrue((result_path / "word2vec.model").is_file())
        self.assertTrue((result_path / "vectors.kv").is_file())
        for filename in AUDIT_FILENAMES:
            self.assertTrue((result_path / filename).is_file())

        with self.assertRaises(FileExistsError):
            run_version3_training(
                corpus_directory=self.corpus_directory,
                output_directory=output_directory,
                config=self.config,
                expected_report_count=2,
            )


if __name__ == "__main__":
    unittest.main()
