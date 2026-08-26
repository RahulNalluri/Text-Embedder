import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors, Word2Vec

from training.config import TrainingConfig
from training.word2vec_trainer import (
    inspect_corpus,
    save_training_artifacts,
    train_word2vec,
)


class TestWord2VecTrainer(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.corpus_path = self.directory / "sentences_phrased.txt"
        self.corpus_path.write_text(
            "net_profit increased revenue growth\n"
            "operating_profit increased revenue growth\n"
            "net_profit improved financial performance\n"
            "operating_profit improved financial performance\n"
            "revenue growth improved performance\n",
            encoding="utf-8",
        )
        self.config = TrainingConfig(
            embedding_dimension=12,
            context_window=2,
            minimum_word_frequency=1,
            negative_samples=2,
            epochs=4,
            workers=1,
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_inspects_and_trains_streamed_corpus(self):
        statistics = inspect_corpus(self.corpus_path)
        result = train_word2vec(self.corpus_path, self.config)

        self.assertEqual(statistics.sentence_count, 5)
        self.assertEqual(statistics.token_count, 20)
        self.assertEqual(result.model.wv.vector_size, 12)
        self.assertIn("net_profit", result.model.wv)
        self.assertTrue(np.isfinite(result.model.wv.vectors).all())

    def test_training_is_reproducible_with_one_worker(self):
        first = train_word2vec(self.corpus_path, self.config)
        second = train_word2vec(self.corpus_path, self.config)

        np.testing.assert_allclose(
            first.model.wv["net_profit"],
            second.model.wv["net_profit"],
            rtol=0,
            atol=0,
        )

    def test_saves_and_reloads_all_artifacts(self):
        result = train_word2vec(self.corpus_path, self.config)
        artifact_directory = self.directory / "artifacts"

        paths = save_training_artifacts(result, artifact_directory)

        reloaded_model = Word2Vec.load(str(paths["model"]))
        reloaded_vectors = KeyedVectors.load(str(paths["vectors"]), mmap="r")
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
        self.assertIn("net_profit", reloaded_model.wv)
        self.assertEqual(reloaded_vectors.vector_size, 12)
        self.assertTrue(report["validation"]["passed"])
        self.assertEqual(report["configuration"]["architecture"], "skip_gram")

    def test_rejects_missing_empty_and_too_rare_corpora(self):
        with self.assertRaises(FileNotFoundError):
            inspect_corpus(self.directory / "missing.txt")

        empty_path = self.directory / "empty.txt"
        empty_path.write_text("\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            inspect_corpus(empty_path)

        rare_path = self.directory / "rare.txt"
        rare_path.write_text("one two three\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            train_word2vec(
                rare_path,
                TrainingConfig(minimum_word_frequency=5),
            )


if __name__ == "__main__":
    unittest.main()
