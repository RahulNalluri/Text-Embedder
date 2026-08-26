import unittest

from training import DEFAULT_CONFIG, TrainingConfig


class TestTrainingConfig(unittest.TestCase):
    def test_default_financial_training_configuration(self):
        self.assertEqual(DEFAULT_CONFIG.context_window, 3)
        self.assertEqual(DEFAULT_CONFIG.embedding_dimension, 50)
        self.assertEqual(DEFAULT_CONFIG.negative_samples, 5)
        self.assertEqual(DEFAULT_CONFIG.random_seed, 42)
        self.assertEqual(DEFAULT_CONFIG.architecture, "skip_gram")
        self.assertEqual(DEFAULT_CONFIG.workers, 1)

    def test_configuration_can_be_serialized(self):
        settings = DEFAULT_CONFIG.to_dict()

        self.assertEqual(
            settings["domain_name"],
            "indian_corporate_financial_reporting",
        )

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            TrainingConfig(context_window=0)
        with self.assertRaises(ValueError):
            TrainingConfig(architecture="unknown")


if __name__ == "__main__":
    unittest.main()
