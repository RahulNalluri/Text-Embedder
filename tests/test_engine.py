import unittest

import numpy as np

from embedding_engine import (
    build_cooccurrence_matrix,
    build_vocabulary,
    calculate_cosine_similarity,
    calculate_ppmi,
    generate_sentence_embedding,
    generate_word_embeddings,
    tokenize,
)


class TestEmbeddingEngine(unittest.TestCase):
    def test_complete_embedding_pipeline(self):
        corpus = "I am Rahul and I love Python"

        tokens = tokenize(corpus)
        vocabulary = build_vocabulary(tokens)

        cooccurrence_matrix = build_cooccurrence_matrix(
            tokens=tokens,
            vocabulary=vocabulary,
            window_size=2,
        )

        ppmi_matrix = calculate_ppmi(
            cooccurrence_matrix
        )

        word_embeddings = generate_word_embeddings(
            ppmi_matrix=ppmi_matrix,
            embedding_dimension=3,
        )

        sentence_tokens = tokenize(
            "I love Python"
        )

        sentence_embedding = generate_sentence_embedding(
            tokens=sentence_tokens,
            vocabulary=vocabulary,
            word_embeddings=word_embeddings,
        )

        self.assertEqual(
            cooccurrence_matrix.shape,
            (6, 6),
        )

        self.assertEqual(
            ppmi_matrix.shape,
            (6, 6),
        )

        self.assertEqual(
            word_embeddings.shape,
            (6, 3),
        )

        self.assertEqual(
            sentence_embedding.shape,
            (3,),
        )

        self.assertTrue(
            np.all(ppmi_matrix >= 0)
        )

        self.assertAlmostEqual(
            np.linalg.norm(sentence_embedding),
            1.0,
            places=6,
        )

        self.assertAlmostEqual(
            calculate_cosine_similarity(
                sentence_embedding,
                sentence_embedding,
            ),
            1.0,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()