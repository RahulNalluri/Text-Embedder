import unittest

from training.preprocessing import (
    normalize_pdf_text,
    preprocess_text,
    split_sentences,
    tokenize_financial_sentence,
)


class TestPreprocessing(unittest.TestCase):
    def test_repairs_observed_pdf_font_artifacts(self):
        text = (
            "Annual Repo/r_t.liga and pe/r_f.ligaormance were ce/r_t.ligaain. "
            "The commi/t_t.ligaee approved it."
        )

        normalized = normalize_pdf_text(text)

        self.assertEqual(
            normalized,
            "Annual Report and performance were certain. The committee approved it.",
        )

    def test_restores_rupee_symbol_and_joins_broken_words(self):
        normalized = normalize_pdf_text(
            "Revenue was /uni20B9100 crore and per-\nformance improved."
        )

        self.assertEqual(
            normalized,
            "Revenue was ₹100 crore and performance improved.",
        )

    def test_splits_sentences_and_paragraphs(self):
        sentences = split_sentences(
            "Revenue increased. Profit improved!\n\nCash flow remained strong."
        )

        self.assertEqual(
            sentences,
            ["Revenue increased.", "Profit improved!", "Cash flow remained strong."],
        )

    def test_tokenizes_financial_values_consistently(self):
        tokens = tokenize_financial_sentence("Revenue was ₹1,250 crore, up 12%.")

        self.assertEqual(
            tokens,
            ["revenue", "was", "rupee", "<number>", "crore", "up", "<number>", "percent"],
        )

    def test_prepares_only_sentences_meeting_minimum_length(self):
        sentences = preprocess_text(
            "Strong growth. Revenue increased by 12%. Cash flow improved significantly."
        )

        self.assertEqual(
            sentences,
            [
                ["revenue", "increased", "by", "<number>", "percent"],
                ["cash", "flow", "improved", "significantly"],
            ],
        )

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(TypeError):
            normalize_pdf_text(None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            preprocess_text("Revenue increased.", minimum_tokens=0)


if __name__ == "__main__":
    unittest.main()
