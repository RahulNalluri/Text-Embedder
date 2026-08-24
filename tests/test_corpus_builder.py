import csv
import json
import tempfile
import unittest
from pathlib import Path

from training.corpus_builder import build_corpus, save_corpus
from training.pdf_extraction import ExtractedPage, PDFExtractionResult


class TestCorpusBuilder(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.primary_pdf = self.directory / "primary.pdf"
        self.duplicate_pdf = self.directory / "duplicate.pdf"
        self.failed_pdf = self.directory / "failed.pdf"
        self.primary_pdf.write_bytes(b"same report content")
        self.duplicate_pdf.write_bytes(b"same report content")
        self.failed_pdf.write_bytes(b"different report content")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def fake_extractor(self, path):
        path = Path(path)
        if path.name == "failed.pdf":
            raise RuntimeError("PDF could not be read")
        return PDFExtractionResult(
            source_path=path,
            pages=(
                ExtractedPage(1, "Revenue increased."),
                ExtractedPage(2, "Cash flow improved."),
            ),
            minimum_text_characters=20,
        )

    @staticmethod
    def fake_preprocessor(text, minimum_tokens):
        del text, minimum_tokens
        return [["revenue", "increased"], ["cash", "flow", "improved"]]

    def test_builds_corpus_and_records_duplicates_and_failures(self):
        result = build_corpus(
            [self.primary_pdf, self.duplicate_pdf, self.failed_pdf],
            extractor=self.fake_extractor,
            preprocessor=self.fake_preprocessor,
        )

        self.assertEqual(len(result.documents), 1)
        self.assertEqual(len(result.skipped_documents), 1)
        self.assertEqual(len(result.failed_documents), 1)
        self.assertEqual(result.token_count, 5)
        self.assertEqual(result.vocabulary_counts["revenue"], 1)
        self.assertEqual(result.to_metadata_dict()["pages_processed"], 2)

    def test_saves_all_corpus_outputs(self):
        result = build_corpus(
            [self.primary_pdf],
            extractor=self.fake_extractor,
            preprocessor=self.fake_preprocessor,
        )
        output_paths = save_corpus(result, self.directory / "processed")

        self.assertEqual(
            output_paths["sentences"].read_text(encoding="utf-8"),
            "revenue increased\ncash flow improved\n",
        )
        with output_paths["vocabulary"].open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(rows[0], {"token": "revenue", "count": "1"})
        metadata = json.loads(output_paths["report"].read_text(encoding="utf-8"))
        self.assertEqual(metadata["documents_processed"], 1)
        self.assertEqual(metadata["sentence_count"], 2)

    def test_rejects_empty_input(self):
        with self.assertRaisesRegex(ValueError, "At least one PDF"):
            build_corpus([])


if __name__ == "__main__":
    unittest.main()
