import csv
import json
import tempfile
import unittest
from pathlib import Path

from training.corpus_builder import (
    build_corpus,
    remove_repeated_page_edges,
    save_corpus,
)
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
        self.assertEqual(metadata["raw_sentence_count"], 2)
        self.assertEqual(metadata["sentence_count"], 2)
        self.assertEqual(metadata["repeated_edge_lines_removed"], 0)
        self.assertEqual(
            (output_paths["documents"] / "primary.txt").read_text(encoding="utf-8"),
            "revenue increased\ncash flow improved\n",
        )

    def test_removes_only_repeated_page_edge_lines(self):
        extraction = PDFExtractionResult(
            source_path=self.primary_pdf,
            pages=tuple(
                ExtractedPage(
                    page_number=page_number,
                    text=(
                        "Annual Report 2025\n"
                        f"Useful page {page_number} financial content.\n"
                        "Company Confidential"
                    ),
                )
                for page_number in range(1, 6)
            ),
            minimum_text_characters=20,
        )

        cleaned_text, removed_count = remove_repeated_page_edges(extraction)

        self.assertEqual(removed_count, 10)
        self.assertNotIn("Annual Report 2025", cleaned_text)
        self.assertNotIn("Company Confidential", cleaned_text)
        self.assertIn("Useful page 3 financial content.", cleaned_text)

    def test_rejects_empty_input(self):
        with self.assertRaisesRegex(ValueError, "At least one PDF"):
            build_corpus([])

    def test_removes_exact_duplicate_sentences_across_documents(self):
        second_pdf = self.directory / "second.pdf"
        second_pdf.write_bytes(b"unique file bytes")

        result = build_corpus(
            [self.primary_pdf, second_pdf],
            extractor=self.fake_extractor,
            preprocessor=self.fake_preprocessor,
        )

        self.assertEqual(len(result.sentences), 2)
        self.assertEqual(result.documents[0].duplicate_sentences_removed, 0)
        self.assertEqual(result.documents[1].duplicate_sentences_removed, 2)


if __name__ == "__main__":
    unittest.main()
