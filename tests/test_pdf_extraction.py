import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from training.pdf_extraction import extract_pdf_text


class TestPDFExtraction(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.pdf_path = Path(self.temporary_directory.name) / "annual-report.pdf"
        self.pdf_path.touch()

    def tearDown(self):
        self.temporary_directory.cleanup()

    @patch("training.pdf_extraction.PdfReader")
    def test_extracts_text_and_reports_low_text_pages(self, reader_class):
        reader = reader_class.return_value
        reader.is_encrypted = False
        reader.pages = [
            Mock(extract_text=Mock(return_value="Revenue increased by 12 percent.")),
            Mock(extract_text=Mock(return_value=None)),
        ]

        result = extract_pdf_text(self.pdf_path)

        self.assertEqual(result.total_pages, 2)
        self.assertEqual(result.text_pages, 1)
        self.assertEqual(result.possible_scanned_pages, (2,))
        self.assertEqual(result.extraction_rate, 0.5)
        self.assertIn("Revenue increased", result.combined_text)

    def test_rejects_non_pdf_files(self):
        text_path = Path(self.temporary_directory.name) / "report.txt"
        text_path.touch()

        with self.assertRaisesRegex(ValueError, "Expected a .pdf file"):
            extract_pdf_text(text_path)

    def test_rejects_missing_files(self):
        with self.assertRaises(FileNotFoundError):
            extract_pdf_text(Path(self.temporary_directory.name) / "missing.pdf")

    def test_rejects_invalid_text_threshold(self):
        with self.assertRaisesRegex(ValueError, "must be at least 1"):
            extract_pdf_text(self.pdf_path, minimum_text_characters=0)


if __name__ == "__main__":
    unittest.main()
