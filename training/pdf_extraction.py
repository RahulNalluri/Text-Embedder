"""Extract page-level text from annual-report PDF files."""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PDFExtractionError(RuntimeError):
    """Raised when an annual-report PDF cannot be read safely."""


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """Text extracted from one PDF page."""

    page_number: int
    text: str

    @property
    def character_count(self) -> int:
        """Return the number of non-surrounding-whitespace characters."""

        return len(self.text.strip())


@dataclass(frozen=True, slots=True)
class PDFExtractionResult:
    """Page text and quality information for one PDF document."""

    source_path: Path
    pages: tuple[ExtractedPage, ...]
    minimum_text_characters: int

    @property
    def combined_text(self) -> str:
        """Join non-empty pages while preserving page boundaries."""

        return "\n\n".join(page.text.strip() for page in self.pages if page.text.strip())

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def text_pages(self) -> int:
        return sum(page.character_count > 0 for page in self.pages)

    @property
    def possible_scanned_pages(self) -> tuple[int, ...]:
        """Return pages with too little extractable text for normal processing."""

        return tuple(
            page.page_number
            for page in self.pages
            if page.character_count < self.minimum_text_characters
        )

    @property
    def extraction_rate(self) -> float:
        """Return the proportion of pages containing extractable text."""

        if self.total_pages == 0:
            return 0.0
        return self.text_pages / self.total_pages


def extract_pdf_text(
    pdf_path: str | Path,
    minimum_text_characters: int = 20,
) -> PDFExtractionResult:
    """Extract text from every page of a digital annual-report PDF.

    Pages containing fewer than ``minimum_text_characters`` are retained but
    flagged as possible scanned or image-only pages. OCR is intentionally not
    performed in this first extraction stage.
    """

    path = Path(pdf_path)

    if minimum_text_characters < 1:
        raise ValueError("minimum_text_characters must be at least 1.")
    if not path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"PDF path must point to a file: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file: {path}")

    try:
        reader = PdfReader(path)

        if reader.is_encrypted and reader.decrypt("") == 0:
            raise PDFExtractionError(
                f"PDF is password-protected and cannot be opened: {path}"
            )

        pages = tuple(
            ExtractedPage(
                page_number=index,
                text=(page.extract_text() or "").replace("\x00", ""),
            )
            for index, page in enumerate(reader.pages, start=1)
        )
    except PDFExtractionError:
        raise
    except (PdfReadError, OSError, ValueError) as error:
        raise PDFExtractionError(f"Could not read PDF file: {path}") from error

    return PDFExtractionResult(
        source_path=path.resolve(),
        pages=pages,
        minimum_text_characters=minimum_text_characters,
    )
