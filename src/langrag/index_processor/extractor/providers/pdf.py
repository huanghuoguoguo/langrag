"""PDF file parser"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

try:
    from pypdf import PdfReader

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("pypdf not installed. PDF parsing will not be available.")

from langrag.entities.document import Document

from ..base import BaseParser


class PdfParser(BaseParser):
    """PDF file parser

    Uses pypdf to extract text content from PDF files.

    Args:
        extract_images: Whether to extract images (not yet supported)
        pages: Page range to parse, None means all pages

    Usage example:
        >>> parser = PdfParser()
        >>> docs = parser.parse("document.pdf")
    """

    def __init__(self, extract_images: bool = False, pages: tuple[int, int] | None = None):
        """Initialize PDF parser

        Args:
            extract_images: Whether to extract images (not implemented)
            pages: (start, end) page range, None means all pages
        """
        if not PDF_AVAILABLE:
            raise ImportError(
                "pypdf is required for PDF parsing. Install it with: pip install pypdf"
            )

        self.extract_images = extract_images
        self.pages = pages

        if extract_images:
            logger.warning("Image extraction from PDF is not yet implemented")

    def parse(self, file_path: str | Path, **_kwargs) -> list[Document]:
        """Parse PDF file

        Args:
            file_path: PDF file path
            **kwargs: Additional parameters

        Returns:
            A list containing a single Document (all pages merged)

        Raises:
            FileNotFoundError: File does not exist
            ValueError: Not a valid PDF file
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        logger.info(f"Parsing PDF file: {path}")

        try:
            with open(path, "rb") as file:
                pdf_reader = PdfReader(file)
                total_pages = len(pdf_reader.pages)

                # Determine the page range to parse
                if self.pages:
                    start, end = self.pages
                    start = max(0, start)
                    end = min(total_pages, end)
                else:
                    start, end = 0, total_pages

                logger.debug(f"Extracting pages {start} to {end} of {total_pages}")

                # Extract text
                text_content = []
                for page_num in range(start, end):
                    try:
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        if text and text.strip():
                            text_content.append(text)
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num}: {e}")

                content = "\n".join(text_content)

                if not content.strip():
                    logger.warning(f"No text content extracted from {path}")
                    logger.warning("This may be a scanned PDF or image-based PDF. OCR is required.")
                    # Provide placeholder to avoid validation error
                    content = f"[PDF file: {path.name}]\n[Unable to extract text content, may be a scanned PDF, OCR support required]"

                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(path.absolute()),
                        "filename": path.name,
                        "extension": path.suffix,
                        "total_pages": total_pages,
                        "extracted_pages": f"{start}-{end}",
                        "parser": "PdfParser",
                        "text_extracted": len(content.strip()) > 0,
                    },
                )

                logger.info(f"Parsed PDF: {total_pages} pages, {len(content)} characters extracted")

                return [doc]

        except Exception as e:
            logger.error(f"Failed to parse PDF {path}: {e}")
            raise ValueError(f"Invalid PDF file: {e}") from e
