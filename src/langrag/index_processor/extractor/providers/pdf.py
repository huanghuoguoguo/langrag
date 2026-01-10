"""PDF file parser with large file protection."""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

try:
    from pypdf import PdfReader
    from pypdf.errors import FileNotDecryptedError

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    FileNotDecryptedError = Exception  # Fallback for type hints
    logger.warning("pypdf not installed. PDF parsing will not be available.")

from langrag.entities.document import Document

from ..base import BaseParser


class FileTooLargeError(ValueError):
    """Raised when a file exceeds the configured size or page limits."""

    pass


# Default limits
DEFAULT_MAX_FILE_SIZE_MB = 100  # 100 MB
DEFAULT_MAX_PAGES = 1000  # 1000 pages


class PdfParser(BaseParser):
    """PDF file parser with large file protection.

    Uses pypdf to extract text content from PDF files.
    Includes safeguards against memory exhaustion from large files.

    Args:
        extract_images: Whether to extract images (not yet supported)
        pages: Page range to parse, None means all pages
        max_file_size_mb: Maximum file size in MB (default: 100)
        max_pages: Maximum number of pages to process (default: 1000)

    Usage example:
        >>> parser = PdfParser()
        >>> docs = parser.parse("document.pdf")

        >>> # Custom limits
        >>> parser = PdfParser(max_file_size_mb=50, max_pages=500)
        >>> docs = parser.parse("large_document.pdf")

    Raises:
        FileTooLargeError: When file exceeds size or page limits
    """

    def __init__(
        self,
        extract_images: bool = False,
        pages: tuple[int, int] | None = None,
        max_file_size_mb: float = DEFAULT_MAX_FILE_SIZE_MB,
        max_pages: int = DEFAULT_MAX_PAGES,
    ):
        """Initialize PDF parser with size limits.

        Args:
            extract_images: Whether to extract images (not implemented)
            pages: (start, end) page range, None means all pages
            max_file_size_mb: Maximum file size in MB (0 = unlimited)
            max_pages: Maximum pages to process (0 = unlimited)
        """
        if not PDF_AVAILABLE:
            raise ImportError(
                "pypdf is required for PDF parsing. Install it with: pip install pypdf"
            )

        self.extract_images = extract_images
        self.pages = pages
        self.max_file_size_mb = max_file_size_mb
        self.max_pages = max_pages

        if extract_images:
            logger.warning("Image extraction from PDF is not yet implemented")

    def _check_file_size(self, path: Path) -> int:
        """Check if file size is within limits.

        Args:
            path: Path to the file

        Returns:
            File size in bytes

        Raises:
            FileTooLargeError: If file exceeds max_file_size_mb
        """
        file_size = os.path.getsize(path)
        file_size_mb = file_size / (1024 * 1024)

        if self.max_file_size_mb > 0 and file_size_mb > self.max_file_size_mb:
            raise FileTooLargeError(
                f"PDF file too large: {file_size_mb:.1f} MB exceeds limit of "
                f"{self.max_file_size_mb} MB. Consider splitting the document or "
                f"increasing max_file_size_mb parameter."
            )

        logger.debug(f"File size: {file_size_mb:.2f} MB (limit: {self.max_file_size_mb} MB)")
        return file_size

    def _check_page_count(self, total_pages: int) -> None:
        """Check if page count is within limits.

        Args:
            total_pages: Total number of pages in the PDF

        Raises:
            FileTooLargeError: If page count exceeds max_pages
        """
        if self.max_pages > 0 and total_pages > self.max_pages:
            raise FileTooLargeError(
                f"PDF has too many pages: {total_pages} exceeds limit of "
                f"{self.max_pages} pages. Consider using the 'pages' parameter "
                f"to process a subset, or increase max_pages parameter."
            )

        logger.debug(f"Page count: {total_pages} (limit: {self.max_pages})")

    def parse(self, file_path: str | Path, **_kwargs) -> list[Document]:
        """Parse PDF file with size protection.

        Args:
            file_path: PDF file path
            **kwargs: Additional parameters

        Returns:
            A list containing a single Document (all pages merged)

        Raises:
            FileNotFoundError: File does not exist
            ValueError: Not a valid PDF file
            FileTooLargeError: File exceeds size or page limits
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        # Check file size before opening
        file_size = self._check_file_size(path)

        logger.info(f"Parsing PDF file: {path}")

        try:
            with open(path, "rb") as file:
                pdf_reader = PdfReader(file)
                total_pages = len(pdf_reader.pages)

                # Check page count
                self._check_page_count(total_pages)

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
                        "file_size_bytes": file_size,
                    },
                )

                logger.info(f"Parsed PDF: {total_pages} pages, {len(content)} characters extracted")

                return [doc]

        except FileNotDecryptedError as e:
            logger.error(f"PDF is password-protected: {path}")
            raise ValueError(
                f"PDF is password-protected and cannot be read: {path}. "
                "Please provide an unencrypted version of the file."
            ) from e

        except FileTooLargeError:
            # Re-raise size limit errors as-is
            raise

        except Exception as e:
            logger.error(f"Failed to parse PDF {path}: {e}")
            raise ValueError(f"Invalid PDF file: {e}") from e
