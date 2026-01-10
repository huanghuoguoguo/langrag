from unittest.mock import MagicMock, mock_open, patch

import pytest

from langrag.index_processor.extractor.providers.docx import DocxParser
from langrag.index_processor.extractor.providers.markdown import MarkdownParser
from langrag.index_processor.extractor.providers.pdf import (
    DEFAULT_MAX_FILE_SIZE_MB,
    DEFAULT_MAX_PAGES,
    FileTooLargeError,
    PdfParser,
)


class TestPdfParser:
    @patch("langrag.index_processor.extractor.providers.pdf.PDF_AVAILABLE", True)
    @patch("langrag.index_processor.extractor.providers.pdf.PdfReader")
    def test_parse_pdf_extracts_text(self, mock_reader_cls):
        # Mock PdfReader
        mock_reader = MagicMock()
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2 content"
        mock_reader.pages = [mock_page1, mock_page2]
        mock_reader_cls.return_value = mock_reader

        parser = PdfParser()

        # We need to mock file open as well since PdfParser opens the file
        with patch("builtins.open", mock_open(read_data=b"pdf_data")):
            # Just use a dummy path
            with patch("pathlib.Path.exists", return_value=True):
                 with patch("pathlib.Path.is_file", return_value=True):
                      with patch("os.path.getsize", return_value=1024):  # 1KB file
                          docs = parser.parse("dummy.pdf")

        assert len(docs) == 1
        assert "Page 1 content" in docs[0].page_content
        assert "Page 2 content" in docs[0].page_content
        assert docs[0].metadata["extension"] == ".pdf"


class TestPdfParserLargeFileProtection:
    """Tests for PDF parser large file protection."""

    def test_default_limits(self):
        """Test that default limits are set correctly."""
        with patch("langrag.index_processor.extractor.providers.pdf.PDF_AVAILABLE", True):
            parser = PdfParser()
            assert parser.max_file_size_mb == DEFAULT_MAX_FILE_SIZE_MB
            assert parser.max_pages == DEFAULT_MAX_PAGES

    def test_custom_limits(self):
        """Test custom size and page limits."""
        with patch("langrag.index_processor.extractor.providers.pdf.PDF_AVAILABLE", True):
            parser = PdfParser(max_file_size_mb=50, max_pages=500)
            assert parser.max_file_size_mb == 50
            assert parser.max_pages == 500

    @patch("langrag.index_processor.extractor.providers.pdf.PDF_AVAILABLE", True)
    def test_file_size_exceeded_raises_error(self):
        """Test that exceeding file size limit raises FileTooLargeError."""
        parser = PdfParser(max_file_size_mb=10)  # 10 MB limit

        # Mock a 50 MB file
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("os.path.getsize", return_value=50 * 1024 * 1024):  # 50 MB

            with pytest.raises(FileTooLargeError) as exc_info:
                parser.parse("large_file.pdf")

            assert "50.0 MB exceeds limit of 10 MB" in str(exc_info.value)

    @patch("langrag.index_processor.extractor.providers.pdf.PDF_AVAILABLE", True)
    @patch("langrag.index_processor.extractor.providers.pdf.PdfReader")
    def test_page_count_exceeded_raises_error(self, mock_reader_cls):
        """Test that exceeding page limit raises FileTooLargeError."""
        parser = PdfParser(max_pages=100)  # 100 page limit

        # Mock a PDF with 500 pages
        mock_reader = MagicMock()
        mock_reader.pages = [MagicMock() for _ in range(500)]
        mock_reader_cls.return_value = mock_reader

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("os.path.getsize", return_value=1024), \
             patch("builtins.open", mock_open(read_data=b"pdf")):

            with pytest.raises(FileTooLargeError) as exc_info:
                parser.parse("many_pages.pdf")

            assert "500 exceeds limit of 100 pages" in str(exc_info.value)

    @patch("langrag.index_processor.extractor.providers.pdf.PDF_AVAILABLE", True)
    @patch("langrag.index_processor.extractor.providers.pdf.PdfReader")
    def test_unlimited_file_size(self, mock_reader_cls):
        """Test that max_file_size_mb=0 means unlimited."""
        parser = PdfParser(max_file_size_mb=0)  # Unlimited

        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Content"
        mock_reader.pages = [mock_page]
        mock_reader_cls.return_value = mock_reader

        # Mock a 500 MB file - should NOT raise error
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("os.path.getsize", return_value=500 * 1024 * 1024), \
             patch("builtins.open", mock_open(read_data=b"pdf")):

            docs = parser.parse("huge_file.pdf")
            assert len(docs) == 1

    @patch("langrag.index_processor.extractor.providers.pdf.PDF_AVAILABLE", True)
    @patch("langrag.index_processor.extractor.providers.pdf.PdfReader")
    def test_unlimited_pages(self, mock_reader_cls):
        """Test that max_pages=0 means unlimited."""
        parser = PdfParser(max_pages=0)  # Unlimited

        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Content"
        mock_reader.pages = [mock_page for _ in range(5000)]  # 5000 pages
        mock_reader_cls.return_value = mock_reader

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("os.path.getsize", return_value=1024), \
             patch("builtins.open", mock_open(read_data=b"pdf")):

            docs = parser.parse("many_pages.pdf")
            assert len(docs) == 1

    @patch("langrag.index_processor.extractor.providers.pdf.PDF_AVAILABLE", True)
    @patch("langrag.index_processor.extractor.providers.pdf.PdfReader")
    def test_file_size_in_metadata(self, mock_reader_cls):
        """Test that file size is stored in document metadata."""
        parser = PdfParser()

        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Content"
        mock_reader.pages = [mock_page]
        mock_reader_cls.return_value = mock_reader

        file_size = 5 * 1024 * 1024  # 5 MB

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("os.path.getsize", return_value=file_size), \
             patch("builtins.open", mock_open(read_data=b"pdf")):

            docs = parser.parse("test.pdf")
            assert docs[0].metadata["file_size_bytes"] == file_size

    @patch("langrag.index_processor.extractor.providers.pdf.PDF_AVAILABLE", True)
    @patch("langrag.index_processor.extractor.providers.pdf.PdfReader")
    def test_encrypted_pdf_raises_value_error(self, mock_reader_cls):
        """Test that encrypted PDFs raise a clear error message."""
        from langrag.index_processor.extractor.providers.pdf import FileNotDecryptedError

        parser = PdfParser()

        # Mock PdfReader raising FileNotDecryptedError
        mock_reader_cls.side_effect = FileNotDecryptedError("PDF is encrypted")

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("os.path.getsize", return_value=1024), \
             patch("builtins.open", mock_open(read_data=b"pdf")):

            with pytest.raises(ValueError) as exc_info:
                parser.parse("encrypted.pdf")

            assert "password-protected" in str(exc_info.value)

    @patch("langrag.index_processor.extractor.providers.pdf.PDF_AVAILABLE", True)
    @patch("langrag.index_processor.extractor.providers.pdf.PdfReader")
    def test_file_within_limits_succeeds(self, mock_reader_cls):
        """Test that files within limits are processed successfully."""
        parser = PdfParser(max_file_size_mb=100, max_pages=1000)

        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page content"
        mock_reader.pages = [mock_page for _ in range(50)]  # 50 pages
        mock_reader_cls.return_value = mock_reader

        # 10 MB file, 50 pages - within limits
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("os.path.getsize", return_value=10 * 1024 * 1024), \
             patch("builtins.open", mock_open(read_data=b"pdf")):

            docs = parser.parse("normal.pdf")
            assert len(docs) == 1
            assert docs[0].metadata["total_pages"] == 50

class TestDocxParser:
    @patch("langrag.index_processor.extractor.providers.docx.DOCX_AVAILABLE", True)
    @patch("langrag.index_processor.extractor.providers.docx.DocxDocument")
    def test_parse_docx_extracts_paragraphs_and_tables(self, mock_doc_cls):
        mock_doc = MagicMock()

        # Mock paragraphs
        p1 = MagicMock()
        p1.text = "Paragraph 1"
        mock_doc.paragraphs = [p1]

        # Mock tables
        t1 = MagicMock()
        r1 = MagicMock()
        c1 = MagicMock()
        c1.text = "Cell 1"
        c2 = MagicMock()
        c2.text = "Cell 2"
        r1.cells = [c1, c2]
        t1.rows = [r1]
        mock_doc.tables = [t1]

        # Mock sections (headers/footers) - just empty for now unless explicitly testing them
        mock_doc.sections = []

        mock_doc_cls.return_value = mock_doc

        parser = DocxParser(include_tables=True)

        with patch("pathlib.Path.exists", return_value=True):
             with patch("pathlib.Path.is_file", return_value=True):
                  docs = parser.parse("dummy.docx")

        content = docs[0].page_content
        assert "Paragraph 1" in content
        assert "Cell 1 | Cell 2" in content

class TestMarkdownParser:
    @patch("langrag.index_processor.extractor.providers.markdown.MARKDOWN_AVAILABLE", False)
    def test_parse_simple(self):
        # Test simple parsing (forced fallback)
        content = "# Header\n\nBody text"

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.read_text", return_value=content):

            parser = MarkdownParser(preserve_structure=True)
            docs = parser.parse("dummy.md")

            assert "Header" in docs[0].page_content
            assert "Body text" in docs[0].page_content

    @patch("langrag.index_processor.extractor.providers.markdown.MARKDOWN_AVAILABLE", True)
    @patch("langrag.index_processor.extractor.providers.markdown.markdown")
    @patch("langrag.index_processor.extractor.providers.markdown.BeautifulSoup")
    def test_parse_structured_calls_deps(self, mock_bs, mock_md):
        # Verify structure path uses markdown lib
        mock_md.markdown.return_value = "<html><p>Content</p></html>"

        # Setup BS4 mock to iterate and return text
        mock_element = MagicMock()
        mock_element.name = "p"
        mock_element.get_text.return_value = "Content"

        mock_bs.return_value.children = [mock_element]

        parser = MarkdownParser(preserve_structure=True)
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.read_text", return_value="# H"):

             docs = parser.parse("dummy.md")

        mock_md.markdown.assert_called_once()
        mock_bs.assert_called_once()
        assert "Content" in docs[0].page_content
