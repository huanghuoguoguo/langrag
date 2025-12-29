import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
from langrag.entities.document import Document
from langrag.index_processor.extractor.providers.pdf import PdfParser
from langrag.index_processor.extractor.providers.docx import DocxParser
from langrag.index_processor.extractor.providers.markdown import MarkdownParser
from langrag.index_processor.extractor.providers.simple_text import SimpleTextParser

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
                      docs = parser.parse("dummy.pdf")
            
        assert len(docs) == 1
        assert "Page 1 content" in docs[0].page_content
        assert "Page 2 content" in docs[0].page_content
        assert docs[0].metadata["extension"] == ".pdf"

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
