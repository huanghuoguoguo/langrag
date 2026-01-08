"""Tests for HTML parser."""

import tempfile
from pathlib import Path

import pytest

from langrag.index_processor.extractor.providers.html import HtmlParser


class TestHtmlParser:
    """Tests for HtmlParser class."""

    @pytest.fixture
    def simple_html(self, tmp_path):
        """Create a simple HTML file."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Main Title</h1>
            <p>First paragraph.</p>
            <p>Second paragraph.</p>
        </body>
        </html>
        """
        file_path = tmp_path / "test.html"
        file_path.write_text(html_content)
        return file_path

    @pytest.fixture
    def html_with_scripts(self, tmp_path):
        """Create HTML with scripts and styles."""
        html_content = """
        <html>
        <head>
            <title>Page with Scripts</title>
            <style>body { color: red; }</style>
        </head>
        <body>
            <h1>Title</h1>
            <script>alert('hello');</script>
            <p>Content here.</p>
        </body>
        </html>
        """
        file_path = tmp_path / "with_scripts.html"
        file_path.write_text(html_content)
        return file_path

    @pytest.fixture
    def html_with_list(self, tmp_path):
        """Create HTML with list."""
        html_content = """
        <html>
        <body>
            <h1>List Example</h1>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
                <li>Item 3</li>
            </ul>
        </body>
        </html>
        """
        file_path = tmp_path / "with_list.html"
        file_path.write_text(html_content)
        return file_path

    @pytest.fixture
    def html_with_table(self, tmp_path):
        """Create HTML with table."""
        html_content = """
        <html>
        <body>
            <table>
                <tr><th>Name</th><th>Age</th></tr>
                <tr><td>Alice</td><td>30</td></tr>
                <tr><td>Bob</td><td>25</td></tr>
            </table>
        </body>
        </html>
        """
        file_path = tmp_path / "with_table.html"
        file_path.write_text(html_content)
        return file_path

    def test_init_default(self):
        """Default initialization."""
        parser = HtmlParser()
        assert parser.remove_scripts is True
        assert parser.remove_styles is True
        assert parser.preserve_structure is True

    def test_init_custom(self):
        """Custom initialization."""
        parser = HtmlParser(
            remove_scripts=False,
            remove_styles=False,
            preserve_structure=False,
        )
        assert parser.remove_scripts is False
        assert parser.remove_styles is False
        assert parser.preserve_structure is False

    def test_parse_simple_html(self, simple_html):
        """Parse simple HTML file."""
        parser = HtmlParser()
        docs = parser.parse(simple_html)

        assert len(docs) == 1
        assert "Main Title" in docs[0].page_content
        assert "First paragraph" in docs[0].page_content
        assert docs[0].metadata["parser"] == "HtmlParser"
        assert docs[0].metadata["title"] == "Test Page"

    def test_parse_removes_scripts(self, html_with_scripts):
        """Scripts are removed from output."""
        parser = HtmlParser(remove_scripts=True)
        docs = parser.parse(html_with_scripts)

        assert "alert" not in docs[0].page_content
        assert "Content here" in docs[0].page_content

    def test_parse_removes_styles(self, html_with_scripts):
        """Styles are removed from output."""
        parser = HtmlParser(remove_styles=True)
        docs = parser.parse(html_with_scripts)

        assert "color: red" not in docs[0].page_content

    def test_parse_preserves_list(self, html_with_list):
        """Lists are preserved in output."""
        parser = HtmlParser(preserve_structure=True)
        docs = parser.parse(html_with_list)

        content = docs[0].page_content
        assert "Item 1" in content
        assert "Item 2" in content
        assert "Item 3" in content

    def test_parse_extracts_table(self, html_with_table):
        """Tables are extracted as markdown."""
        parser = HtmlParser(preserve_structure=True)
        docs = parser.parse(html_with_table)

        content = docs[0].page_content
        assert "Name" in content
        assert "Alice" in content
        assert "Bob" in content

    def test_parse_file_not_found(self):
        """Raises error for non-existent file."""
        parser = HtmlParser()

        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/file.html")

    def test_parse_not_a_file(self, tmp_path):
        """Raises error for directory."""
        parser = HtmlParser()

        with pytest.raises(ValueError, match="Not a file"):
            parser.parse(tmp_path)

    def test_parse_no_preserve_structure(self, simple_html):
        """Parse without preserving structure."""
        parser = HtmlParser(preserve_structure=False)
        docs = parser.parse(simple_html)

        assert len(docs) == 1
        assert "Main Title" in docs[0].page_content

    def test_metadata_contains_file_info(self, simple_html):
        """Metadata contains file information."""
        parser = HtmlParser()
        docs = parser.parse(simple_html)

        metadata = docs[0].metadata
        assert "source" in metadata
        assert "filename" in metadata
        assert "extension" in metadata
        assert metadata["extension"] == ".html"
