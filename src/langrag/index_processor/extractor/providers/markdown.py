"""Markdown file parser"""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

try:
    import markdown
    from bs4 import BeautifulSoup

    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
    logger.warning("markdown/beautifulsoup4 not installed. Markdown parsing will be limited.")

from langrag.entities.document import Document

from ..base import BaseParser


class MarkdownParser(BaseParser):
    """Markdown file parser

    Converts Markdown to structured plain text.
    Supports headings, lists, code blocks, tables, etc.

    Args:
        preserve_structure: Whether to preserve Markdown structure (heading markers, etc.)
        extract_code: Whether to include code blocks

    Usage example:
        >>> parser = MarkdownParser()
        >>> docs = parser.parse("README.md")
    """

    def __init__(self, preserve_structure: bool = True, extract_code: bool = True):
        """Initialize Markdown parser

        Args:
            preserve_structure: Whether to preserve Markdown structure
            extract_code: Whether to include code blocks
        """
        self.preserve_structure = preserve_structure
        self.extract_code = extract_code
        self.has_advanced_parser = MARKDOWN_AVAILABLE

    def parse(self, file_path: str | Path, **_kwargs) -> list[Document]:
        """Parse Markdown file

        Args:
            file_path: Markdown file path
            **kwargs: Additional parameters

        Returns:
            A list containing a single Document

        Raises:
            FileNotFoundError: File does not exist
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        logger.info(f"Parsing Markdown file: {path}")

        content = path.read_text(encoding="utf-8")

        # If advanced parser is available, use structured parsing
        if self.has_advanced_parser and self.preserve_structure:
            parsed_content = self._parse_structured(content)
        else:
            # Simple parsing: only basic cleanup
            parsed_content = self._parse_simple(content)

        doc = Document(
            page_content=parsed_content,
            metadata={
                "source": str(path.absolute()),
                "filename": path.name,
                "extension": path.suffix,
                "parser": "MarkdownParser",
                "structured": self.has_advanced_parser and self.preserve_structure,
            },
        )

        logger.info(f"Parsed Markdown: {len(parsed_content)} characters")
        return [doc]

    def _parse_structured(self, content: str) -> str:
        """Structured parsing of Markdown

        Uses markdown + BeautifulSoup to convert to structured text
        """
        # Convert Markdown to HTML
        html_content = markdown.markdown(
            content, extensions=["extra", "codehilite", "tables", "toc", "fenced_code"]
        )

        soup = BeautifulSoup(html_content, "html.parser")
        text_parts = []

        for element in soup.children:
            if not element.name:
                continue

            # Headings
            if element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                level = int(element.name[1])
                text_parts.append("#" * level + " " + element.get_text().strip())

            # Paragraphs
            elif element.name == "p":
                text = element.get_text().strip()
                if text:
                    text_parts.append(text)

            # Lists
            elif element.name in ["ul", "ol"]:
                for li in element.find_all("li"):
                    text_parts.append(f"* {li.get_text().strip()}")

            # Code blocks
            elif element.name == "pre" and self.extract_code:
                code_block = element.get_text().strip()
                if code_block:
                    text_parts.append(f"```\n{code_block}\n```")

            # Tables
            elif element.name == "table":
                table_str = self._extract_table(element)
                if table_str:
                    text_parts.append(table_str)

            # Other elements
            elif element.name:
                text = element.get_text(separator=" ", strip=True)
                if text:
                    text_parts.append(text)

        # Clean up extra empty lines
        cleaned_text = re.sub(r"\n\s*\n", "\n\n", "\n".join(text_parts))
        return cleaned_text.strip()

    def _parse_simple(self, content: str) -> str:
        """Simple parsing of Markdown

        Only basic cleanup, preserves Markdown syntax
        """
        lines = []

        for line in content.split("\n"):
            line = line.rstrip()

            # Skip code blocks (if not extracting code)
            if not self.extract_code and line.strip().startswith("```"):
                continue

            lines.append(line)

        return "\n".join(lines)

    def _extract_table(self, table_element) -> str:
        """Extract table to Markdown format"""
        headers = [th.get_text().strip() for th in table_element.find_all("th")]
        rows = []

        for tr in table_element.find_all("tr"):
            cells = [td.get_text().strip() for td in tr.find_all("td")]
            if cells:
                rows.append(cells)

        if not headers and not rows:
            return ""

        table_lines = []
        if headers:
            table_lines.append(" | ".join(headers))
            table_lines.append(" | ".join(["---"] * len(headers)))

        for row_cells in rows:
            padded_cells = (
                row_cells + [""] * (len(headers) - len(row_cells)) if headers else row_cells
            )
            table_lines.append(" | ".join(padded_cells))

        return "\n".join(table_lines)
