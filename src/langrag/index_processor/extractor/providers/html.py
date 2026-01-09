"""HTML file parser"""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

try:
    from bs4 import BeautifulSoup

    HTML_AVAILABLE = True
except ImportError:
    HTML_AVAILABLE = False
    logger.warning("beautifulsoup4 not installed. HTML parsing will not be available.")

from langrag.entities.document import Document

from ..base import BaseParser


class HtmlParser(BaseParser):
    """HTML file parser

    Extracts text content from HTML files while preserving structure.

    Args:
        remove_scripts: Whether to remove script tags
        remove_styles: Whether to remove style tags
        preserve_structure: Whether to preserve structure like headings

    Usage example:
        >>> parser = HtmlParser()
        >>> docs = parser.parse("page.html")
    """

    def __init__(
        self,
        remove_scripts: bool = True,
        remove_styles: bool = True,
        preserve_structure: bool = True,
    ):
        """Initialize HTML parser

        Args:
            remove_scripts: Remove script tags
            remove_styles: Remove style tags
            preserve_structure: Preserve structure (headings, etc.)
        """
        if not HTML_AVAILABLE:
            raise ImportError(
                "beautifulsoup4 is required for HTML parsing. "
                "Install it with: pip install beautifulsoup4"
            )

        self.remove_scripts = remove_scripts
        self.remove_styles = remove_styles
        self.preserve_structure = preserve_structure

    def parse(self, file_path: str | Path, **_kwargs) -> list[Document]:
        """Parse HTML file

        Args:
            file_path: HTML file path
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

        logger.info(f"Parsing HTML file: {path}")

        try:
            html_content = path.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove unwanted tags
            if self.remove_scripts:
                for script in soup(["script"]):
                    script.decompose()

            if self.remove_styles:
                for style in soup(["style"]):
                    style.decompose()

            # Extract text
            if self.preserve_structure:
                text_content = self._extract_structured(soup)
            else:
                text_content = soup.get_text(separator=" ", strip=True)

            # Clean up extra whitespace
            text_content = re.sub(r"\n\s*\n", "\n\n", text_content)
            text_content = text_content.strip()

            if not text_content:
                logger.warning(f"No text content extracted from {path}")

            doc = Document(
                page_content=text_content,
                metadata={
                    "source": str(path.absolute()),
                    "filename": path.name,
                    "extension": path.suffix,
                    "parser": "HtmlParser",
                    "title": soup.title.string if soup.title else None,
                },
            )

            logger.info(f"Parsed HTML: {len(text_content)} characters")
            return [doc]

        except Exception as e:
            logger.error(f"Failed to parse HTML {path}: {e}")
            raise ValueError(f"Invalid HTML file: {e}") from e

    def _extract_structured(self, soup: BeautifulSoup) -> str:
        """Extract structured text"""
        text_parts = []

        # Prioritize body, otherwise process entire document
        root = soup.body if soup.body else soup

        for element in root.children:
            if not hasattr(element, "name") or not element.name:
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
                    text = li.get_text().strip()
                    if text:
                        text_parts.append(f"* {text}")

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

        return "\n".join(text_parts)

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
