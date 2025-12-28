"""HTML 文件解析器"""

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
    """HTML 文件解析器

    提取 HTML 文件的文本内容，保留结构。

    参数:
        remove_scripts: 是否移除 script 标签
        remove_styles: 是否移除 style 标签
        preserve_structure: 是否保留标题等结构

    使用示例:
        >>> parser = HtmlParser()
        >>> docs = parser.parse("page.html")
    """

    def __init__(
        self,
        remove_scripts: bool = True,
        remove_styles: bool = True,
        preserve_structure: bool = True,
    ):
        """初始化 HTML 解析器

        Args:
            remove_scripts: 移除 script 标签
            remove_styles: 移除 style 标签
            preserve_structure: 保留结构（标题等）
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
        """解析 HTML 文件

        Args:
            file_path: HTML 文件路径
            **kwargs: 额外参数

        Returns:
            包含单个 Document 的列表

        Raises:
            FileNotFoundError: 文件不存在
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

            # 移除不需要的标签
            if self.remove_scripts:
                for script in soup(["script"]):
                    script.decompose()

            if self.remove_styles:
                for style in soup(["style"]):
                    style.decompose()

            # 提取文本
            if self.preserve_structure:
                text_content = self._extract_structured(soup)
            else:
                text_content = soup.get_text(separator=" ", strip=True)

            # 清理多余空白
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
        """提取结构化文本"""
        text_parts = []

        # 优先处理 body，如果没有则处理整个文档
        root = soup.body if soup.body else soup

        for element in root.children:
            if not hasattr(element, "name") or not element.name:
                continue

            # 标题
            if element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                level = int(element.name[1])
                text_parts.append("#" * level + " " + element.get_text().strip())

            # 段落
            elif element.name == "p":
                text = element.get_text().strip()
                if text:
                    text_parts.append(text)

            # 列表
            elif element.name in ["ul", "ol"]:
                for li in element.find_all("li"):
                    text = li.get_text().strip()
                    if text:
                        text_parts.append(f"* {text}")

            # 表格
            elif element.name == "table":
                table_str = self._extract_table(element)
                if table_str:
                    text_parts.append(table_str)

            # 其他元素
            elif element.name:
                text = element.get_text(separator=" ", strip=True)
                if text:
                    text_parts.append(text)

        return "\n".join(text_parts)

    def _extract_table(self, table_element) -> str:
        """提取表格为 Markdown 格式"""
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
