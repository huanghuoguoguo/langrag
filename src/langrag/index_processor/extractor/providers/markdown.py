"""Markdown 文件解析器"""

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

from ...core.document import Document
from ..base import BaseParser


class MarkdownParser(BaseParser):
    """Markdown 文件解析器

    将 Markdown 转换为结构化的纯文本。
    支持标题、列表、代码块、表格等。

    参数:
        preserve_structure: 是否保留 Markdown 结构（标题标记等）
        extract_code: 是否包含代码块

    使用示例:
        >>> parser = MarkdownParser()
        >>> docs = parser.parse("README.md")
    """

    def __init__(self, preserve_structure: bool = True, extract_code: bool = True):
        """初始化 Markdown 解析器

        Args:
            preserve_structure: 是否保留 Markdown 结构
            extract_code: 是否包含代码块
        """
        self.preserve_structure = preserve_structure
        self.extract_code = extract_code
        self.has_advanced_parser = MARKDOWN_AVAILABLE

    def parse(self, file_path: str | Path, **_kwargs) -> list[Document]:
        """解析 Markdown 文件

        Args:
            file_path: Markdown 文件路径
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

        logger.info(f"Parsing Markdown file: {path}")

        content = path.read_text(encoding="utf-8")

        # 如果有高级解析器，使用结构化解析
        if self.has_advanced_parser and self.preserve_structure:
            parsed_content = self._parse_structured(content)
        else:
            # 简单解析：只做基本清理
            parsed_content = self._parse_simple(content)

        doc = Document(
            content=parsed_content,
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
        """结构化解析 Markdown

        使用 markdown + BeautifulSoup 转换为结构化文本
        """
        # 转换 Markdown 为 HTML
        html_content = markdown.markdown(
            content, extensions=["extra", "codehilite", "tables", "toc", "fenced_code"]
        )

        soup = BeautifulSoup(html_content, "html.parser")
        text_parts = []

        for element in soup.children:
            if not element.name:
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
                    text_parts.append(f"* {li.get_text().strip()}")

            # 代码块
            elif element.name == "pre" and self.extract_code:
                code_block = element.get_text().strip()
                if code_block:
                    text_parts.append(f"```\n{code_block}\n```")

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

        # 清理多余的空行
        cleaned_text = re.sub(r"\n\s*\n", "\n\n", "\n".join(text_parts))
        return cleaned_text.strip()

    def _parse_simple(self, content: str) -> str:
        """简单解析 Markdown

        只做基本的清理，保留 Markdown 语法
        """
        lines = []

        for line in content.split("\n"):
            line = line.rstrip()

            # 跳过代码块（如果不提取代码）
            if not self.extract_code and line.strip().startswith("```"):
                continue

            lines.append(line)

        return "\n".join(lines)

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
