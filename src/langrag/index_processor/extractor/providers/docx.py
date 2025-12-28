"""DOCX 文件解析器"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

try:
    from docx import Document as DocxDocument

    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logger.warning("python-docx not installed. DOCX parsing will not be available.")

from langrag.entities.document import Document
from ..base import BaseParser


class DocxParser(BaseParser):
    """DOCX 文件解析器

    使用 python-docx 提取 Word 文档的文本内容。

    参数:
        include_tables: 是否包含表格内容
        include_headers: 是否包含页眉内容
        include_footers: 是否包含页脚内容

    使用示例:
        >>> parser = DocxParser()
        >>> docs = parser.parse("document.docx")
    """

    def __init__(
        self,
        include_tables: bool = True,
        include_headers: bool = False,
        include_footers: bool = False,
    ):
        """初始化 DOCX 解析器

        Args:
            include_tables: 是否包含表格
            include_headers: 是否包含页眉
            include_footers: 是否包含页脚
        """
        if not DOCX_AVAILABLE:
            raise ImportError(
                "python-docx is required for DOCX parsing. Install it with: pip install python-docx"
            )

        self.include_tables = include_tables
        self.include_headers = include_headers
        self.include_footers = include_footers

    def parse(self, file_path: str | Path, **_kwargs) -> list[Document]:
        """解析 DOCX 文件

        Args:
            file_path: DOCX 文件路径
            **kwargs: 额外参数

        Returns:
            包含单个 Document 的列表

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不是有效的 DOCX 文件
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        logger.info(f"Parsing DOCX file: {path}")

        try:
            doc = DocxDocument(str(path))
            text_content = []

            # 提取页眉
            if self.include_headers:
                for section in doc.sections:
                    header = section.header
                    for paragraph in header.paragraphs:
                        if paragraph.text.strip():
                            text_content.append(f"[Header] {paragraph.text}")

            # 提取段落
            paragraph_count = 0
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text)
                    paragraph_count += 1

            # 提取表格
            table_count = 0
            if self.include_tables:
                for table in doc.tables:
                    table_text = self._extract_table(table)
                    if table_text:
                        text_content.append(table_text)
                        table_count += 1

            # 提取页脚
            if self.include_footers:
                for section in doc.sections:
                    footer = section.footer
                    for paragraph in footer.paragraphs:
                        if paragraph.text.strip():
                            text_content.append(f"[Footer] {paragraph.text}")

            content = "\n".join(text_content)

            if not content.strip():
                logger.warning(f"No text content extracted from {path}")

            doc_obj = Document(
                page_content=content,
                metadata={
                    "source": str(path.absolute()),
                    "filename": path.name,
                    "extension": path.suffix,
                    "paragraphs": paragraph_count,
                    "tables": table_count,
                    "parser": "DocxParser",
                },
            )

            logger.info(
                f"Parsed DOCX: {paragraph_count} paragraphs, "
                f"{table_count} tables, {len(content)} characters"
            )

            return [doc_obj]

        except Exception as e:
            logger.error(f"Failed to parse DOCX {path}: {e}")
            raise ValueError(f"Invalid DOCX file: {e}") from e

    def _extract_table(self, table) -> str:
        """提取表格内容为 Markdown 格式

        Args:
            table: python-docx Table 对象

        Returns:
            Markdown 格式的表格字符串
        """
        lines = []

        for i, row in enumerate(table.rows):
            cells = [cell.text.strip() for cell in row.cells]
            lines.append(" | ".join(cells))

            # 在第一行后添加分隔符
            if i == 0:
                lines.append(" | ".join(["---"] * len(cells)))

        return "\n".join(lines) if lines else ""
