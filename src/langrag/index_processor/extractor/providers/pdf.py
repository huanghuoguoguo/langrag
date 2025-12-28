"""PDF 文件解析器"""

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
    """PDF 文件解析器

    使用 pypdf 提取 PDF 文件的文本内容。

    参数:
        extract_images: 是否提取图片（暂不支持）
        pages: 解析的页码范围，None 表示全部

    使用示例:
        >>> parser = PdfParser()
        >>> docs = parser.parse("document.pdf")
    """

    def __init__(self, extract_images: bool = False, pages: tuple[int, int] | None = None):
        """初始化 PDF 解析器

        Args:
            extract_images: 是否提取图片（未实现）
            pages: (start, end) 页码范围，None 表示全部
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
        """解析 PDF 文件

        Args:
            file_path: PDF 文件路径
            **kwargs: 额外参数

        Returns:
            包含单个 Document 的列表（所有页面合并）

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不是有效的 PDF 文件
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

                # 确定要解析的页面范围
                if self.pages:
                    start, end = self.pages
                    start = max(0, start)
                    end = min(total_pages, end)
                else:
                    start, end = 0, total_pages

                logger.debug(f"Extracting pages {start} to {end} of {total_pages}")

                # 提取文本
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

                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(path.absolute()),
                        "filename": path.name,
                        "extension": path.suffix,
                        "total_pages": total_pages,
                        "extracted_pages": f"{start}-{end}",
                        "parser": "PdfParser",
                    },
                )

                logger.info(f"Parsed PDF: {total_pages} pages, {len(content)} characters extracted")

                return [doc]

        except Exception as e:
            logger.error(f"Failed to parse PDF {path}: {e}")
            raise ValueError(f"Invalid PDF file: {e}") from e
