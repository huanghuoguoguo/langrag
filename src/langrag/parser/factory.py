"""Parser factory for creating parser instances."""

from typing import Any
from loguru import logger

from .base import BaseParser
from .providers.simple_text import SimpleTextParser

# 可选依赖的 parsers
try:
    from .providers.pdf import PdfParser
    PDF_PARSER_AVAILABLE = True
except ImportError:
    PDF_PARSER_AVAILABLE = False

try:
    from .providers.docx import DocxParser
    DOCX_PARSER_AVAILABLE = True
except ImportError:
    DOCX_PARSER_AVAILABLE = False

try:
    from .providers.markdown import MarkdownParser
    MARKDOWN_PARSER_AVAILABLE = True
except ImportError:
    MARKDOWN_PARSER_AVAILABLE = False

try:
    from .providers.html import HtmlParser
    HTML_PARSER_AVAILABLE = True
except ImportError:
    HTML_PARSER_AVAILABLE = False


class ParserFactory:
    """Parser 工厂
    
    根据类型创建 parser 实例。
    支持的格式取决于已安装的依赖。
    
    基础支持（无额外依赖）：
    - simple_text: 纯文本文件
    
    扩展支持（需要安装依赖）：
    - pdf: PDF 文件（需要 PyPDF2）
    - docx: Word 文档（需要 python-docx）
    - markdown: Markdown 文件（需要 markdown, beautifulsoup4）
    - html: HTML 文件（需要 beautifulsoup4）
    """

    _registry: dict[str, type[BaseParser]] = {
        "simple_text": SimpleTextParser,
        "text": SimpleTextParser,  # 别名
        "txt": SimpleTextParser,   # 别名
    }
    
    # 动态注册可用的 parsers
    if PDF_PARSER_AVAILABLE:
        _registry["pdf"] = PdfParser
    
    if DOCX_PARSER_AVAILABLE:
        _registry["docx"] = DocxParser
        _registry["doc"] = DocxParser  # 别名（虽然不完全支持 .doc）
    
    if MARKDOWN_PARSER_AVAILABLE:
        _registry["markdown"] = MarkdownParser
        _registry["md"] = MarkdownParser  # 别名
    
    if HTML_PARSER_AVAILABLE:
        _registry["html"] = HtmlParser
        _registry["htm"] = HtmlParser  # 别名

    @classmethod
    def create(cls, parser_type: str, **params: Any) -> BaseParser:
        """Create a parser instance by type.

        Args:
            parser_type: Type identifier (e.g., "simple_text")
            **params: Initialization parameters for the parser

        Returns:
            Parser instance

        Raises:
            ValueError: If parser type is not registered
        """
        if parser_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown parser type: '{parser_type}'. "
                f"Available types: {available}"
            )

        parser_class = cls._registry[parser_type]
        logger.debug(f"Creating {parser_class.__name__} with params: {params}")

        return parser_class(**params)

    @classmethod
    def register(cls, parser_type: str, parser_class: type[BaseParser]):
        """Register a new parser type.

        Args:
            parser_type: Type identifier
            parser_class: Parser class to register

        Raises:
            TypeError: If parser_class is not a subclass of BaseParser
        """
        if not issubclass(parser_class, BaseParser):
            raise TypeError(
                f"{parser_class.__name__} must be a subclass of BaseParser"
            )

        cls._registry[parser_type] = parser_class
        logger.info(f"Registered parser type '{parser_type}': {parser_class.__name__}")

    @classmethod
    def list_types(cls) -> list[str]:
        """Get list of available parser types.

        Returns:
            List of registered parser type identifiers
        """
        return list(cls._registry.keys())
