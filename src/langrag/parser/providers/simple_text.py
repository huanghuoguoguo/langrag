"""Simple text file parser implementation."""

from pathlib import Path
from loguru import logger

try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False
    logger.debug("chardet not installed. Encoding detection will not be available.")

from ..base import BaseParser
from ...core.document import Document


class SimpleTextParser(BaseParser):
    """Parser for plain text files.

    This parser reads .txt files and returns a single Document
    containing the entire file content.
    
    支持自动编码检测（如果安装了 chardet）。

    Attributes:
        encoding: Character encoding to use (default: utf-8)
        auto_detect_encoding: 是否自动检测编码
    """

    def __init__(
        self,
        encoding: str = "utf-8",
        auto_detect_encoding: bool = True
    ):
        """Initialize the text parser.

        Args:
            encoding: 默认字符编码
            auto_detect_encoding: 是否自动检测编码（需要 chardet）
        """
        self.encoding = encoding
        self.auto_detect_encoding = auto_detect_encoding and CHARDET_AVAILABLE

    def parse(self, file_path: str | Path, **kwargs) -> list[Document]:
        """Parse a text file into a single document.

        Args:
            file_path: Path to the text file
            **kwargs: Additional arguments (ignored)

        Returns:
            List containing a single Document with file contents

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If path is not a file
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        logger.info(f"Parsing text file: {path}")

        # 自动检测编码
        encoding = self.encoding
        if self.auto_detect_encoding:
            try:
                raw_data = path.read_bytes()
                detected = chardet.detect(raw_data)
                if detected['encoding']:
                    encoding = detected['encoding']
                    logger.debug(f"Detected encoding: {encoding} (confidence: {detected['confidence']:.2f})")
            except Exception as e:
                logger.warning(f"Failed to detect encoding, using {self.encoding}: {e}")
                encoding = self.encoding

        # 读取文件
        try:
            content = path.read_text(encoding=encoding, errors='ignore')
        except Exception as e:
            logger.warning(f"Failed to read with encoding {encoding}, retrying with utf-8: {e}")
            content = path.read_text(encoding='utf-8', errors='ignore')
            encoding = 'utf-8'

        doc = Document(
            content=content,
            metadata={
                "source": str(path.absolute()),
                "filename": path.name,
                "extension": path.suffix,
                "encoding": encoding,
                "parser": "SimpleTextParser"
            }
        )

        logger.debug(f"Parsed document {doc.id} with {len(content)} characters")
        return [doc]
