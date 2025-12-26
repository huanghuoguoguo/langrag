"""Simple text file parser implementation."""

from pathlib import Path
from loguru import logger

from ..base import BaseParser
from ...core.document import Document


class SimpleTextParser(BaseParser):
    """Parser for plain text files.

    This parser reads .txt files and returns a single Document
    containing the entire file content.

    Attributes:
        encoding: Character encoding to use (default: utf-8)
    """

    def __init__(self, encoding: str = "utf-8"):
        """Initialize the text parser.

        Args:
            encoding: Character encoding for reading files
        """
        self.encoding = encoding

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

        content = path.read_text(encoding=self.encoding)

        doc = Document(
            content=content,
            metadata={
                "source": str(path.absolute()),
                "filename": path.name,
                "extension": path.suffix,
            }
        )

        logger.debug(f"Parsed document {doc.id} with {len(content)} characters")
        return [doc]
