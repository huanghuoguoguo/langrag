"""Base parser interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from ..core.document import Document


class BaseParser(ABC):
    """Abstract base class for document parsers.

    Parsers are responsible for reading files and converting them
    into Document objects.
    """

    @abstractmethod
    def parse(self, file_path: str | Path, **kwargs) -> list[Document]:
        """Parse a file and return document(s).

        Args:
            file_path: Path to the file to parse
            **kwargs: Implementation-specific options

        Returns:
            List of parsed documents

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        pass
