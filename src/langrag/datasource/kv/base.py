from abc import ABC, abstractmethod
from typing import Any, List, Optional

class BaseKVStore(ABC):
    """
    Abstract Base Class for Key-Value Stores.
    Used for storing Parent Documents in Parent-Child Indexing.
    """

    @abstractmethod
    def mget(self, keys: List[str]) -> List[Optional[Any]]:
        """Get multiple values."""
        pass

    @abstractmethod
    def mset(self, data: dict[str, Any]) -> None:
        """Set multiple values."""
        pass

    @abstractmethod
    def delete(self, keys: List[str]) -> None:
        """Delete multiple keys."""
        pass
        
    def get(self, key: str) -> Optional[Any]:
        """Get single value."""
        results = self.mget([key])
        return results[0] if results else None
        
    def set(self, key: str, value: Any) -> None:
        """Set single value."""
        self.mset({key: value})
