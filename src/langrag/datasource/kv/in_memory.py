from typing import Any, List, Optional
from .base import BaseKVStore

class InMemoryKV(BaseKVStore):
    """
    Simple In-Memory Key-Value Store.
    Not persistent.
    """

    def __init__(self):
        self._store = {}

    def mget(self, keys: List[str]) -> List[Optional[Any]]:
        return [self._store.get(k) for k in keys]

    def mset(self, data: dict[str, Any]) -> None:
        self._store.update(data)

    def delete(self, keys: List[str]) -> None:
        for k in keys:
            self._store.pop(k, None)
