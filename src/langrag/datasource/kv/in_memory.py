from typing import Any

from .base import BaseKVStore


class InMemoryKV(BaseKVStore):
    """
    Simple In-Memory Key-Value Store.
    Not persistent.
    """

    def __init__(self):
        self._store = {}

    def mget(self, keys: list[str]) -> list[Any | None]:
        return [self._store.get(k) for k in keys]

    def mset(self, data: dict[str, Any]) -> None:
        self._store.update(data)

    def delete(self, keys: list[str]) -> None:
        for k in keys:
            self._store.pop(k, None)
