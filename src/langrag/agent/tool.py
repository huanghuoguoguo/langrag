from dataclasses import dataclass
from typing import Any, Callable, Awaitable

@dataclass
class Tool:
    """
    Definition of a tool that can be used by an agent.
    """
    name: str
    description: str
    func: Callable[..., Awaitable[Any]]
    parameters: dict[str, Any]  # JSON Schema for arguments
