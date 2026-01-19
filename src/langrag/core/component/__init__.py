"""
Core Component Module.

Provides the base abstractions for building modular, composable pipeline components.
"""

from .base import Component, ComponentState
from .config import ComponentConfig, validate_config
from .context import ComponentContext

__all__ = [
    "Component",
    "ComponentState",
    "ComponentConfig",
    "validate_config",
    "ComponentContext",
]
