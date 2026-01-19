"""
Embedder Manager for Web layer.

Manages multiple embedder instances, allowing each knowledge base to use
its own embedder configuration.
"""

from typing import Any

from langrag.llm.embedder.base import BaseEmbedder
from loguru import logger

from web.core.factories.embedder_factory import EmbedderFactory


class EmbedderManager:
    """
    Manages multiple embedder instances.

    Each knowledge base can have its own embedder, configured by name.
    Embedders are lazily instantiated and cached.

    Example:
        manager = EmbedderManager()

        # Register from config
        manager.register("openai-embed", "openai", {
            "base_url": "...", "api_key": "...", "model": "text-embedding-ada-002"
        })
        manager.register("local-embed", "seekdb", {})

        # Get embedder for a KB
        embedder = manager.get("openai-embed")
    """

    def __init__(self):
        # name -> embedder instance
        self._embedders: dict[str, BaseEmbedder] = {}
        # name -> config (for lazy instantiation)
        self._configs: dict[str, dict[str, Any]] = {}
        self._default_name: str | None = None

    def register(
        self,
        name: str,
        embedder_type: str,
        config: dict[str, Any],
        set_as_default: bool = False,
    ) -> None:
        """
        Register an embedder configuration.

        The embedder is instantiated lazily on first access.

        Args:
            name: Unique name for this embedder config.
            embedder_type: Type of embedder ("openai", "seekdb", etc.)
            config: Configuration dict (base_url, api_key, model, etc.)
            set_as_default: Whether to set this as the default embedder.
        """
        self._configs[name] = {
            "type": embedder_type,
            "config": config,
        }

        # Clear cached instance if exists (config might have changed)
        if name in self._embedders:
            del self._embedders[name]

        if set_as_default or self._default_name is None:
            self._default_name = name

        logger.info(f"Registered embedder config: {name} (type: {embedder_type})")

    def register_instance(
        self,
        name: str,
        embedder: BaseEmbedder,
        set_as_default: bool = False,
    ) -> None:
        """
        Register an existing embedder instance directly.

        Args:
            name: Unique name for this embedder.
            embedder: The embedder instance.
            set_as_default: Whether to set this as the default.
        """
        self._embedders[name] = embedder

        if set_as_default or self._default_name is None:
            self._default_name = name

        logger.info(f"Registered embedder instance: {name} ({type(embedder).__name__})")

    def get(self, name: str | None = None) -> BaseEmbedder | None:
        """
        Get an embedder by name.

        Instantiates the embedder lazily if not already created.

        Args:
            name: Embedder name. If None, returns the default embedder.

        Returns:
            BaseEmbedder instance or None if not found.
        """
        target_name = name or self._default_name
        if not target_name:
            return None

        # Return cached instance if exists
        if target_name in self._embedders:
            return self._embedders[target_name]

        # Try to instantiate from config
        if target_name in self._configs:
            try:
                cfg = self._configs[target_name]
                embedder = EmbedderFactory.create(cfg["type"], cfg["config"])
                self._embedders[target_name] = embedder
                logger.info(f"Instantiated embedder: {target_name}")
                return embedder
            except Exception as e:
                logger.error(f"Failed to instantiate embedder '{target_name}': {e}")
                return None

        logger.warning(f"Embedder not found: {target_name}")
        return None

    def remove(self, name: str) -> None:
        """Remove an embedder by name."""
        if name in self._embedders:
            del self._embedders[name]
        if name in self._configs:
            del self._configs[name]

        if self._default_name == name:
            self._default_name = next(iter(self._configs), None)

        logger.info(f"Removed embedder: {name}")

    def list_names(self) -> list[str]:
        """List all registered embedder names."""
        return list(self._configs.keys())

    def get_default_name(self) -> str | None:
        """Get the default embedder name."""
        return self._default_name

    def set_default(self, name: str) -> None:
        """Set the default embedder."""
        if name not in self._configs and name not in self._embedders:
            raise ValueError(f"Embedder '{name}' not registered")
        self._default_name = name
        logger.info(f"Set default embedder to: {name}")
