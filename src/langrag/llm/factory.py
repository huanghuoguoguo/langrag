"""LLM factory for creating LLM instances."""

from typing import Any
from loguru import logger

from .base import BaseLLM


class LLMFactory:
    """Factory for creating LLM instances based on type.

    This factory maintains a registry of available LLM types
    and creates instances based on string identifiers.
    """

    _registry: dict[str, type[BaseLLM]] = {
        # Future LLMs can be registered here:
        # "openai": OpenAILLM,
        # "anthropic": AnthropicLLM,
        # "local": LocalLLM,
    }

    @classmethod
    def create(cls, llm_type: str, **params: Any) -> BaseLLM:
        """Create an LLM instance by type.

        Args:
            llm_type: Type identifier (e.g., "openai")
            **params: Initialization parameters for the LLM

        Returns:
            LLM instance

        Raises:
            ValueError: If LLM type is not registered
        """
        if llm_type not in cls._registry:
            available = ", ".join(cls._registry.keys()) if cls._registry else "none"
            raise ValueError(
                f"Unknown LLM type: '{llm_type}'. "
                f"Available types: {available}"
            )

        llm_class = cls._registry[llm_type]
        logger.debug(f"Creating {llm_class.__name__} with params: {params}")

        return llm_class(**params)

    @classmethod
    def register(cls, llm_type: str, llm_class: type[BaseLLM]):
        """Register a new LLM type.

        Args:
            llm_type: Type identifier
            llm_class: LLM class to register

        Raises:
            TypeError: If llm_class is not a subclass of BaseLLM
        """
        if not issubclass(llm_class, BaseLLM):
            raise TypeError(
                f"{llm_class.__name__} must be a subclass of BaseLLM"
            )

        cls._registry[llm_type] = llm_class
        logger.info(f"Registered LLM type '{llm_type}': {llm_class.__name__}")

    @classmethod
    def list_types(cls) -> list[str]:
        """Get list of available LLM types.

        Returns:
            List of registered LLM type identifiers
        """
        return list(cls._registry.keys())
