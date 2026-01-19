"""
Component Configuration.

Provides Pydantic-based configuration validation for components.
This centralizes all validation logic, removing scattered boundary checks.
"""

from typing import Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel, ConfigDict, model_validator
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="ComponentConfig")


class ComponentConfig(BaseModel):
    """
    Base configuration class for pipeline components.
    
    All component configurations should inherit from this class.
    Use Pydantic validators to enforce constraints at initialization time,
    not at runtime.
    
    Example:
        class EmbedderConfig(ComponentConfig):
            model_name: str
            batch_size: int = 32
            
            @field_validator('batch_size')
            @classmethod
            def validate_batch_size(cls, v):
                if v < 1:
                    raise ValueError("batch_size must be positive")
                return v
    """
    
    model_config = ConfigDict(
        extra="forbid",           # Disallow unknown fields
        validate_assignment=True, # Validate on attribute assignment
        frozen=False,             # Allow mutation after creation
    )
    
    # Common fields for all components
    enabled: bool = True
    timeout: Optional[float] = None  # Timeout in seconds (None = no timeout)
    
    def merge(self, overrides: Dict[str, Any]) -> "ComponentConfig":
        """
        Create a new config with overrides applied.
        
        Args:
            overrides: Dict of field values to override.
            
        Returns:
            New config instance with overrides applied.
        """
        current = self.model_dump()
        current.update(overrides)
        return self.__class__(**current)


def validate_config(config_class: Type[T], data: Dict[str, Any]) -> T:
    """
    Validate and create a config instance.
    
    This is a utility function for validating configuration dictionaries
    against a config class.
    
    Args:
        config_class: The ComponentConfig subclass to validate against.
        data: Dictionary of configuration values.
        
    Returns:
        Validated config instance.
        
    Raises:
        pydantic.ValidationError: If validation fails.
    """
    return config_class.model_validate(data)


# === Common Config Classes ===

class ChunkerConfig(ComponentConfig):
    """Configuration for text chunking components."""
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkerConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        return self


class EmbedderConfig(ComponentConfig):
    """Configuration for embedding components."""
    model_name: str = "default"
    batch_size: int = 32
    normalize: bool = True
    
    @model_validator(mode="after")
    def validate_batch(self) -> "EmbedderConfig":
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        return self


class RetrieverConfig(ComponentConfig):
    """Configuration for retrieval components."""
    top_k: int = 5
    score_threshold: Optional[float] = None
    
    @model_validator(mode="after")
    def validate_top_k(self) -> "RetrieverConfig":
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.score_threshold is not None and not (0 <= self.score_threshold <= 1):
            raise ValueError("score_threshold must be between 0 and 1")
        return self


class LLMConfig(ComponentConfig):
    """Configuration for LLM components."""
    model_name: str = "default"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    
    @model_validator(mode="after")
    def validate_temperature(self) -> "LLMConfig":
        if not (0 <= self.temperature <= 2):
            raise ValueError("temperature must be between 0 and 2")
        return self
