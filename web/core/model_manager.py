from typing import Dict, Optional, Any
from langrag.llm.base import BaseLLM
from loguru import logger

class WebModelManager:
    """
    Manages a pool of LLM instances for the Web Application.
    
    This class acts as a central registry for all available Large Language Models (LLMs),
    whether they are remote APIs (OpenAI, Claude) or local models (Llama.cpp).
    
    It allows the application to:
    1. Register multiple models with unique names.
    2. Retrieve specific model instances by name.
    3. Manage the lifecycle and configuration of these models independently of the business logic.
    """

    def __init__(self):
        self._models: Dict[str, BaseLLM] = {}
        self._default_model_name: Optional[str] = None

    def register_model(self, name: str, llm_instance: BaseLLM, set_as_default: bool = False) -> None:
        """
        Register a new LLM instance.

        Args:
            name: Unique identifier for the model (e.g., "gpt-4", "qwen-local").
            llm_instance: The initialized LLM instance (BaseLLM subclass).
            set_as_default: Whether to set this model as the default.
        """
        if name in self._models:
            logger.warning(f"Overwriting existing model registration: {name}")
        
        self._models[name] = llm_instance
        logger.info(f"Registered LLM: {name} ({type(llm_instance).__name__})")
        
        if set_as_default or self._default_model_name is None:
            self._default_model_name = name
            logger.info(f"Set default LLM to: {name}")

    def get_model(self, name: str = None) -> Optional[BaseLLM]:
        """
        Retrieve an LLM instance by name.
        
        Args:
            name: The name of the model to retrieve. If None, returns the default model.
        
        Returns:
            The BaseLLM instance, or None if not found.
        """
        target_name = name or self._default_model_name
        if not target_name:
            return None
            
        return self._models.get(target_name)

    def list_models(self) -> list[str]:
        """List all registered model names."""
        return list(self._models.keys())

    def remove_model(self, name: str) -> None:
        """Unregister a model."""
        if name in self._models:
            del self._models[name]
            # Reset default if we deleted it
            if self._default_model_name == name:
                self._default_model_name = next(iter(self._models)) if self._models else None
