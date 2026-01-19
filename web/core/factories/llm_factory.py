from typing import Any
from langrag.llm.base import BaseLLM
from loguru import logger
import os

class LLMFactory:
    """
    Factory for creating LLM instances.
    """
    
    @staticmethod
    def create(config: dict[str, Any]) -> BaseLLM:
        """
        Create an LLM instance based on configuration.
        
        Args:
            config: Configuration dictionary.
            
        Returns:
            BaseLLM instance.
            
        Raises:
            ValueError: If configuration is invalid.
        """
        try:
            llm_type = config.get("type", "remote")

            if llm_type == "mock":
                # Mock LLM for testing - returns predefined responses
                from langrag.llm.base import BaseLLM

                class MockLLM(BaseLLM):
                    """Mock LLM for testing purposes."""

                    def embed_documents(self, texts: list[str]) -> list[list[float]]:
                        """Return mock embeddings."""
                        return [[0.1] * 384 for _ in texts]

                    def embed_query(self, text: str) -> list[float]:
                        """Return mock query embedding."""
                        return [0.1] * 384

                    def chat(self, messages: list[dict], **kwargs) -> str:
                        """Return mock chat response."""
                        last_content = messages[-1].get("content", "") if messages else ""
                        return f"这是对问题 '{last_content}' 的模拟回答。"

                    def stream_chat(self, messages: list[dict], **kwargs):
                        """Yield mock chat response tokens."""
                        response = self.chat(messages, **kwargs)
                        for word in response.split():
                            yield word + " "

                return MockLLM()

            elif llm_type == "local":
                from langrag.llm.providers.local import LocalLLM
                # Default to smaller model for testing, fallback to larger model
                DEFAULT_MODEL_PATHS = [
                    "~/models/qwen2-0_5b-instruct-q4_k_m.gguf",  # ~300MB test model
                    "~/models/qwen2.5-7b-instruct-q4_k_m.gguf"  # ~4GB full model
                ]

                model_path = config.get("model_path")
                if model_path:
                    model_path = os.path.expanduser(model_path)

                if not model_path:
                    # Try smaller model first, then fallback to larger model
                    for path in DEFAULT_MODEL_PATHS:
                        expanded_path = os.path.expanduser(path)
                        if os.path.exists(expanded_path):
                            model_path = expanded_path
                            break
                    else:
                        # No model found, use first default and let it fail with clear error
                        model_path = os.path.expanduser(DEFAULT_MODEL_PATHS[0])
                        logger.warning(f"No local model found, will try: {model_path}")

                logger.info(f"Loading local LLM from: {model_path}")
                return LocalLLM(
                    model_path=model_path,
                    n_ctx=config.get("max_tokens", 2048)
                )
            
            elif llm_type == "remote" or (config.get("base_url") and config.get("api_key")):
                from openai import AsyncOpenAI
                from langrag.llm.providers.openai import OpenAILLM
                
                # Check required fields for remote
                # Check required fields for remote
                base_url = config.get("base_url", "")
                if not base_url:
                     base_url = None
                elif not base_url.startswith("http"):
                     base_url = "https://" + base_url

                client = AsyncOpenAI(
                    api_key=config.get("api_key") or "dummy", 
                    base_url=base_url
                )
                return OpenAILLM(
                    client, 
                    model=config.get("model", "default")
                )
            
            else:
                 # Fallback/Error
                 raise ValueError("Invalid LLM configuration: must specify type='local' or provide API credentials for remote.")
                 
        except ImportError as e:
            logger.error(f"Failed to import required module for LLM creation: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to create LLM: {e}")
            raise
