import logging
from typing import Any, Generator

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from langrag.llm.base import BaseLLM

logger = logging.getLogger(__name__)

class LocalLLM(BaseLLM):
    """
    Local LLM implementation using llama.cpp.
    Optimized for running small models (SLM) locally for auxiliary tasks like rewriting.
    """

    def __init__(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = -1, verbose: bool = False):
        if Llama is None:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Please install it using `pip install llama-cpp-python`."
            )
        
        self.model_path = model_path
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose
            )
            logger.info(f"LocalLLM loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load LocalLLM: {e}")
            raise

    @property
    def model(self) -> str:
        """Return the model name (filename)."""
        import os
        return os.path.basename(self.model_path)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Not implemented for chat-optimized local models."""
        return []

    def embed_query(self, text: str) -> list[float]:
        """Not implemented for chat-optimized local models."""
        return []

    def chat(self, messages: list[dict], **kwargs) -> str:
        """
        Chat completion using local model.
        
        Args:
            messages: List of message dicts [{"role": "user", "content": "..."}]
            **kwargs:
                - max_tokens: int (default: 512, shorter for aux tasks)
                - temperature: float (default: 0.1, lower for deterministic tasks)
        """
        max_tokens = kwargs.get("max_tokens", 512)
        temperature = kwargs.get("temperature", 0.1)
        
        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            # Response validation
            if not response or "choices" not in response:
                return ""
            
            return response["choices"][0]["message"]["content"] or ""
            
        except Exception as e:
            logger.error(f"LocalLLM chat failed: {e}")
            raise

    def stream_chat(self, messages: list[dict], **kwargs) -> Generator[str, None, None]:
        """
        Stream chat completion.
        """
        max_tokens = kwargs.get("max_tokens", 512)
        temperature = kwargs.get("temperature", 0.1)
        
        try:
            stream = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            
            for chunk in stream:
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    yield delta["content"]
                    
        except Exception as e:
            logger.error(f"LocalLLM stream failed: {e}")
            raise
