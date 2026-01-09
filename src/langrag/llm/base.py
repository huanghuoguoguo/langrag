from abc import ABC, abstractmethod
from typing import Any


class BaseLLM(ABC):
    """
    Interface for a specific LLM instance (e.g., 'gpt-4', 'text-embedding-3').
    LangRAG components use this to perform actual inference/embedding.
    """

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts (for indexing)."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query (for retrieval)."""
        pass

    @abstractmethod
    def chat(self, messages: list[dict], **kwargs) -> str:
        """
        Chat completion.
        messages: [{"role": "user", "content": "..."}, ...]
        """
        pass

    @abstractmethod
    def stream_chat(self, messages: list[dict], **kwargs):
        """
        Stream chat completion.
        Returns a generator yielding tokens.
        """
        pass


class ModelManager(ABC):
    """
    Interface for managing and retrieving LLM instances.
    The host application implements this to provide LangRAG with access to configured models.
    """

    @abstractmethod
    def get_embedding_model(self, model_uid: str = None) -> BaseLLM:
        """
        Get an embedding model instance.
        If model_uid is None, return the system default.
        """
        pass

    @abstractmethod
    def get_chat_model(self, model_uid: str = None) -> BaseLLM:
        """
        Get a chat/generation model instance.
        If model_uid is None, return the system default.
        """
        pass

    @abstractmethod
    def get_rerank_model(self, model_uid: str = None) -> Any:
        # Rerank model interface might be different
        pass
