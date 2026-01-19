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

    def chat_dict(self, messages: list[dict], **kwargs) -> dict:
        """
        Chat completion returning full message dict (for tool calls).
        Default implementation wraps chat() result.
        """
        content = self.chat(messages, **kwargs)
        return {"role": "assistant", "content": content}

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

    This interface supports both simple model retrieval and advanced stage-based configuration,
    allowing different LLM models to be used for different RAG pipeline stages.
    """

    @abstractmethod
    def get_model(self, name: str = None) -> BaseLLM | None:
        """
        Get an LLM instance by name.
        If name is None, return the default model.

        Args:
            name: Model name to retrieve. If None, returns default model.

        Returns:
            BaseLLM instance or None if not found.
        """
        pass

    @abstractmethod
    def list_models(self) -> list[str]:
        """
        List all available model names.

        Returns:
            List of registered model names.
        """
        pass

    @abstractmethod
    def get_stage_model(self, stage: str) -> BaseLLM | None:
        """
        Get the LLM configured for a specific stage.

        Args:
            stage: Stage name (e.g., "chat", "router", "rewriter")

        Returns:
            BaseLLM instance configured for the stage, or None if not configured.
        """
        pass

    @abstractmethod
    def set_stage_model(self, stage: str, model_name: str) -> None:
        """
        Configure an LLM for a specific stage.

        Args:
            stage: Stage name (e.g., "chat", "router", "rewriter")
            model_name: Name of the model to assign to this stage

        Raises:
            ValueError: If stage or model_name is invalid
        """
        pass

    @abstractmethod
    def get_stage_model_name(self, stage: str) -> str | None:
        """
        Get the model name configured for a specific stage.

        Args:
            stage: Stage name

        Returns:
            Model name configured for the stage, or None if not configured.
        """
        pass

    @abstractmethod
    def list_stages(self) -> list[str]:
        """
        List all available stages.

        Returns:
            List of stage names.
        """
        pass

    # Legacy compatibility methods (deprecated but kept for backward compatibility)
    def get_embedding_model(self, model_uid: str = None) -> BaseLLM:
        """
        [DEPRECATED] Use get_model() instead.
        Get an embedding model instance.
        If model_uid is None, return the system default.
        """
        return self.get_model(model_uid)

    def get_chat_model(self, model_uid: str = None) -> BaseLLM:
        """
        [DEPRECATED] Use get_stage_model("chat") or get_model() instead.
        Get a chat/generation model instance.
        If model_uid is None, return the system default.
        """
        chat_model = self.get_stage_model("chat")
        if chat_model:
            return chat_model
        return self.get_model(model_uid)

    def get_rerank_model(self, model_uid: str = None) -> Any:
        """
        [DEPRECATED] Use get_stage_model("reranker") instead.
        Rerank model interface - kept for compatibility.
        """
        return self.get_stage_model("reranker")
