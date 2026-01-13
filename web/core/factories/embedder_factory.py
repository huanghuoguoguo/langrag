from typing import Any
from langrag import BaseEmbedder
from loguru import logger
from web.core.embedders import SeekDBEmbedder, WebOpenAIEmbedder

class EmbedderFactory:
    """
    Factory for creating Embedder instances.
    """
    
    @staticmethod
    def create(embedder_type: str, config: dict[str, Any]) -> BaseEmbedder:
        """
        Create an Embedder instance.
        
        Args:
            embedder_type: Type of embedder ("openai", "seekdb", etc.)
            config: Configuration dictionary.
            
        Returns:
            BaseEmbedder instance.
        """
        if embedder_type == "openai":
            base_url = config.get("base_url")
            api_key = config.get("api_key")
            model = config.get("model")
            
            if not base_url or not api_key or not model:
                raise ValueError("OpenAI embedder requires base_url, api_key and model")
                
            logger.info(f"OpenAI-compatible embedder configured: {model}")
            return WebOpenAIEmbedder(base_url, api_key, model)

        elif embedder_type == "seekdb":
            logger.info("SeekDB embedder configured (all-MiniLM-L6-v2)")
            return SeekDBEmbedder()

        else:
            raise ValueError(f"Unsupported embedder type: {embedder_type}")
