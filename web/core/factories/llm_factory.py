from typing import Any
from langrag.llm.base import BaseLLM
from loguru import logger

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
                from langrag.entities.query import Query
                import asyncio

                class MockLLM(BaseLLM):
                    def __init__(self):
                        pass

                    async def generate(self, query: str | Query, **kwargs) -> str:
                        query_text = query.text if isinstance(query, Query) else query
                        # Return a simple mock response
                        return f"这是对问题 '{query_text}' 的模拟回答。在实际部署中，这里会返回真实的LLM生成的结果。"

                    async def generate_stream(self, query: str | Query, **kwargs):
                        query_text = query.text if isinstance(query, Query) else query
                        response = f"这是对问题 '{query_text}' 的模拟回答。在实际部署中，这里会返回真实的LLM生成的结果。"
                        for word in response.split():
                            yield word + " "
                            await asyncio.sleep(0.1)

                return MockLLM()

            elif llm_type == "local":
                from langrag.llm.providers.local import LocalLLM
                # Default to smaller model for testing, fallback to larger model
                DEFAULT_MODEL_PATHS = [
                    "/home/yhh/models/qwen2-0_5b-instruct-q4_k_m.gguf",  # ~300MB test model
                    "/home/yhh/models/qwen2.5-7b-instruct-q4_k_m.gguf"  # ~4GB full model
                ]

                model_path = config.get("model_path")
                if not model_path:
                    # Try smaller model first, then fallback to larger model
                    import os
                    for path in DEFAULT_MODEL_PATHS:
                        if os.path.exists(path):
                            model_path = path
                            break
                    else:
                        # No model found, use first default and let it fail with clear error
                        model_path = DEFAULT_MODEL_PATHS[0]
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
