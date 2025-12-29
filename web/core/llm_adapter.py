from typing import List, Dict, Any
import httpx
from langrag.llm.base import BaseLLM
from openai import AsyncOpenAI
import logging

logger = logging.getLogger(__name__)

class WebLLMAdapter(BaseLLM):
    """
    Adapter that wraps Web App's LLM client to match LangRAG's BaseLLM interface.
    This allows LangRAG core components (Router, Rewriter) to use the LLM managed by Web App.
    """
    
    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model
        # We might need a sync client for some synchronous RAG parts if they are not async yet
        # But LangRAG core is mostly sync for now, so we need to bridge sync/async or use sync client.
        # For simplicity in this demo, we'll assume sync usage and create a sync client or bridge it.
        # Ideally LangRAG core should be async, but let's wrap it synchronously for now using httpx directly 
        # to avoid event loop issues if called from sync context.
        self.base_url = str(client.base_url)
        self.api_key = client.api_key
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Not strictly needed if we use a separate Embedder logic, 
        # but good to implement effectively using the sync client approach
        return []

    def embed_query(self, text: str) -> list[float]:
        return []

    def chat(self, messages: list[dict], **kwargs) -> str:
        """
        Sync implementation using httpx for LangRAG core compatibility.
        """
        try:
            # We use a temporary sync interactions for now as core is sync
            import httpx
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get('temperature', 0.0) # Determinstic for tools
            }
            
            # Simple sync call
            resp = httpx.post(
                f"{self.base_url}chat/completions",
                json=payload,
                headers=headers,
                timeout=60.0
            )
            resp.raise_for_status()
            data = resp.json()
            return data['choices'][0]['message']['content']
            
        except Exception as e:
            logger.error(f"WebLLMAdapter chat failed: {e}")
            raise e
