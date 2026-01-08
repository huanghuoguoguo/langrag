import pytest
from unittest.mock import MagicMock, AsyncMock
from web.core.rag_kernel import RAGKernel
from langrag.entities.document import Document

class TestStreamingChat:
    @pytest.mark.asyncio
    async def test_chat_streaming(self):
        # 1. Setup Kernel
        kernel = RAGKernel()
        
        # Mock LLM Client
        mock_llm_client = MagicMock()
        mock_stream = AsyncMock()
        
        # Async generator mock for stream
        async def mock_async_gen():
            chunks = ["Hello", " ", "World"]
            for c in chunks:
                chunk = MagicMock()
                chunk.choices[0].delta.content = c
                yield chunk
        
        mock_stream.create.return_value = mock_async_gen()
        mock_llm_client.chat.completions = mock_stream
        
        kernel.llm_client = mock_llm_client
        kernel.llm_config = {"model": "gpt-4", "temperature": 0.7, "max_tokens": 100}
        
        # Mock Retrieval (return empty for simplicity)
        kernel.multi_search = MagicMock(return_value=([], "none"))
        
        # 2. Execute
        generator = await kernel.chat(
            kb_ids=[], 
            query="test", 
            stream=True
        )
        
        # 3. Consume Generator
        items = []
        async for item in generator:
            items.append(item)
            
        # 4. Assert
        # First item should be sources (empty list)
        import json
        first = json.loads(items[0])
        assert first["type"] == "sources"
        assert first["data"] == []
        
        # Subsequent items should be content
        content = ""
        for item_str in items[1:]:
            item = json.loads(item_str)
            if item["type"] == "content":
                content += item["data"]
            elif item["type"] == "error":
                pytest.fail(f"Stream error: {item['data']}")
                
        assert content == "Hello World"
