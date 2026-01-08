import pytest
from typing import List, Dict, Any
from langrag.llm.base import BaseLLM
from langrag.entities.dataset import Dataset
from langrag.entities.document import Document
from langrag.retrieval.workflow import RetrievalWorkflow
from langrag.retrieval.router.llm_router import LLMRouter
from langrag.retrieval.rewriter.llm_rewriter import LLMRewriter
from unittest.mock import MagicMock, patch

# --- 1. Web / Application Layer Implementation ---

class MockWebLLM(BaseLLM):
    """
    This class simulates an implementation residing in the Web App layer.
    It wraps the actual model calls (e.g. to OpenAI, Anthropic, or local VLLM).
    LangRAG does not know about this class, it only sees BaseLLM.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        # In a real app, you would init openai.Client here
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Simulate embedding call
        return [[0.1] * 768 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        # Simulate embedding call
        return [0.1] * 768

    def chat(self, messages: list[dict], **kwargs) -> str:
        # Simulate LLM intelligence
        last_msg = messages[-1]["content"]
        
        # Simulate Query Rewrite logic
        if "Rewrite" in last_msg or "optimizer" in last_msg:
            return "distributed system consensus algorithms" # Simulated rewritten query
            
        # Simulate Router logic
        if "routing agent" in last_msg:
            return '```json\n{"dataset_names": ["engineering_docs"]}\n```'
            
        return "I am a helpful assistant."

    def stream_chat(self, messages: list[dict], **kwargs):
        # Simulate streaming
        yield "I "
        yield "am "
        yield "a "
        yield "helpful "
        yield "assistant."


# --- 2. Integration Test: Injecting Web LLM into LangRAG ---

class TestWebLLMInjection:
    
    def test_end_to_end_injection(self):
        # A. Web App initializes its LLM provider
        # This is where the API key is managed, outside of LangRAG
        my_secret_key = "sk-xxxxxxxx"
        web_llm = MockWebLLM(api_key=my_secret_key)
        
        # B. Web App configures LangRAG components using the injected LLM
        # 1. Create Router with injected LLM
        router = LLMRouter(llm=web_llm)
        
        # 2. Create Rewriter with injected LLM
        rewriter = LLMRewriter(llm=web_llm)
        
        # 3. Create Workflow
        workflow = RetrievalWorkflow(
            router=router,
            rewriter=rewriter,
            # vector_store_cls is injected or defaults can be used
            # Here we mock the service to focus on LLM flow
        )
        
        # C. Run RAG
        # We mock RetrievalService just to avoid needing a real DB for this test
        # but we verify Router and Rewriter (which use the LLM) are called.
        
        with patch("langrag.retrieval.workflow.RetrievalService") as mock_service:
            # Setup mock return for retrieval
            mock_service.retrieve.return_value = [
                Document(page_content="Paxos is a consensus algorithm.", metadata={"score": 0.9})
            ]
            
            # Setup Datasets
            ds1 = Dataset(name="engineering_docs", collection_name="eng")
            ds2 = Dataset(name="hr_docs", collection_name="hr")
            
            # user query
            user_query = "explain paxos"
            
            # Execute
            results = workflow.retrieve(user_query, [ds1, ds2])
            
            # D. Verification
            
            # 1. Verify Rewriter worked (called WebLLM internally)
            # The WebLLM mock simulates rewriting "explain paxos" -> "distributed system consensus algorithms"
            # So RetrievalService should have been called with the REWRITTEN query.
            call_args = mock_service.retrieve.call_args
            assert call_args is not None
            actual_query_used = call_args[1]['query']
            assert actual_query_used == "distributed system consensus algorithms"
            
            # 2. Verify Router worked (called WebLLM internally)
            # The WebLLM mock returns "engineering_docs" JSON.
            # So only ds1 should be passed to RetrievalService
            datasets_used = call_args[1]['dataset'] # retrieve is called per dataset in loop
            # Wait, workflow loops over selected datasets.
            # We assert that retrieve was called for 'engineering_docs'
            # and NOT for 'hr_docs'
            
            # Let's inspect all calls to retrieve
            called_dataset_names = [c[1]['dataset'].name for c in mock_service.retrieve.call_args_list]
            assert "engineering_docs" in called_dataset_names
            assert "hr_docs" not in called_dataset_names
            
            print("\n✅ Injection Successful: WebLLM logic drove the RAG flow!")
