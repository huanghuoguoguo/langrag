import sys
from pathlib import Path
import logging
import os

# Add project root to path
sys.path.append(str(Path(__file__).parents[1]))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock sqlmodel/httpx deps if needed (usually handled by environment now)
import sys
from unittest.mock import MagicMock
sys.modules["sqlmodel"] = MagicMock()
# httpx should be installed now

from web.core.rag_kernel import RAGKernel

def test_web_rag_flow():
    print(">>> Initializing RAG Kernel...")
    kernel = RAGKernel()
    
    # 1. Create a Web Search Knowledge Base
    kb_id = "internet_kb"
    print(f">>> Creating Web Search KB: {kb_id}")
    kernel.create_vector_store(kb_id=kb_id, collection_name="web", vdb_type="web_search")
    
    # 2. Test Search (Retrieval Only) - Skipped to avoid Rate Limits
    query = "What is the release date of Python 3.13?"
    print(f">>> Testing Search for: '{query}'")
    
    # results, search_type = kernel.search(kb_id, query, top_k=3)
    # ... (skipping printing loop)

    # 3. Test Multi-Path Retrieval (Unified RAG)
    print(f"\n>>> Testing Multi-Path Retrieval (Unified RAG)...")
    # Call multi_search with just the web KB to verify the interface
    multi_results, _ = kernel.multi_search([kb_id], query, top_k=5)
    
    print(f">>> Multi-Path Results count: {len(multi_results)}")
    if len(multi_results) > 0:
        print(">>> SUCCESS: Multi-search returned results.")
        
        # Verify provider type
        first_source = multi_results[0].metadata.get('source')
        print(f">>> Active Provider seems to be: {first_source}")
        
        if "duckduckgo" in str(first_source) or "duckduckgo" in str(multi_results[0].metadata.get('link', '')):
             print(">>> CONFIRMED: Using DuckDuckGo Search Provider")
        elif "mock" in str(first_source):
             print(">>> WARNING: Using Mock Provider (DDG might have failed or not installed)")

if __name__ == "__main__":
    test_web_rag_flow()
