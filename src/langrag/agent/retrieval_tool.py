from typing import Any, List, Optional
from langrag.agent.tool import Tool
from langrag.datasource.vdb.base import BaseVector
import langrag

def create_retrieval_tool(
    vector_stores: List[BaseVector],
    top_k: int = 5
) -> Tool:
    """
    Create a retrieval tool wrapping langrag.search.
    """
    
    async def search_func(query: str) -> str:
        # We bind the vector_stores and top_k from closure
        # In a real agent, the agent might decide which KB to search if we exposed kb_id
        # For minimal start, we search all provided stores.
        result = await langrag.search(
            query=query,
            vector_stores=vector_stores,
            top_k=top_k
        )
        
        # Format results for the LLM
        output = []
        for i, r in enumerate(result.results):
            output.append(f"Source {i+1} ({r.source}): {r.document.page_content}")
            
        if not output:
            return "No relevant information found."
            
        return "\n\n".join(output)

    return Tool(
        name="search_knowledge_base",
        description="Search for information in the knowledge base. Use this whenever you need to answer questions based on external documents.",
        func=search_func,
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to retrieve information."
                }
            },
            "required": ["query"]
        }
    )
