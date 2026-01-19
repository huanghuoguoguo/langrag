import langrag
from langrag.agent.tool import Tool
from langrag.datasource.vdb.base import BaseVector
from langrag.entities.dataset import Dataset
from langrag.llm.embedder.base import BaseEmbedder
from langrag.retrieval.rerank.base import BaseReranker
from langrag.retrieval.rewriter.base import BaseRewriter
from langrag.retrieval.router.base import BaseRouter


def create_retrieval_tool(
    vector_stores: list[BaseVector],
    top_k: int = 5,
    embedder: BaseEmbedder | None = None,
    reranker: BaseReranker | None = None,
    rewriter: BaseRewriter | None = None,
    router: BaseRouter | None = None,
    datasets: list[Dataset] | None = None,
    tool_name: str = "search_knowledge_base",
    tool_description: str | None = None,
) -> Tool:
    """
    Create a retrieval tool wrapping langrag.search.

    Args:
        vector_stores: List of vector stores to search.
        top_k: Number of results to return.
        embedder: Embedding model for query (optional).
        reranker: Reranker model (optional, enables reranking).
        rewriter: Query rewriter (optional, enables query rewriting).
        router: Router for KB selection (optional, requires datasets).
        datasets: Dataset metadata for router (required if router is provided).
        tool_name: Name of the tool (default: "search_knowledge_base").
        tool_description: Custom tool description.

    Returns:
        Tool instance ready for use with run_agent().
    """

    async def search_func(query: str) -> str:
        """Execute search with all configured pipeline components."""
        result = await langrag.search(
            query=query,
            vector_stores=vector_stores,
            embedder=embedder,
            reranker=reranker,
            rewriter=rewriter,
            router=router,
            datasets=datasets,
            top_k=top_k,
        )

        # Format results for the LLM
        output = []

        # Include rewritten query info if available
        if result.rewritten_query:
            output.append(f"[Query rewritten to: {result.rewritten_query}]")

        for i, r in enumerate(result.results):
            source_info = r.source
            # Include KB name if available from datasets
            if datasets:
                kb_name = next((d.name for d in datasets if d.id == r.source or d.collection_name == r.source), None)
                if kb_name:
                    source_info = f"{kb_name}"

            output.append(f"Source {i+1} ({source_info}, score={r.score:.3f}):\n{r.document.page_content}")

        if not output or (len(output) == 1 and output[0].startswith("[Query")):
            return "No relevant information found in the knowledge base."

        return "\n\n".join(output)

    default_description = (
        "Search for information in the knowledge base. "
        "Use this tool when you need to answer questions that require external knowledge or documents. "
        "Provide a clear, specific search query."
    )

    return Tool(
        name=tool_name,
        description=tool_description or default_description,
        func=search_func,
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to retrieve relevant information."
                }
            },
            "required": ["query"]
        }
    )
