"""Chat API - Simplified using langrag high-level API"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

import langrag
from langrag.agent import create_retrieval_tool, run_agent
from langrag.entities.dataset import Dataset
from langrag.retrieval.rerank.factory import RerankerFactory
from langrag.retrieval.rerank.providers.llm_template import LLMTemplateReranker
from langrag.retrieval.rewriter.llm_rewriter import LLMRewriter
from langrag.retrieval.router.llm_router import LLMRouter
from web.core.database import get_session
from web.core.rag_kernel import RAGKernel
from web.services.kb_service import KBService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    kb_ids: list[str] = []
    query: str
    history: list[Message] = []
    stream: bool = False
    model_name: str | None = None

    # Retrieval config (dynamic, per-request)
    use_rerank: bool = False
    reranker_type: str | None = None
    reranker_model: str | None = None
    use_router: bool = False
    router_model: str | None = None
    use_rewriter: bool = False
    rewriter_model: str | None = None
    top_k: int = 5

    # Agent mode (Agentic RAG)
    use_agent: bool = False
    agent_max_steps: int = 5
    agent_system_prompt: str | None = None


class SourceItem(BaseModel):
    content: str
    score: float
    source: str
    kb_id: str | None = None
    kb_name: str | None = None
    title: str | None = None
    type: str | None = None


class EvaluationRequest(BaseModel):
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None


class MetricResult(BaseModel):
    score: float
    reason: str | None = None


class EvaluationResponse(BaseModel):
    faithfulness: MetricResult
    answer_relevancy: MetricResult
    context_relevancy: MetricResult


def get_rag_kernel():
    """Dependency injection: Get RAG Kernel singleton"""
    from web.app import rag_kernel
    return rag_kernel


@router.post("", response_model=None)
async def chat(
    req: ChatRequest,
    session: Session = Depends(get_session),
    rag_kernel: RAGKernel = Depends(get_rag_kernel)
):
    """
    Execute RAG conversation.

    Uses langrag.search() directly - all complex logic is in langrag core.
    """
    try:
        # 1. Determine target KBs
        target_kb_ids = req.kb_ids
        if not target_kb_ids:
            all_kbs = KBService.list_kbs(session)
            target_kb_ids = [kb.kb_id for kb in all_kbs]
            logger.info(f"[Chat] No KB selected, using all {len(target_kb_ids)} KBs")

        # 2. Collect resources
        stores = []
        datasets = []
        for kb_id in target_kb_ids:
            store = rag_kernel.get_vector_store(kb_id)
            if store:
                stores.append(store)
                kb = KBService.get_kb(session, kb_id)
                datasets.append(Dataset(
                    id=kb_id,
                    name=kb.name if kb else kb_id,
                    description=kb.description if kb else "",
                    collection_name=kb.collection_name if kb else kb_id,
                ))

        # 3. Create optional components (based on request switches)
        rewriter = None
        if req.use_rewriter and req.rewriter_model:
            llm = rag_kernel.model_manager.get_model(req.rewriter_model)
            if llm:
                rewriter = LLMRewriter(llm=llm)

        kb_router = None
        if req.use_router and req.router_model:
            llm = rag_kernel.model_manager.get_model(req.router_model)
            if llm:
                kb_router = LLMRouter(llm=llm)

        reranker = None
        if req.use_rerank and req.reranker_type:
            if req.reranker_type == "llm_template" and req.reranker_model:
                llm = rag_kernel.model_manager.get_model(req.reranker_model)
                if llm:
                    reranker = LLMTemplateReranker(llm_model=llm)
            else:
                try:
                    reranker = RerankerFactory.create(req.reranker_type)
                except Exception as e:
                    logger.warning(f"[Chat] Failed to create reranker: {e}")

        # Resolve LLM for generation
        llm = None
        if req.model_name:
            llm = rag_kernel.model_manager.get_model(req.model_name)
        if not llm:
            chat_model = rag_kernel.stage_config.get("chat")
            if chat_model:
                llm = rag_kernel.model_manager.get_model(chat_model)

        # =====================================================================
        # AGENT MODE: LLM decides when/how to search
        # =====================================================================
        if req.use_agent and llm:
            logger.info("[Chat] Using Agent mode (Agentic RAG)")

            # Create retrieval tool with all pipeline components
            tools = []
            if stores:
                retrieval_tool = create_retrieval_tool(
                    vector_stores=stores,
                    top_k=req.top_k,
                    embedder=rag_kernel.embedder,
                    reranker=reranker,
                    rewriter=rewriter,
                    router=kb_router,
                    datasets=datasets,
                )
                tools.append(retrieval_tool)

            # Build messages for agent
            agent_messages = [
                {"role": m.role, "content": m.content} for m in req.history
            ]
            agent_messages.append({"role": "user", "content": req.query})

            # Run agent with trace
            agent_result = await run_agent(
                llm=llm,
                tools=tools,
                messages=agent_messages,
                max_steps=req.agent_max_steps,
                system_prompt=req.agent_system_prompt,
                return_trace=True,  # Get full execution trace
            )

            # Convert trace to serializable format
            trace_steps = []
            for step in agent_result.steps:
                step_data = {
                    "step": step.step,
                    "thought": step.thought,
                    "elapsed_ms": round(step.elapsed_ms, 2),
                    "tool_calls": [
                        {
                            "name": tc.name,
                            "arguments": tc.arguments,
                            "result": tc.result[:500] + "..." if len(tc.result) > 500 else tc.result,
                            "error": tc.error,
                            "elapsed_ms": round(tc.elapsed_ms, 2),
                        }
                        for tc in step.tool_calls
                    ]
                }
                trace_steps.append(step_data)

            return {
                "query": req.query,
                "rewritten_query": None,
                "sources": [],
                "pipeline": {
                    "agent": {
                        "enabled": True,
                        "max_steps": req.agent_max_steps,
                        "total_steps": agent_result.total_steps,
                        "total_tool_calls": agent_result.total_tool_calls,
                        "elapsed_ms": round(agent_result.total_elapsed_ms, 2),
                        "finished_reason": agent_result.finished_reason,
                    }
                },
                "agent_trace": trace_steps,
                "answer": agent_result.answer,
                "message": f"Agent mode - {agent_result.total_tool_calls} tool calls in {agent_result.total_steps} steps"
            }

        # =====================================================================
        # STATIC RAG MODE: Fixed pipeline (search -> generate)
        # =====================================================================
        # 4. Execute search via langrag API (if stores available)
        result_rewritten_query = None
        result_pipeline = {}
        sources = []

        if stores:
            result = await langrag.search(
                query=req.query,
                vector_stores=stores,
                embedder=rag_kernel.embedder,
                rewriter=rewriter,
                router=kb_router,
                datasets=datasets,
                reranker=reranker,
                top_k=req.top_k,
            )
            result_rewritten_query = result.rewritten_query
            result_pipeline = result.pipeline

            # 5. Format sources
            for r in result.results:
                metadata = r.document.metadata or {}
                kb_id = metadata.get("kb_id", r.source)
                # Find KB name from datasets
                kb_name = next((d.name for d in datasets if d.id == kb_id), kb_id)
                sources.append(SourceItem(
                    content=r.document.page_content,
                    score=r.score,
                    source=metadata.get("source", "unknown"),
                    kb_id=kb_id,
                    kb_name=kb_name,
                    title=metadata.get("source"),
                    type="document"
                ))
        else:
            logger.info("[Chat] No KB stores available, skipping retrieval")

        # 6. Generate answer (LLM already resolved above)
        answer = None
        if llm:
            if sources:
                context_str = "\n\n".join([
                    f"[{i+1}] ({s.kb_name}) {s.content}"
                    for i, s in enumerate(sources)
                ])
                messages = [
                    {"role": "system", "content": f"Answer based on context:\n\n{context_str}"},
                    *[{"role": m.role, "content": m.content} for m in req.history[-4:]],
                    {"role": "user", "content": req.query}
                ]
            else:
                # Pure Chat Mode
                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    *[{"role": m.role, "content": m.content} for m in req.history],
                    {"role": "user", "content": req.query}
                ]

            answer = llm.chat(messages=messages)

        return {
            "query": req.query,
            "rewritten_query": result_rewritten_query,
            "sources": sources,
            "pipeline": result_pipeline,
            "answer": answer,
            "message": f"Retrieved {len(sources)} documents" if sources else "Pure chat mode"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"[Chat] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate", response_model=EvaluationResponse)
def evaluate(
    req: EvaluationRequest,
    session: Session = Depends(get_session),
    rag_kernel: RAGKernel = Depends(get_rag_kernel)
):
    """Evaluate a RAG response using LLM Judge"""
    from web.services.evaluation_service import EvaluationService

    if not rag_kernel.model_manager.list_models():
        raise HTTPException(status_code=400, detail="LLM is not configured")

    try:
        service = EvaluationService(rag_kernel)
        results = service.evaluate_sample(
            question=req.question,
            answer=req.answer,
            contexts=req.contexts,
            ground_truth=req.ground_truth
        )

        return EvaluationResponse(
            faithfulness=MetricResult(**results.get("faithfulness", {"score": 0.0})),
            answer_relevancy=MetricResult(**results.get("answer_relevancy", {"score": 0.0})),
            context_relevancy=MetricResult(**results.get("context_relevancy", {"score": 0.0}))
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
