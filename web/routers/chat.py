"""Chat API - Simplified using langrag high-level API"""

import logging

import langrag
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

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

        # 6. Generate answer (if LLM available)
        answer = None
        llm = None
        if req.model_name:
            llm = rag_kernel.model_manager.get_model(req.model_name)
        if not llm:
            chat_model = rag_kernel.stage_config.get("chat")
            if chat_model:
                llm = rag_kernel.model_manager.get_model(chat_model)

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
