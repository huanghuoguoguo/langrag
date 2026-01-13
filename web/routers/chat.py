"""Chat API"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

logger = logging.getLogger(__name__)

from web.core.database import get_session
from web.core.rag_kernel import RAGKernel
from web.services.kb_service import KBService

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

    # 检索配置参数
    use_rerank: bool = False
    reranker_type: str | None = None
    reranker_model: str | None = None  # For llm_template reranker
    use_router: bool = False
    router_model: str | None = None
    use_rewriter: bool = False
    rewriter_model: str | None = None


class SourceItem(BaseModel):
    content: str
    score: float
    source: str
    kb_id: str | None = None
    kb_name: str | None = None
    title: str | None = None
    link: str | None = None
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


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


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
    """Execute RAG conversation (Support Streaming)"""
    import logging
    logger = logging.getLogger(__name__)

    try:
        # Convert Pydantic models to dicts for internal use
        history_dicts = [{"role": m.role, "content": m.content} for m in req.history]

        # 确定目标知识库
        target_kb_ids = req.kb_ids

        # 如果没有选择知识库且没有启用 router，默认选择所有知识库
        if not target_kb_ids and not req.use_router:
            all_kbs = KBService.list_kbs(session)
            target_kb_ids = [kb.kb_id for kb in all_kbs]
            logger.info(f"[Chat] No KB selected, using all {len(target_kb_ids)} KBs")

        # 获取 LLM
        llm = None
        if req.model_name:
            llm = rag_kernel.model_manager.get_model(req.model_name)
            logger.info(f"[Chat] Using specified model: {req.model_name}, found: {llm is not None}")

        if not llm:
            # 尝试获取 chat 阶段的默认模型
            chat_model_name = rag_kernel.stage_config.get("chat")
            if chat_model_name:
                llm = rag_kernel.model_manager.get_model(chat_model_name)
                logger.info(f"[Chat] Using chat stage model: {chat_model_name}, found: {llm is not None}")

        logger.info(f"[Chat] User query: {req.query}")
        logger.info(f"[Chat] LLM type: {type(llm).__name__ if llm else 'None'}")

        # Build stores dict from KB IDs and KB names mapping
        kb_stores = {}
        kb_names = {}
        for kb_id in target_kb_ids:
            store = rag_kernel.get_vector_store(kb_id)
            if store:
                kb_stores[kb_id] = store

            # 获取知识库名称
            kb = KBService.get_kb(session, kb_id)
            if kb:
                kb_names[kb_id] = kb.name
            else:
                kb_names[kb_id] = kb_id

        # 执行检索
        sources_value = []
        rewritten_query_value = None
        retrieval_stats_value = {}

        if kb_stores:
            # Create RAG chat service for retrieval
            from web.services.rag_chat_service import RAGChatService
            rag_service = RAGChatService(rag_kernel.model_manager, rag_kernel.embedder)

            result = await rag_service.retrieve(
                kb_ids=target_kb_ids,
                kb_stores=kb_stores,
                query=req.query,
                use_rerank=req.use_rerank,
                reranker_type=req.reranker_type,
                reranker_model=req.reranker_model,
                use_router=req.use_router,
                router_model=req.router_model,
                use_rewriter=req.use_rewriter,
                rewriter_model=req.rewriter_model,
                top_k=5,
                kb_names=kb_names
            )

            sources_value = result.get("sources", [])
            rewritten_query_value = result.get("rewritten_query")
            retrieval_stats_value = result.get("retrieval_stats", {})
            logger.info(f"[Chat] Retrieved {len(sources_value)} sources")
            if rewritten_query_value:
                logger.info(f"[Chat] Query rewritten to: {rewritten_query_value}")

        # 生成答案
        answer = None
        if llm:
            if sources_value:
                # 有检索结果，构建带上下文的 prompt
                context_str = "\n\n".join([
                    f"--- Source {i+1} ({s.get('kb_name', 'unknown')}) ---\n{s['content']}"
                    for i, s in enumerate(sources_value)
                ])

                system_prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context.
If the answer is not in the context, you can use your own knowledge but prefer the context.

Context:
{context_str}
"""
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(history_dicts[-4:])
                messages.append({"role": "user", "content": req.query})
            else:
                # 没有检索结果，直接对话
                messages = history_dicts.copy()
                messages.append({"role": "user", "content": req.query})

            logger.info(f"[Chat] Calling LLM with {len(messages)} messages")
            answer = llm.chat(messages=messages)
            logger.info(f"[Chat] LLM response: {answer[:200] if answer and len(answer) > 200 else answer}")
        else:
            logger.warning("[Chat] No LLM available for answer generation")

        return {
            "query": req.query,
            "rewritten_query": rewritten_query_value,
            "sources": [
                SourceItem(
                    content=item["content"],
                    score=item["score"],
                    source=item["source"],
                    kb_id=item["kb_id"],
                    kb_name=item["kb_name"],
                    title=item.get("source"),
                    type="document"
                ) for item in sources_value
            ],
            "retrieval_stats": retrieval_stats_value,
            "answer": answer,
            "message": f"Retrieved {len(sources_value)} documents" if sources_value else "No documents retrieved"
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

    # Check if LLM is ready - use model_manager instead of llm_client
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
