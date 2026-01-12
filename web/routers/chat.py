"""Chat API"""


from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

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
    try:
        # Convert Pydantic models to dicts for internal use
        history_dicts = [{"role": m.role, "content": m.content} for m in req.history]

        target_kb_ids = req.kb_ids
        # Auto-select all KBs if none provided
        if not target_kb_ids:
            all_kbs = KBService.list_kbs(session)
            target_kb_ids = [kb.kb_id for kb in all_kbs]

        result = await rag_kernel.chat(
            kb_ids=target_kb_ids,
            query=req.query,
            history=history_dicts,
            stream=req.stream,
            model_name=req.model_name
        )

        if req.stream:
            from fastapi.responses import StreamingResponse
            return StreamingResponse(result, media_type="text/event-stream")

        return ChatResponse(
            answer=result["answer"],
            sources=[
                SourceItem(**item) for item in result["sources"]
            ]
        )

    except ValueError as e:
        # Typically "LLM not configured"
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate", response_model=EvaluationResponse)
def evaluate(
    req: EvaluationRequest,
    session: Session = Depends(get_session),
    rag_kernel: RAGKernel = Depends(get_rag_kernel)
):
    """Evaluate a RAG response using LLM Judge"""
    from web.services.evaluation_service import EvaluationService
    
    # Check if LLM is ready
    if not rag_kernel.llm_client:
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
