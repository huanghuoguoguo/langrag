"""Model Configuration API"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

from web.core.database import get_session
from web.core.rag_kernel import RAGKernel
from web.services.embedder_service import EmbedderService

router = APIRouter(prefix="/api/config", tags=["config"])


class EmbedderConfigRequest(BaseModel):
    name: str = "default"
    embedder_type: str = "openai"  # openai or seekdb
    model: str
    base_url: str = ""  # Required for openai type
    api_key: str = ""  # Required for openai type


def get_rag_kernel():
    """Dependency injection: Get RAG Kernel singleton"""
    from web.app import rag_kernel
    return rag_kernel


@router.post("/embedder")
def save_embedder_config(
    req: EmbedderConfigRequest,
    session: Session = Depends(get_session),
    rag_kernel: RAGKernel = Depends(get_rag_kernel)
):
    """Save and activate Embedder configuration"""
    try:
        config = EmbedderService.save_config(
            session,
            rag_kernel,
            name=req.name,
            embedder_type=req.embedder_type,
            model=req.model,
            base_url=req.base_url,
            api_key=req.api_key
        )

        return {
            "status": "ok",
            "message": "Embedder configured successfully",
            "config": {
                "name": config.name,
                "embedder_type": config.embedder_type,
                "model": config.model
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/embedder")
def get_active_embedder(session: Session = Depends(get_session)):
    """Get currently active Embedder configuration"""
    config = EmbedderService.get_active_config(session)
    if not config:
        return {"status": "none", "message": "No active embedder configuration"}

    return {
        "status": "ok",
        "config": {
            "name": config.name,
            "base_url": config.base_url,
            "model": config.model
        }
    }


@router.get("/embedders")
def list_embedders(session: Session = Depends(get_session)):
    """List all Embedder configurations"""
    configs = EmbedderService.list_all(session)
    return {
        "embedders": [
            {
                "name": cfg.name,
                "embedder_type": cfg.embedder_type,
                "model": cfg.model,
                "is_active": cfg.is_active
            }
            for cfg in configs
        ]
    }


# ================= LLM Config =================

class LLMConfigRequest(BaseModel):
    name: str = "default"
    base_url: str = ""
    api_key: str = ""
    model: str
    temperature: float = 0.7
    max_tokens: int = 2048
    model_path: str | None = None

@router.post("/llm")
def save_llm_config(
    req: LLMConfigRequest,
    session: Session = Depends(get_session),
    rag_kernel: RAGKernel = Depends(get_rag_kernel)
):
    """Save and activate LLM configuration"""
    from web.services.llm_service import LLMService
    try:
        config = LLMService.save_config(
            session,
            rag_kernel,
            name=req.name,
            base_url=req.base_url,
            api_key=req.api_key,
            model=req.model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            model_path=req.model_path
        )

        return {
            "status": "ok",
            "message": "LLM configured successfully",
            "config": {
                "name": config.name,
                "model": config.model
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/llms")
def list_llms(session: Session = Depends(get_session)):
    """List all LLM configurations"""
    from web.services.llm_service import LLMService
    configs = LLMService.list_all(session)
    return {
        "llms": [
            {
                "name": cfg.name,
                "base_url": cfg.base_url,
                "model": cfg.model,
                "is_active": cfg.is_active
            }
            for cfg in configs
        ]
    }

class ActivateRequest(BaseModel):
    name: str

@router.post("/llm/activate")
def activate_llm_config(
    req: ActivateRequest,
    session: Session = Depends(get_session),
    rag_kernel: RAGKernel = Depends(get_rag_kernel)
):
    """Activate an LLM configuration"""
    from web.services.llm_service import LLMService
    try:
        config = LLMService.activate_config(session, rag_kernel, req.name)
        if not config:
            raise HTTPException(status_code=404, detail="Configuration not found")
            
        return {
             "status": "ok",
             "message": f"Activated LLM: {config.name}",
             "config": {
                 "name": config.name
             }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/embedder/activate")
def activate_embedder_config(
    req: ActivateRequest,
    session: Session = Depends(get_session),
    rag_kernel: RAGKernel = Depends(get_rag_kernel)
):
    """Activate an Embedder configuration"""
    try:
        config = EmbedderService.activate_config(session, rag_kernel, req.name)
        if not config:
             raise HTTPException(status_code=404, detail="Configuration not found")

        return {
             "status": "ok",
             "message": f"Activated Embedder: {config.name}",
             "config": {
                 "name": config.name 
             }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
