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


# ================= Stage Config =================

@router.get("/stages")
def get_stage_config(rag_kernel: RAGKernel = Depends(get_rag_kernel)):
    """Get current stage configuration"""
    try:
        stage_config = rag_kernel.get_stage_config()
        available_models = rag_kernel.get_available_models()

        return {
            "status": "ok",
            "stages": stage_config,
            "available_models": available_models,
            "available_stages": rag_kernel.get_available_stages()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stage config: {str(e)}")


@router.put("/stages")
def update_stage_config(
    stage_updates: dict[str, str | None],
    rag_kernel: RAGKernel = Depends(get_rag_kernel)
):
    """Update stage configuration"""
    try:
        # Validate stages
        available_stages = rag_kernel.get_available_stages()
        invalid_stages = [stage for stage in stage_updates.keys() if stage not in available_stages]
        if invalid_stages:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stages: {invalid_stages}. Available: {available_stages}"
            )

        # Validate models (allow None for unconfigured)
        available_models = rag_kernel.get_available_models()
        for stage, model_name in stage_updates.items():
            if model_name and model_name not in available_models:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{model_name}' not available for stage '{stage}'. Available: {available_models}"
                )

        # Apply configuration
        for stage, model_name in stage_updates.items():
            rag_kernel.configure_stage(stage, model_name)

        return {
            "status": "ok",
            "message": "Stage configuration updated successfully",
            "updated_stages": list(stage_updates.keys())
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update stage config: {str(e)}")


@router.put("/stages/{stage}")
def update_single_stage(
    stage: str,
    model_name: str | None = None,
    rag_kernel: RAGKernel = Depends(get_rag_kernel)
):
    """Update a single stage configuration"""
    try:
        # Validate stage
        available_stages = rag_kernel.get_available_stages()
        if stage not in available_stages:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stage: {stage}. Available: {available_stages}"
            )

        # Validate model (allow None for unconfigured)
        if model_name:
            available_models = rag_kernel.get_available_models()
            if model_name not in available_models:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{model_name}' not available. Available: {available_models}"
                )

        # Apply configuration
        rag_kernel.configure_stage(stage, model_name)

        return {
            "status": "ok",
            "message": f"Stage '{stage}' configured successfully",
            "stage": stage,
            "model": model_name
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update stage '{stage}': {str(e)}")


@router.get("/stages/info")
def get_stage_info(rag_kernel: RAGKernel = Depends(get_rag_kernel)):
    """Get detailed stage information"""
    try:
        from langrag.llm.stages import LLMStage

        stages_info = {}
        for stage in LLMStage.ALL_STAGES:
            model_name = rag_kernel.model_manager.get_stage_model_name(stage)
            stages_info[stage] = {
                "display_name": LLMStage.STAGE_DESCRIPTIONS.get(stage, stage),
                "description": LLMStage.STAGE_DESCRIPTIONS.get(stage, ""),
                "model_name": model_name,
                "is_configured": model_name is not None,
                "is_required": stage in LLMStage.get_required_stages()
            }

        return {
            "status": "ok",
            "stages": stages_info,
            "available_models": rag_kernel.get_available_models()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stage info: {str(e)}")
