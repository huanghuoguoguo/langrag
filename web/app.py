"""
LangRAG Web Application
业务层应用，使用 langrag 作为核心 RAG 引擎
"""

import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from web.core.database import init_db, engine, get_session
from web.models.database import KnowledgeBase, LLMConfig
from web.core.rag_kernel import RAGKernel
from web.core.context import rag_kernel
from web.routers import kb_router, document_router, search_router, config_router

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("web-app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # Startup
    logger.info("Initializing database...")
    init_db()
    
    # Restore vector stores for existing knowledge bases
    logger.info("Restoring vector stores for existing knowledge bases...")
    from web.core.database import get_session
    from web.services.kb_service import KBService
    
    session_gen = get_session()
    session = next(session_gen)
    try:
        kbs = KBService.list_kbs(session)
        for kb in kbs:
            rag_kernel.create_vector_store(kb.kb_id, kb.collection_name, kb.vdb_type)
            logger.info(f"Restored vector store for KB: {kb.kb_id} (type: {kb.vdb_type})")
            
        # Restore active Embedder
        from web.services.embedder_service import EmbedderService
        active_emb = EmbedderService.get_active_config(session)
        if active_emb:
            rag_kernel.set_embedder(active_emb.embedder_type, active_emb.model, active_emb.base_url, active_emb.api_key)
            logger.info(f"Restored active embedder: {active_emb.name}")

        # Restore active LLM
        from web.services.llm_service import LLMService
        active_llm = LLMService.get_active_config(session)
        if active_llm:
            rag_kernel.set_llm(
                base_url=active_llm.base_url,
                api_key=active_llm.api_key,
                model=active_llm.model,
                temperature=active_llm.temperature,
                max_tokens=active_llm.max_tokens
            )
            logger.info(f"Restored active LLM: {active_llm.name}")
            
    finally:
        session.close()
    
    logger.info("Application started successfully")
    
    yield
    
    # Shutdown
    logger.info("Application shutting down...")


# Initialize FastAPI
app = FastAPI(
    title="LangRAG API",
    description="RAG Knowledge Base Management API",
    version="0.1.0",
    lifespan=lifespan
)

# Register routers
# Register routers
app.include_router(kb_router)
app.include_router(document_router)
app.include_router(search_router)
app.include_router(config_router)
from web.routers.chat import router as chat_router
app.include_router(chat_router)

# Mount static files
app.mount("/", StaticFiles(directory="web/static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
