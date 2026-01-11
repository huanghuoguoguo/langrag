"""
LangRAG Web Application
Business layer application using langrag as the core RAG engine
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from web.core.context import rag_kernel
from web.core.database import get_session, init_db
from web.routers import config_router, document_router, kb_router, search_router

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("web-app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Startup
    logger.info("Initializing database...")
    init_db()

    # Inject Web VDB Manager into LangRAG Core
    logger.info("Injecting WebVectorStoreManager into LangRAG Factory...")
    from langrag.datasource.vdb.global_manager import set_vector_manager
    # rag_kernel is globally imported from web.core.context
    set_vector_manager(rag_kernel.vdb_manager)

    # Restore vector stores for existing knowledge bases
    logger.info("Restoring vector stores for existing knowledge bases...")
    from web.services.kb_service import KBService

    session_gen = get_session()
    session = next(session_gen)
    try:
        kbs = KBService.list_kbs(session)
        for kb in kbs:
            rag_kernel.create_vector_store(kb.kb_id, kb.collection_name, kb.vdb_type, name=kb.name)
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
from web.routers.playground import router as playground_router

app.include_router(chat_router)
app.include_router(playground_router)

# Mount static files
app.mount("/", StaticFiles(directory="web/static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
