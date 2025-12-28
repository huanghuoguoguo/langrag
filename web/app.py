"""
LangRAG Web Application
业务层应用，使用 langrag 作为核心 RAG 引擎
"""

import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from web.core.database import init_db
from web.core.rag_kernel import RAGKernel
from web.routers import kb_router, document_router, search_router, config_router

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("web-app")

# Global RAG Kernel instance
rag_kernel = RAGKernel()


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
    finally:
        session.close()
    
    logger.info("Application started successfully")
    
    yield
    
    # Shutdown
    logger.info("Application shutting down...")


# Create FastAPI app
app = FastAPI(
    title="LangRAG Web Application",
    description="业务层应用，使用 langrag 核心库",
    version="1.0.0",
    lifespan=lifespan
)

# Register routers
app.include_router(kb_router)
app.include_router(document_router)
app.include_router(search_router)
app.include_router(config_router)

# Mount static files
app.mount("/", StaticFiles(directory="web/static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
