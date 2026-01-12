"""
LangRAG Web Application
Business layer application using langrag as the core RAG engine
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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
    from web.core.kb_retrieval_config import KBRetrievalConfig

    session_gen = get_session()
    session = next(session_gen)
    try:
        kbs = KBService.list_kbs(session)
        for kb in kbs:
            # 恢复向量存储
            rag_kernel.create_vector_store(kb.kb_id, kb.collection_name, kb.vdb_type, name=kb.name)
            logger.info(f"Restored vector store for KB: {kb.kb_id} (type: {kb.vdb_type})")
            
            # 恢复 KB 级别的检索配置
            retrieval_config = KBRetrievalConfig.from_kb_model(kb)
            rag_kernel.set_kb_retrieval_config(retrieval_config)
            logger.info(f"Restored retrieval config for KB: {kb.kb_id} (reranker={kb.reranker_enabled}, rewriter={kb.rewriter_enabled})")

        # Restore active Embedder
        from web.services.embedder_service import EmbedderService
        active_emb = EmbedderService.get_active_config(session)
        if active_emb:
            rag_kernel.set_embedder(active_emb.embedder_type, active_emb.model, active_emb.base_url, active_emb.api_key)
            logger.info(f"Restored active embedder: {active_emb.name}")

        # Restore all LLMs
        from web.services.llm_service import LLMService

        all_llms = LLMService.list_all(session)
        for llm_cfg in all_llms:
            config = {
                 "temperature": llm_cfg.temperature,
                 "max_tokens": llm_cfg.max_tokens,
            }
            
            if llm_cfg.model_path:
                 config["type"] = "local"
                 config["model_path"] = llm_cfg.model_path
            else:
                 config["type"] = "remote"
                 config["base_url"] = llm_cfg.base_url
                 config["api_key"] = llm_cfg.api_key
                 config["model"] = llm_cfg.model

            try:
                rag_kernel.add_llm(llm_cfg.name, config, set_as_default=llm_cfg.is_active)
                logger.info(f"Restored LLM: {llm_cfg.name} (active={llm_cfg.is_active})")
            except Exception as e:
                logger.error(f"Failed to restore LLM {llm_cfg.name}: {e}")

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


# Templates
templates = Jinja2Templates(directory="web/templates")

# Mount static files
app.mount("/static", StaticFiles(directory="web/static"), name="static")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
