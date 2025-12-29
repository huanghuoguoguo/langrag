"""RAG Kernel wrapper - interfaces with the core langrag library."""

import sys
from pathlib import Path
import logging
import httpx
from typing import List, Optional

# Add src to python path
sys.path.append(str(Path(__file__).parents[2] / "src"))

from langrag import (
    Dataset,
    Document as LangRAGDocument,
    SimpleTextParser,
    RecursiveCharacterChunker,
    BaseVector,
    BaseEmbedder,
    EmbedderFactory
)

logger = logging.getLogger(__name__)


class WebOpenAIEmbedder(BaseEmbedder):
    """
    OpenAI 兼容的 Embedder 实现
    通过外部注入的方式使用，不由 RAG 内核管理
    """
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.client = httpx.Client(timeout=30.0)
        self._dimension = 1536  # Default
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": texts,
            "model": self.model
        }
        
        try:
            resp = self.client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            
            results = data.get("data", [])
            results.sort(key=lambda x: x.get("index", 0))
            
            vector_list = [item["embedding"] for item in results]
            if vector_list:
                self._dimension = len(vector_list[0])
                
            return vector_list
            
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise e  # Don't use fallback, let it fail properly

    @property
    def dimension(self) -> int:
        return self._dimension


class SeekDBEmbedder(BaseEmbedder):
    """
    SeekDB 内置 Embedder
    使用 pyseekdb 的默认 embedding function (all-MiniLM-L6-v2)
    """
    def __init__(self):
        self._dimension = 384  # all-MiniLM-L6-v2 的维度
        self._embedding_function = None
    
    def _initialize(self):
        """延迟初始化 embedding function"""
        if self._embedding_function is None:
            try:
                import pyseekdb
                self._embedding_function = pyseekdb.get_default_embedding_function()
                logger.info("SeekDB default embedding function initialized (all-MiniLM-L6-v2)")
            except ImportError:
                logger.error("pyseekdb not installed. Install with: pip install pyseekdb")
                raise ImportError("pyseekdb is required for SeekDB embedder")
            except Exception as e:
                logger.error(f"Failed to initialize SeekDB embedding function: {e}")
                raise e
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """使用 pyseekdb 的默认 embedding function，分批处理以提高性能"""
        try:
            self._initialize()
            
            # 分批处理，每批 32 个文本
            batch_size = 32
            all_embeddings = []
            
            total_batches = (len(texts) + batch_size - 1) // batch_size
            logger.info(f"Processing {len(texts)} texts in {total_batches} batches (batch_size={batch_size})")
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)...")
                batch_embeddings = self._embedding_function(batch)
                all_embeddings.extend(batch_embeddings)
            
            logger.info(f"Completed embedding {len(texts)} texts")
            return all_embeddings
        except Exception as e:
            logger.error(f"SeekDB embedding failed: {e}")
            raise e
    
    @property
    def dimension(self) -> int:
        return self._dimension


class RAGKernel:
    """
    RAG 内核封装
    负责与 langrag 核心库交互
    使用真实的向量数据库（chroma/duckdb/seekdb）
    """
    def __init__(self):
        self.embedder: Optional[BaseEmbedder] = None
        self.vector_stores: dict[str, BaseVector] = {}
        
        # LLM Client (OpenAI API Compatible)
        self.llm_client = None
        self.llm_config = {}
        
        # Agentic Components
        self.router = None
        self.rewriter = None
        self.workflow = None
        
        # Register custom embedder type
        try:
            EmbedderFactory.register("web_openai", WebOpenAIEmbedder)
        except Exception:
            pass

    def set_llm(self, base_url: str, api_key: str, model: str, temperature: float = 0.7, max_tokens: int = 2048):
        """配置 LLM 客户端"""
        try:
            from openai import AsyncOpenAI
            from web.core.llm_adapter import WebLLMAdapter
            from langrag.retrieval.router.llm_router import LLMRouter
            from langrag.retrieval.rewriter.llm_rewriter import LLMRewriter
            from langrag.retrieval.workflow import RetrievalWorkflow
            
            # 1. Update Web LLM Client (for Chat endpoint)
            self.llm_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            self.llm_config = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            logger.info(f"LLM configured: {model} (base_url={base_url})")
            
            # 2. Inject into LangRAG Core
            # Create Adapter to bridge core -> web client
            # We pass the same client but adapter handles interface matching
            # Note: Adapter currently creates its own sync client internally for core compatibility
            # In a real async migration, we'd pass self.llm_client directly if interfaces matched async
            web_llm_adapter = WebLLMAdapter(self.llm_client, model=model)
            
            # Initialize Agentic Components
            self.router = LLMRouter(llm=web_llm_adapter)
            self.rewriter = LLMRewriter(llm=web_llm_adapter)
            
            # Initialize Workflow with injected components
            self.workflow = RetrievalWorkflow(
                router=self.router,
                rewriter=self.rewriter,
                # vector_store_cls is determined dynamically or via factory inside/outside
                # Workflow largely uses datasets and RetrievalService so we don't strictly need to pass store cls here 
                # unless using the default one.
            )
            logger.info("LangRAG Workflow initialized with Agentic components (Router, Rewriter)")
            
        except ImportError:
            logger.error("openai package not installed. Cannot configure LLM.")
        except Exception as e:
            logger.error(f"Failed to configure LLM: {e}")
            import traceback
            traceback.print_exc()
    
    def set_embedder(self, embedder_type: str, model: str = "", base_url: str = "", api_key: str = ""):
        """设置 Embedder（外部注入）"""
        if embedder_type == "openai":
            if not base_url or not api_key or not model:
                raise ValueError("OpenAI embedder requires base_url, api_key and model")
            self.embedder = WebOpenAIEmbedder(base_url, api_key, model)
            logger.info(f"OpenAI-compatible embedder configured: {model}")
        elif embedder_type == "seekdb":
            self.embedder = SeekDBEmbedder()
            logger.info("SeekDB embedder configured (all-MiniLM-L6-v2)")
        else:
            raise ValueError(f"Unsupported embedder type: {embedder_type}")
    
    def create_vector_store(self, kb_id: str, collection_name: str, vdb_type: str) -> BaseVector:
        """为知识库创建向量存储"""
        logger.info(f"[RAGKernel] Creating vector store: kb_id={kb_id}, type={vdb_type}, collection={collection_name}")
        
        # Import config for data directories
        from web.config import CHROMA_DIR, DUCKDB_DIR, SEEKDB_DIR
        
        dataset = Dataset(
            id=kb_id,
            tenant_id="default",
            name=kb_id,
            description="",
            indexing_technique="high_quality",
            collection_name=collection_name
        )
        
        # Import vector store based on type with configured paths
        if vdb_type == "chroma":
            from langrag.datasource.vdb.chroma import ChromaVector
            store = ChromaVector(dataset, persist_directory=str(CHROMA_DIR))
            logger.info(f"[RAGKernel] ChromaDB data directory: {CHROMA_DIR}")
        elif vdb_type == "duckdb":
            from langrag.datasource.vdb.duckdb import DuckDBVector
            store = DuckDBVector(dataset, persist_directory=str(DUCKDB_DIR))
            logger.info(f"[RAGKernel] DuckDB data directory: {DUCKDB_DIR}")
        elif vdb_type == "seekdb":
            from langrag.datasource.vdb.seekdb import SeekDBVector
            # SeekDB uses db_path instead of persist_directory
            store = SeekDBVector(dataset, mode="embedded", db_path=str(SEEKDB_DIR))
            logger.info(f"[RAGKernel] SeekDB data directory: {SEEKDB_DIR}")
            logger.info(f"[RAGKernel] SeekDB supports hybrid search (vector + full-text)")
        else:
            raise ValueError(f"Unsupported vector database type: {vdb_type}")
        
        self.vector_stores[kb_id] = store
        logger.info(f"[RAGKernel] Vector store created and registered for kb_id={kb_id}")
        logger.info(f"[RAGKernel] Total vector stores: {len(self.vector_stores)}")
        return store
    
    def get_vector_store(self, kb_id: str) -> Optional[BaseVector]:
        """获取知识库的向量存储"""
        return self.vector_stores.get(kb_id)
    
    def process_document(
        self, 
        file_path: Path, 
        kb_id: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> int:
        """
        处理文档：解析 -> 分块 -> 嵌入 -> 存储
        返回生成的 chunk 数量
        """
        logger.info(f"[RAGKernel] process_document called with kb_id={kb_id}, file={file_path}")
        logger.info(f"[RAGKernel] Current vector_stores keys: {list(self.vector_stores.keys())}")
        
        store = self.get_vector_store(kb_id)
        if not store:
            logger.error(f"[RAGKernel] Vector store not found for kb_id: {kb_id}")
            logger.error(f"[RAGKernel] Available stores: {list(self.vector_stores.keys())}")
            raise ValueError(f"Vector store not found for kb_id: {kb_id}")
        
        logger.info(f"[RAGKernel] Vector store found for kb_id: {kb_id}")
        
        # 1. Parse - 根据文件类型选择解析器
        logger.info(f"[RAGKernel] Step 1: Parsing document...")
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.pdf':
            try:
                from langrag.index_processor.extractor.providers.pdf import PdfParser
                parser = PdfParser()
                logger.info(f"[RAGKernel] Using PdfParser for {file_ext} file")
            except ImportError:
                logger.warning("pypdf not installed, falling back to SimpleTextParser (may produce incorrect results)")
                parser = SimpleTextParser()
        elif file_ext in ['.md', '.markdown']:
            try:
                from langrag.index_processor.extractor.providers.markdown import MarkdownParser
                parser = MarkdownParser()
                logger.info(f"[RAGKernel] Using MarkdownParser for {file_ext} file")
            except ImportError:
                parser = SimpleTextParser()
        elif file_ext in ['.html', '.htm']:
            try:
                from langrag.index_processor.extractor.providers.html import HtmlParser
                parser = HtmlParser()
                logger.info(f"[RAGKernel] Using HtmlParser for {file_ext} file")
            except ImportError:
                parser = SimpleTextParser()
        elif file_ext in ['.docx', '.doc']:
            try:
                from langrag.index_processor.extractor.providers.docx import DocxParser
                parser = DocxParser()
                logger.info(f"[RAGKernel] Using DocxParser for {file_ext} file")
            except ImportError:
                parser = SimpleTextParser()
        else:
            parser = SimpleTextParser()
            logger.info(f"[RAGKernel] Using SimpleTextParser for {file_ext} file")
        
        raw_docs = parser.parse(file_path)
        logger.info(f"[RAGKernel] Parsed {len(raw_docs)} raw documents")
        
        # 2. Chunk
        logger.info(f"[RAGKernel] Step 2: Chunking with size={chunk_size}, overlap={chunk_overlap}...")
        chunker = RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.split(raw_docs)
        logger.info(f"[RAGKernel] Created {len(chunks)} chunks")
        
        # 3. Embed (if configured)
        if self.embedder:
            text_list = [c.page_content for c in chunks]
            try:
                logger.info(f"[RAGKernel] Step 3: Embedding {len(text_list)} chunks with {self.embedder.__class__.__name__}...")
                vectors = self.embedder.embed(text_list)
                logger.info(f"[RAGKernel] Received {len(vectors)} embedding vectors")
                for doc, vec in zip(chunks, vectors):
                    doc.vector = vec  # Store in vector field, not metadata
                logger.info(f"[RAGKernel] Embeddings attached to chunks")
            except Exception as e:
                logger.error(f"[RAGKernel] Embedding error: {e}")
                logger.exception("[RAGKernel] Embedding traceback:")
                raise e  # Fail if embedding fails
        else:
            logger.info(f"[RAGKernel] Step 3: No embedder configured, skipping embedding")
        
        # 4. Store
        logger.info(f"[RAGKernel] Step 4: Storing chunks in vector store...")
        store.add_texts(chunks)
        logger.info(f"[RAGKernel] Successfully stored {len(chunks)} chunks")
        
        return len(chunks)
    
    def search(
        self, 
        kb_id: str, 
        query: str, 
        top_k: int = 5
    ) -> tuple[list[LangRAGDocument], str]:
        """
        检索
        返回：(结果列表, 检索类型)
        """
        logger.info(f"[RAGKernel] Search called: kb_id={kb_id}, query='{query[:50]}...', top_k={top_k}")
        
        # 0. Check dependencies
        if not self.workflow:
            logger.warning("[RAGKernel] Workflow not initialized (LLM probably not set), falling back to manual search.")
            # Fallback to manual search below (legacy code structure kept for safety or simplicity)
            # For strict agentic adherence, we should require workflow.
            
            # --- Legacy Manual Path Start ---
            store = self.get_vector_store(kb_id)
            if not store:
                logger.error(f"[RAGKernel] Vector store not found for kb_id: {kb_id}")
                raise ValueError(f"Vector store not found for kb_id: {kb_id}")
            
            query_vector = None
            search_type = "keyword"
            
            # Generate query embedding if available
            if self.embedder:
                try:
                    vectors = self.embedder.embed([query])
                    if vectors:
                        query_vector = vectors[0]
                        search_type = "vector"
                except Exception as e:
                    logger.error(f"Query embedding failed: {e}")
            
            if store.__class__.__name__ == 'SeekDBVector' and query_vector:
                search_type = "hybrid"
                results = store.search(query, query_vector=query_vector, top_k=top_k, search_type='hybrid')
            else:
                results = store.search(query, query_vector=query_vector, top_k=top_k)
            
            return results, search_type
            # --- Legacy Manual Path End ---

        # 1. Use Workflow (Agentic Path)
        try:
             # Construct minimal Dataset object
             # We rely on RetrievalService getting the STORE from factory, but here stores are in self.vector_stores
             # This is a little tricky: Core uses VectorStoreFactory or explicit class.
             # But here we manage stores instance manually in self.vector_stores.
             # To bridge this, we can patch/override how RetrievalService gets the store,
             # OR we register our stores into the Factory if possible, 
             # OR we pass the specific store instance if RetrievalService supported it (it supports cls).
             
             # Easier fix: The Workflow calls RetrievalService.
             # RetrievalService.retrieve calls VectorStoreFactory.get_vector_store(dataset)
             # So we need to ensure Factory returns OUR store instance for this dataset.
             # BUT Factory typically creates news ones.
             
             # HACK/ADAPTER: To use our pre-loaded stores, we might need a custom logical injection.
             # Let's bypass the strict Workflow.retrieve loop and invoke functionality directly? 
             # No, that defeats the purpose of Workflow logic (Router/Rewriter).
             
             # Solution: Re-implement the key steps of Workflow HERE using the components we have.
             # (Since RetrievalService is tightly coupled to Factory/Class instatiation currently).
             
             # 1.1 Rewrite
             final_query = query
             if self.rewriter:
                 final_query = self.rewriter.rewrite(query)
                 logger.info(f" >>> [Agentic RAG] Query Rewrite: '{query}' -> '{final_query}'")
                 print(f" >>> [Agentic RAG] Query Rewrite: '{query}' -> '{final_query}'")

             # 1.2 Route (skip if single KB)
             # Single KB search doesn't really need routing, but let's keep logic consistent
             
             # 1.3 Retrieval (Manual dispatch to our stores)
             store = self.get_vector_store(kb_id)
             if not store:
                raise ValueError(f"Store not found: {kb_id}")
                
             # Embed query for retrieval
             query_vec = None
             if self.embedder:
                 try:
                    vecs = self.embedder.embed([final_query]) # Use rewritten query
                    if vecs: query_vec = vecs[0]
                 except Exception as e:
                     logger.error(f"Embed failed: {e}")
                     
             # Execute search on store directly
             # TODO: Support Hybrid decision logic here similar to workflow
             search_type = "keyword"
             if store.__class__.__name__ == 'SeekDBVector' and query_vec:
                 search_type = "hybrid" 
                 results = store.search(final_query, query_vector=query_vec, top_k=top_k, search_type='hybrid')
             elif query_vec:
                 search_type = "vector"
                 results = store.search(final_query, query_vector=query_vec, top_k=top_k)
             else:
                 results = store.search(final_query, top_k=top_k)

             # 1.4 Post Process (Rerank/Dedupe) - Optional / Future
             # If we had a Reranker injected, we would call it here.
             
             return results, search_type

        except Exception as e:
            logger.error(f"[RAGKernel] Agentic search failed: {e}")
            raise e

    def multi_search(
        self,
        kb_ids: List[str],
        query: str,
        top_k: int = 5
    ) -> tuple[List[LangRAGDocument], str]:
        """
        多知识库检索
        """
        all_results = []
        primary_search_type = "keyword"
        
        # Determine search type and query vector once
        query_vector = None
        if self.embedder:
            try:
                vectors = self.embedder.embed([query])
                if vectors:
                    query_vector = vectors[0]
                    primary_search_type = "vector"
            except Exception as e:
                logger.error(f"Query embedding failed: {e}")
        
        for kb_id in kb_ids:
            store = self.get_vector_store(kb_id)
            if not store:
                logger.warning(f"Vector store not found for kb_id: {kb_id}, skipping")
                continue
                
            # Determine per-store search type
            current_search_type = primary_search_type
            if store.__class__.__name__ == 'SeekDBVector' and query_vector:
                current_search_type = "hybrid"
                
            try:
                results = store.search(query, query_vector=query_vector, top_k=top_k, search_type=current_search_type)
                # Add KB ID to metadata
                for doc in results:
                    doc.metadata['kb_id'] = kb_id
                all_results.extend(results)
                
                # Update return search type if we used hybrid anywhere
                if current_search_type == "hybrid":
                    primary_search_type = "hybrid"
            except Exception as e:
                logger.error(f"Search failed for KB {kb_id}: {e}")
        
        # Sort combined results by score (descending)
        # Note: Scores might not be perfectly comparable across different vector stores if they use different metrics,
        # but within similar stores it's acceptable.
        all_results.sort(key=lambda x: x.metadata.get('score', 0), reverse=True)
        
        return all_results[:top_k], primary_search_type

    async def chat(
        self,
        kb_ids: List[str],
        query: str,
        history: List[dict] = None
    ) -> dict:
        """
        RAG 对话 (支持多知识库)
        """
        if not self.llm_client:
            raise ValueError("LLM is not configured")

        # 1. Retrieve
        if not kb_ids:
             # Allowed to chat without context if no KB selected
             results = []
             search_type = "none"
        else:
            final_query = query
            if self.rewriter:
                try:
                    final_query = self.rewriter.rewrite(query)
                    logger.info(f" >>> [Agentic RAG] Chat Rewrite: '{query}' -> '{final_query}'")
                    print(f" >>> [Agentic RAG] Chat Rewrite: '{query}' -> '{final_query}'")
                except Exception as e:
                    logger.error(f"Chat rewrite failed: {e}")
            
            # Agentic Router (Filter KBs)
            search_kb_ids = kb_ids
            if self.router and len(kb_ids) > 1:
                try:
                    # Construct Dataset objects for routing
                    # Note: We ideally need descriptions, but using ID as name for now
                    candidate_datasets = [
                        Dataset(name=kid, collection_name=kid, description=f"Knowledge base: {kid}") 
                        for kid in kb_ids
                    ]
                    
                    selected_datasets = self.router.route(final_query, candidate_datasets)
                    selected_ids = [d.name for d in selected_datasets]
                    
                    if selected_ids and len(selected_ids) < len(kb_ids):
                         search_kb_ids = selected_ids
                         logger.info(f" >>> [Agentic RAG] Router filtered KBs: {kb_ids} -> {search_kb_ids}")
                         print(f" >>> [Agentic RAG] Router filtered KBs: {kb_ids} -> {search_kb_ids}")
                    else:
                         logger.info(f" >>> [Agentic RAG] Router selected all KBs (or failed to filter)")
                except Exception as e:
                    logger.error(f"Chat router failed: {e}")

            results, search_type = self.multi_search(search_kb_ids, final_query, top_k=5)
        
        # 2. Construct Prompt
        if results:
            context_text = "\n\n".join([f"--- Source {i+1} (KB: {doc.metadata.get('kb_id','?')}) ---\n{doc.page_content}" for i, doc in enumerate(results)])
            system_prompt = f"""You are a helpful AI assistant capable of answering questions based on the provided context.
Use the context below to answer the user's question clearly and accurately.
If the answer is not in the context, use your own knowledge but prefer the context.

Context:
{context_text}
"""
        else:
            system_prompt = "You are a helpful AI assistant."
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent history (e.g. last 4 messages to avoid token limit)
        if history:
            messages.extend(history[-4:])
            
        messages.append({"role": "user", "content": query})
        
        logger.info(f"[RAGKernel] Sending request to LLM: model={self.llm_config['model']}")
        
        # 3. Generate
        try:
            response = await self.llm_client.chat.completions.create(
                model=self.llm_config["model"],
                messages=messages,
                temperature=self.llm_config["temperature"],
                max_tokens=self.llm_config["max_tokens"]
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "sources": [
                    {
                        "content": doc.page_content,
                        "score": doc.metadata.get('score', 0),
                        "source": doc.metadata.get('source', 'unknown'),
                        "kb_id": doc.metadata.get('kb_id', 'unknown')
                    }
                    for doc in results
                ]
            }
        except Exception as e:
            logger.error(f"[RAGKernel] LLM generation failed: {e}")
            raise e
