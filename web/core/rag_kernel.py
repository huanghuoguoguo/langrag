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
    QAIndexProcessor,
    ParagraphIndexProcessor,
    ParentChildIndexProcessor,
    RecursiveCharacterChunker,
    BaseVector,
    BaseEmbedder,
    EmbedderFactory
)
from langrag.datasource.kv.sqlite import SQLiteKV
from langrag.retrieval.rerank.base import BaseReranker
from langrag.retrieval.rerank.factory import RerankerFactory
from langrag.entities.search_result import SearchResult

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
        self.kb_names: dict[str, str] = {}  # kb_id -> human-readable name
        
        # LLM Client (OpenAI API Compatible)
        self.llm_client = None
        self.llm_config = {}
        self.llm_adapter = None # Store adapter for internal components
        
        # Reranker
        self.reranker: Optional[BaseReranker] = None
        
        # Use SQLite for persistent KV storage
        from web.config import DATA_DIR
        kv_path = DATA_DIR / "kv_store.sqlite"
        self.kv_store = SQLiteKV(db_path=str(kv_path))
        logger.info(f"[RAGKernel] KV Store initialized at {kv_path}")
        
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
            self.llm_adapter = web_llm_adapter # Store for use in processors
            
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

    def set_reranker(self, provider: str, **kwargs):
        """Set Rerank model."""
        logger.info(f"[RAGKernel] Setting reranker: {provider}")
        try:
            self.reranker = RerankerFactory.create(provider, **kwargs)
        except Exception as e:
            logger.error(f"Failed to set reranker: {e}")
            raise e
    
    def create_vector_store(self, kb_id: str, collection_name: str, vdb_type: str, name: Optional[str] = None) -> BaseVector:
        """为知识库创建向量存储"""
        logger.info(f"[RAGKernel] Creating vector store: kb_id={kb_id}, type={vdb_type}, collection={collection_name}")
        
        # Import config for data directories
        from web.config import CHROMA_DIR, DUCKDB_DIR, SEEKDB_DIR
        
        dataset = Dataset(
            id=kb_id,
            tenant_id="default",
            name=name or kb_id,
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
        elif vdb_type == "web_search":
            from langrag.datasource.vdb.web import WebVector
            store = WebVector(dataset)
            logger.info(f"[RAGKernel] Web Search Vector Store initialized")
        else:
            raise ValueError(f"Unsupported vector database type: {vdb_type}")
        
        self.vector_stores[kb_id] = store
        self.kb_names[kb_id] = name or kb_id
        logger.info(f"[RAGKernel] Vector store created and registered for kb_id={kb_id}")
        logger.info(f"[RAGKernel] Total vector stores: {len(self.vector_stores)}")
        return store
    
    def _process_qa_results(self, results: List[LangRAGDocument]):
        """
        Post-process results for QA Indexing:
        Swap the indexed Question (content) with the original Answer (metadata).
        """
        for doc in results:
             if doc.metadata.get("is_qa"):
                 # Swap content
                 question_text = doc.page_content
                 # Store question in metadata
                 doc.metadata["matched_question"] = question_text
                 
                 # Restore original answer
                 if "answer" in doc.metadata:
                     doc.page_content = doc.metadata["answer"]
                 
                 # Restore original doc ID
                 if "original_doc_id" in doc.metadata:
                     doc.id = doc.metadata["original_doc_id"]
                     doc.metadata["document_id"] = doc.metadata["original_doc_id"]

    def _process_parent_child_results(self, results: List[LangRAGDocument]):
        """
        Post-process results for Parent-Child Indexing:
        Swap the retrieved Child Chunk (content) with the Parent Chunk (from KV Store).
        Deduplicate parents.
        """
        # Collect parent IDs
        parent_ids = []
        for doc in results:
            if "parent_id" in doc.metadata:
                parent_ids.append(doc.metadata["parent_id"])
        
        if not parent_ids:
             return results
             
        # Fetch parents (batch get)
        # Note: In-Memory KV is synchronous, but mget is structured for batching
        parents_content = self.kv_store.mget(parent_ids)
        parent_map = dict(zip(parent_ids, parents_content))
        
        # Replace content
        for doc in results:
            pid = doc.metadata.get("parent_id")
            if pid and pid in parent_map and parent_map[pid]:
                # Swap content
                doc.page_content = parent_map[pid]
                # Update ID to parent ID to facilitate proper deduplication downstream
                # (If we don't, we might have multiple chunks showing same parent text)
                doc.id = pid
                doc.metadata["is_parent"] = True
                
    def get_vector_store(self, kb_id: str) -> Optional[BaseVector]:
        """获取知识库的向量存储"""
        return self.vector_stores.get(kb_id)
    
    def process_document(
        self, 
        file_path: Path, 
        kb_id: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        indexing_technique: str = "high_quality"
    ) -> int:
        """
        处理文档：解析 -> 分块 -> 嵌入 -> 存储
        返回生成的 chunk 数量
        """
        logger.info(f"[RAGKernel] process_document called with kb_id={kb_id}, file={file_path}, technique={indexing_technique}")
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

        # 2. Processing Strategy
        if indexing_technique == 'qa':
             logger.info(f"[RAGKernel] Using QA Indexing Technique")
             if not self.llm_adapter:
                  raise ValueError("LLM not configured, cannot use QA indexing")
             if not self.embedder:
                  raise ValueError("Embedder not configured, cannot use QA indexing")
             
             chunker = RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
             
             processor = QAIndexProcessor(
                  vector_store=store,
                  llm=self.llm_adapter,
                  embedder=self.embedder,
                  splitter=chunker
             )
             
             processor.process(store.dataset, raw_docs)
             
             # Estimate chunk count (or exact if we tracked it)
             # QA processor produces roughly 1 Q per chunk
             # We return a placeholder count since we don't get exact from processor yet
             return len(raw_docs) * 2 # Crude estimate
        
        elif indexing_technique == 'parent_child':
             logger.info(f"[RAGKernel] Using Parent-Child Indexing Technique")
             if not self.embedder:
                 raise ValueError("Embedder not configured, cannot use Parent-Child indexing")
             
             # Strategies:
             # Parent: Large chunks (e.g. chunk_size * 2 or fixed ~1000)
             # Child: Small chunks (e.g. chunk_size / 2 or fixed ~200)
             # We will use simple configuration derived from args
             
             parent_chunk_size = chunk_size * 2
             child_chunk_size = chunk_size // 2  # Smaller for retrieval
             if child_chunk_size < 100: child_chunk_size = 200
             
             parent_splitter = RecursiveCharacterChunker(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
             child_splitter = RecursiveCharacterChunker(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap // 2)
             
             processor = ParentChildIndexProcessor(
                 vector_store=store,
                 kv_store=self.kv_store,
                 embedder=self.embedder,
                 parent_splitter=parent_splitter,
                 child_splitter=child_splitter
             )
             
             processor.process(store.dataset, raw_docs)
             return len(raw_docs) * 4 # Crude estimate

        else:
            # Default / High Quality (Original Logic)
            
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
            
            # Determine retrieval top_k (expand if reranker is active)
            k = top_k * 5 if self.reranker else top_k

            if store.__class__.__name__ == 'SeekDBVector' and query_vector:
                search_type = "hybrid"
                results = store.search(query, query_vector=query_vector, top_k=k, search_type='hybrid')
            else:
                results = store.search(query, query_vector=query_vector, top_k=k)
            
            self._process_qa_results(results)
            self._process_parent_child_results(results)

            # Rerank
            if self.reranker and results:
                # Convert to SearchResult objects for reranker
                search_results = [
                    SearchResult(chunk=doc, score=doc.metadata.get('score', 0.0))
                    for doc in results
                ]
                
                try:
                    reranked = self.reranker.rerank(query, search_results, top_k=top_k)
                    
                    # Convert back to Document list & update scores
                    final_results = []
                    for res in reranked:
                        doc = res.chunk
                        doc.metadata['score'] = res.score
                        final_results.append(doc)
                    results = final_results
                    search_type += "+rerank"
                except Exception as e:
                    logger.error(f"[RAGKernel] Rerank failed: {e}")
                    results = results[:top_k] # Fallback to top_k original
            
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
                     
             # Determine retrieval top_k (expand if reranker is active)
             k = top_k * 5 if self.reranker else top_k

             # Execute search on store directly
             # TODO: Support Hybrid decision logic here similar to workflow
             search_type = "keyword"
             if store.__class__.__name__ == 'SeekDBVector' and query_vec:
                 search_type = "hybrid" 
                 results = store.search(final_query, query_vector=query_vec, top_k=k, search_type='hybrid')
             elif query_vec:
                 search_type = "vector"
                 results = store.search(final_query, query_vector=query_vec, top_k=k)
             else:
                 results = store.search(final_query, top_k=k)

             self._process_qa_results(results)
             self._process_parent_child_results(results)

             # Rerank
             if self.reranker and results:
                # Convert to SearchResult objects for reranker
                search_results = [
                    SearchResult(chunk=doc, score=doc.metadata.get('score', 0.0))
                    for doc in results
                ]
                
                try:
                    reranked = self.reranker.rerank(final_query, search_results, top_k=top_k)
                    
                    # Convert back to Document list & update scores
                    final_results = []
                    for res in reranked:
                        doc = res.chunk
                        doc.metadata['score'] = res.score
                        final_results.append(doc)
                    results = final_results
                    search_type += "+rerank"
                except Exception as e:
                    logger.error(f"[RAGKernel] Rerank failed (Agentic): {e}")
                    results = results[:top_k]

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
        
        # Determine retrieval top_k (expand if reranker is active)
        k = top_k * 5 if self.reranker else top_k
        
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
                results = store.search(query, query_vector=query_vector, top_k=k, search_type=current_search_type)
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
        
        self._process_qa_results(all_results)
        self._process_parent_child_results(all_results)

        # Rerank
        if self.reranker and all_results:
            # Convert to SearchResult objects for reranker
            search_results = [
                SearchResult(chunk=doc, score=doc.metadata.get('score', 0.0))
                for doc in all_results
            ]
            
            try:
                reranked = self.reranker.rerank(query, search_results, top_k=top_k)
                
                # Convert back to Document list & update scores
                final_results = []
                for res in reranked:
                    doc = res.chunk
                    doc.metadata['score'] = res.score
                    final_results.append(doc)
                return final_results, primary_search_type + "+rerank"
            except Exception as e:
                logger.error(f"[RAGKernel] Rerank failed in multi-search: {e}")
                
        return all_results[:top_k], primary_search_type

    async def chat(
        self,
        kb_ids: List[str],
        query: str,
        history: List[dict] = None,
        stream: bool = False
    ) -> dict | object:
        """
        RAG 对话 (支持多知识库)
        If stream=True, returns an AsyncGenerator yielding chunks (str or dict).
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
                    candidate_datasets = []
                    for kid in kb_ids:
                        store = self.get_vector_store(kid)
                        description = f"Knowledge base: {kid}"
                        
                        # Dynamically generate better description based on store type
                        if store:
                            store_type = store.__class__.__name__
                            if store_type == "WebVector":
                                description = "Internet Search Engine. Use this for questions about current events, latest news, public information, or release dates (e.g. Python version, iPhone specs)."
                            elif "SeekDB" in store_type or "Chroma" in store_type:
                                description = f"Local Private Knowledge Base ({kid}). Use this for internal documents, company policy, or specific domain knowledge."
                        
                        candidate_datasets.append(
                            Dataset(name=kid, collection_name=kid, description=description)
                        )
                    
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
        
        # 3. Generate
        sources_list = [
            {
                "content": doc.page_content,
                "score": doc.metadata.get('score', 0),
                "source": doc.metadata.get('source', 'unknown'),
                "kb_id": doc.metadata.get('kb_id', 'unknown'),
                "kb_name": self.kb_names.get(doc.metadata.get('kb_id'), 'Unknown'),
                "title": doc.metadata.get('title'),
                "link": doc.metadata.get('link'),
                "type": doc.metadata.get('type')
            }
            for doc in results
        ]

        logger.info(f"[RAGKernel] Sending request to LLM: model={self.llm_config['model']}, stream={stream}")

        if stream:
             # Streaming Logic
             async def stream_generator():
                 # Yield sources first as JSON line
                 import json
                 yield json.dumps({"type": "sources", "data": sources_list}) + "\n"
                 
                 try:
                    stream_resp = await self.llm_client.chat.completions.create(
                        model=self.llm_config["model"],
                        messages=messages,
                        temperature=self.llm_config["temperature"],
                        max_tokens=self.llm_config["max_tokens"],
                        stream=True
                    )
                    
                    async for chunk in stream_resp:
                        content = chunk.choices[0].delta.content
                        if content:
                            yield json.dumps({"type": "content", "data": content}) + "\n"
                            
                 except Exception as e:
                     logger.error(f"[RAGKernel] LLM streaming generation failed: {e}")
                     yield json.dumps({"type": "error", "data": str(e)}) + "\n"

             return stream_generator()

        else:
            # Sync Logic
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
                    "sources": sources_list
                }
            except Exception as e:
                logger.error(f"[RAGKernel] LLM generation failed: {e}")
                raise e
