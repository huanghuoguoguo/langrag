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
        
        # Register custom embedder type
        try:
            EmbedderFactory.register("web_openai", WebOpenAIEmbedder)
        except Exception:
            pass
    
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
            store = SeekDBVector(dataset, persist_directory=str(SEEKDB_DIR))
            logger.info(f"[RAGKernel] SeekDB data directory: {SEEKDB_DIR}")
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
        store = self.get_vector_store(kb_id)
        if not store:
            raise ValueError(f"Vector store not found for kb_id: {kb_id}")
        
        query_vector = None
        search_type = "keyword"
        
        # Generate query embedding if available
        if self.embedder:
            try:
                vectors = self.embedder.embed([query])
                if vectors and len(vectors) > 0:
                    query_vector = vectors[0]
                    search_type = "vector"
            except Exception as e:
                logger.error(f"Query embedding failed: {e}")
        
        results = store.search(query, query_vector=query_vector, top_k=top_k)
        return results, search_type
