"""知识库实体"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from pathlib import Path
from loguru import logger

from ..config.models import StorageRole

if TYPE_CHECKING:
    from ..vector_store import BaseVectorStore
    from ..vector_store.manager import VectorStoreManager
    from ..embedder import BaseEmbedder
    from ..reranker import BaseReranker
    from ..core.search_result import SearchResult
    from ..indexing import IndexingPipeline


class KnowledgeBase:
    """知识库实体
    
    一个知识库包含：
    - 唯一的 KB ID
    - 绑定的数据源列表（引用数据源名称）
    - 绑定的嵌入模型
    - 绑定的重排序模型（可选）
    - 索引和检索能力
    
    示例：
        >>> kb = KnowledgeBase(
        ...     kb_id="kb-001",
        ...     store_manager=store_mgr,
        ...     datasource_refs=[("chroma-1", StorageRole.PRIMARY)],
        ...     embedder=embedder
        ... )
        >>> kb.index_file("document.txt")
        >>> results = kb.retrieve("query text")
    """

    def __init__(
        self,
        kb_id: str,
        store_manager: VectorStoreManager,
        datasource_refs: List[Tuple[str, StorageRole]],
        embedder: BaseEmbedder,
        reranker: Optional[BaseReranker] = None,
        parser_config: Optional[Dict] = None,
        chunker_config: Optional[Dict] = None,
        retrieval_config: Optional[Dict] = None
    ):
        """初始化知识库
        
        Args:
            kb_id: 知识库ID（唯一标识）
            store_manager: 数据源管理器（全局共享）
            datasource_refs: 数据源引用列表 [("ds-name", role), ...]
            embedder: 嵌入模型实例
            reranker: 重排序模型实例（可选）
            parser_config: Parser 配置（可选，默认使用 SimpleTextParser）
            chunker_config: Chunker 配置（可选，默认使用 RecursiveChunker）
            retrieval_config: 检索配置（可选）
        """
        self.kb_id = kb_id
        self.store_manager = store_manager
        self.datasource_refs = datasource_refs
        self.embedder = embedder
        self.reranker = reranker
        
        # 配置
        self.parser_config = parser_config or {"type": "simple_text", "params": {}}
        self.chunker_config = chunker_config or {
            "type": "recursive",
            "params": {"chunk_size": 500, "chunk_overlap": 50}
        }
        self.retrieval_config = retrieval_config or {
            "top_k": 5,
            "fusion_strategy": "rrf"
        }
        
        # 延迟初始化（只在需要时创建）
        self._indexing_pipeline: Optional[IndexingPipeline] = None
        self._retrieval_pipeline = None
        
        # 验证数据源存在
        self._validate_datasources()
        
        logger.info(
            f"Initialized KnowledgeBase '{kb_id}' with "
            f"{len(datasource_refs)} datasource(s): "
            f"{[(name, role.value) for name, role in datasource_refs]}"
        )

    def _validate_datasources(self):
        """验证所有数据源都存在"""
        missing = []
        for ds_name, _ in self.datasource_refs:
            if not self.store_manager.has_datasource(ds_name):
                missing.append(ds_name)
        
        if missing:
            raise ValueError(
                f"Datasources not found: {missing}. "
                f"Please create them first using store_manager.create_datasource()"
            )

    def _get_vector_stores(self) -> List[Tuple[BaseVectorStore, StorageRole]]:
        """从数据源管理器获取实际的存储实例
        
        Returns:
            [(store_instance, role), ...] 列表
        """
        stores = []
        for ds_name, role in self.datasource_refs:
            store = self.store_manager.get_store(ds_name)
            if store is None:
                logger.warning(
                    f"Datasource '{ds_name}' not found in KB '{self.kb_id}', skipping"
                )
                continue
            
            # 检查是否启用
            config = self.store_manager.get_config(ds_name)
            if config and not config.enabled:
                logger.info(f"Datasource '{ds_name}' is disabled, skipping")
                continue
            
            stores.append((store, role))
        
        if not stores:
            raise ValueError(
                f"No valid datasources available for KB '{self.kb_id}'"
            )
        
        return stores

    @property
    def indexing_pipeline(self) -> IndexingPipeline:
        """获取索引管道（延迟创建）
        
        Returns:
            IndexingPipeline 实例
        """
        if self._indexing_pipeline is None:
            from ..config.factory import ComponentFactory
            from ..config.models import ComponentConfig
            from ..indexing import IndexingPipeline
            
            # 创建 parser
            parser = ComponentFactory.create_parser(
                ComponentConfig(**self.parser_config)
            )
            
            # 创建 chunker
            chunker = ComponentFactory.create_chunker(
                ComponentConfig(**self.chunker_config)
            )
            
            # 获取数据源实例
            stores = self._get_vector_stores()
            
            # 创建索引管道
            self._indexing_pipeline = IndexingPipeline(
                parser=parser,
                chunker=chunker,
                embedder=self.embedder,
                vector_stores=stores
            )
            
            logger.info(f"Created indexing pipeline for KB '{self.kb_id}'")
        
        return self._indexing_pipeline

    @property
    def retrieval_pipeline(self):
        """获取检索管道（延迟创建）
        
        Returns:
            AdaptiveRetrievalPipeline 实例
        """
        if self._retrieval_pipeline is None:
            from ..retrieval import Retriever
            from ..engine import AdaptiveRetrievalPipeline
            
            stores = self._get_vector_stores()
            
            # 根据数据源数量选择检索模式
            if len(stores) == 1:
                # 单数据源：能力自适应
                logger.info(
                    f"KB '{self.kb_id}': Using single-store retrieval "
                    f"({stores[0][0].__class__.__name__})"
                )
                retriever = Retriever.from_single_store(
                    embedder=self.embedder,
                    vector_store=stores[0][0],
                    storage_role=stores[0][1]
                )
            else:
                # 多数据源：RRF 融合
                logger.info(
                    f"KB '{self.kb_id}': Using multi-store retrieval "
                    f"with {len(stores)} stores"
                )
                retriever = Retriever.from_multi_stores(
                    embedder=self.embedder,
                    stores_config=stores,
                    fusion_strategy=self.retrieval_config.get('fusion_strategy', 'rrf'),
                    fusion_weights=self.retrieval_config.get('fusion_weights')
                )
            
            # 创建检索管道
            self._retrieval_pipeline = AdaptiveRetrievalPipeline(
                retriever=retriever,
                reranker=self.reranker,
                top_k=self.retrieval_config.get('top_k', 5),
                rerank_top_k=self.retrieval_config.get('rerank_top_k')
            )
            
            logger.info(f"Created retrieval pipeline for KB '{self.kb_id}'")
        
        return self._retrieval_pipeline

    # ==================== 业务接口 ====================

    def index_file(self, file_path: str | Path) -> int:
        """索引单个文件到这个知识库
        
        Args:
            file_path: 文件路径
            
        Returns:
            索引的 chunk 数量
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式不支持
        """
        logger.info(f"KB '{self.kb_id}': Indexing file '{file_path}'")
        num_chunks = self.indexing_pipeline.index_file(file_path)
        logger.info(
            f"KB '{self.kb_id}': Successfully indexed {num_chunks} chunks "
            f"from '{file_path}'"
        )
        return num_chunks

    def index_files(self, file_paths: List[str | Path]) -> int:
        """索引多个文件到这个知识库
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            总共索引的 chunk 数量
        """
        logger.info(f"KB '{self.kb_id}': Indexing {len(file_paths)} files")
        total = self.indexing_pipeline.index_files(file_paths)
        logger.info(f"KB '{self.kb_id}': Indexed {total} chunks in total")
        return total

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """从这个知识库检索相关内容
        
        Args:
            query: 查询文本
            top_k: 返回的结果数（可选，默认使用配置中的值）
            
        Returns:
            SearchResult 列表
        """
        # 临时覆盖 top_k
        if top_k is not None:
            original_top_k = self.retrieval_pipeline.top_k
            self.retrieval_pipeline.top_k = top_k
        
        try:
            logger.info(
                f"KB '{self.kb_id}': Retrieving for query: {query[:50]}..."
            )
            results = self.retrieval_pipeline.retrieve(query)
            logger.info(
                f"KB '{self.kb_id}': Retrieved {len(results)} results"
            )
            return results
        finally:
            # 恢复原值
            if top_k is not None:
                self.retrieval_pipeline.top_k = original_top_k

    async def retrieve_async(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """异步检索（推荐使用）
        
        Args:
            query: 查询文本
            top_k: 返回的结果数
            
        Returns:
            SearchResult 列表
        """
        if top_k is not None:
            original_top_k = self.retrieval_pipeline.top_k
            self.retrieval_pipeline.top_k = top_k
        
        try:
            results = await self.retrieval_pipeline.retrieve_async(query)
            return results
        finally:
            if top_k is not None:
                self.retrieval_pipeline.top_k = original_top_k

    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息
        
        Returns:
            统计信息字典
        """
        stores = self._get_vector_stores()
        
        return {
            "kb_id": self.kb_id,
            "num_datasources": len(self.datasource_refs),
            "datasources": [
                {
                    "name": name,
                    "role": role.value,
                    "type": self.store_manager.get_config(name).type
                    if self.store_manager.get_config(name) else "unknown"
                }
                for name, role in self.datasource_refs
            ],
            "embedder": self.embedder.__class__.__name__,
            "reranker": self.reranker.__class__.__name__ if self.reranker else None,
            "indexing_pipeline_created": self._indexing_pipeline is not None,
            "retrieval_pipeline_created": self._retrieval_pipeline is not None
        }

    def __repr__(self) -> str:
        return (
            f"KnowledgeBase(kb_id='{self.kb_id}', "
            f"datasources={len(self.datasource_refs)})"
        )

