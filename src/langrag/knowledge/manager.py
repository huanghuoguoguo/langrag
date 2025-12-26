"""知识库管理器 - LangRAG 的主入口"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from ..config.factory import ComponentFactory
from ..config.models import ComponentConfig, StorageRole
from ..core.search_result import SearchResult
from ..embedder import BaseEmbedder
from ..reranker import BaseReranker
from ..vector_store.manager import VectorStoreManager
from .knowledge_base import KnowledgeBase


class KnowledgeBaseManager:
    """知识库管理器 - LangRAG 内核的主入口

    提供完整的知识库管理功能：
    - 数据源管理：创建、删除、列出数据源
    - 知识库管理：创建、删除、列出知识库
    - 文档索引：将文档索引到知识库
    - 检索查询：从知识库检索相关内容

    特性：
    - 数据源复用：多个知识库可以共享同一个数据源实例
    - 模型复用：多个知识库可以共享同一个 embedder/reranker
    - 动态配置：支持运行时创建和配置

    示例：
        >>> manager = KnowledgeBaseManager()
        >>>
        >>> # 创建数据源
        >>> manager.create_datasource("chroma-1", "chroma", {"persist_directory": "./data"})
        >>>
        >>> # 创建知识库
        >>> manager.create_knowledge_base(
        ...     kb_id="kb-001",
        ...     datasource_names=["chroma-1"],
        ...     embedder_config={"type": "mock", "params": {"dimension": 384}}
        ... )
        >>>
        >>> # 索引文档
        >>> manager.index_document("kb-001", "document.txt")
        >>>
        >>> # 检索
        >>> results = manager.search("kb-001", "查询文本")
    """

    def __init__(self):
        """初始化知识库管理器"""
        # 数据源管理器
        self.store_manager = VectorStoreManager()

        # 知识库映射
        self.knowledge_bases: dict[str, KnowledgeBase] = {}

        # 模型缓存（复用机制）
        self._embedders: dict[str, BaseEmbedder] = {}
        self._rerankers: dict[str, BaseReranker] = {}

        logger.info("KnowledgeBaseManager initialized")

    # ==================== 数据源管理 ====================

    def create_datasource(
        self,
        name: str,
        store_type: str,
        params: dict[str, Any],
        role: str = "primary",
        enabled: bool = True,
    ) -> dict[str, Any]:
        """创建数据源

        Args:
            name: 数据源名称（唯一标识）
            store_type: 存储类型（"chroma", "seekdb", "duckdb", "in_memory"）
            params: 存储参数字典
            role: 存储角色（"primary", "vector_only", "fulltext_only", "backup"）
            enabled: 是否启用

        Returns:
            创建结果字典 {"datasource_id": name, "status": "success"}

        Raises:
            ValueError: 数据源名称已存在或类型不支持
        """
        role_enum = StorageRole(role)

        try:
            ds_id = self.store_manager.create_datasource(
                name=name, store_type=store_type, params=params, role=role_enum, enabled=enabled
            )

            logger.info(f"Created datasource '{name}' (type={store_type}, role={role})")

            return {
                "datasource_id": ds_id,
                "type": store_type,
                "role": role,
                "enabled": enabled,
                "status": "success",
            }
        except Exception as e:
            logger.error(f"Failed to create datasource '{name}': {e}")
            return {"datasource_id": name, "status": "error", "error": str(e)}

    def list_datasources(self) -> list[dict[str, Any]]:
        """列出所有数据源

        Returns:
            数据源信息列表
        """
        return self.store_manager.list_datasources()

    def get_datasource_info(self, name: str) -> dict[str, Any] | None:
        """获取数据源详细信息

        Args:
            name: 数据源名称

        Returns:
            数据源信息字典，不存在返回 None
        """
        store = self.store_manager.get_store(name)
        config = self.store_manager.get_config(name)

        if store is None or config is None:
            return None

        caps = store.capabilities

        return {
            "name": name,
            "type": config.type,
            "role": config.role.value,
            "enabled": config.enabled,
            "capabilities": {
                "vector": caps.supports_vector,
                "fulltext": caps.supports_fulltext,
                "hybrid": caps.supports_hybrid,
            },
            "params": config.params,
        }

    def delete_datasource(self, name: str, force: bool = False) -> dict[str, Any]:
        """删除数据源

        Args:
            name: 数据源名称
            force: 是否强制删除（忽略正在使用的检查）

        Returns:
            删除结果字典
        """
        # 检查是否有知识库在使用
        if not force:
            using_kbs = self._get_datasource_usage(name)
            if using_kbs:
                return {
                    "datasource_id": name,
                    "status": "error",
                    "error": f"Datasource is being used by KBs: {using_kbs}",
                    "hint": "Delete KBs first or use force=True",
                }

        # 删除数据源
        success = self.store_manager.delete_datasource(name)

        if success:
            logger.info(f"Deleted datasource '{name}'")
            return {"datasource_id": name, "status": "success"}
        else:
            return {"datasource_id": name, "status": "error", "error": "Datasource not found"}

    def _get_datasource_usage(self, datasource_name: str) -> list[str]:
        """获取使用某个数据源的知识库列表

        Args:
            datasource_name: 数据源名称

        Returns:
            使用该数据源的知识库ID列表
        """
        using_kbs = []
        for kb_id, kb in self.knowledge_bases.items():
            if any(ds_name == datasource_name for ds_name, _ in kb.datasource_refs):
                using_kbs.append(kb_id)
        return using_kbs

    # ==================== 知识库管理 ====================

    def create_knowledge_base(
        self,
        kb_id: str,
        datasource_names: list[str],
        embedder_config: dict[str, Any],
        datasource_roles: list[str] | None = None,
        reranker_config: dict[str, Any] | None = None,
        parser_config: dict[str, Any] | None = None,
        chunker_config: dict[str, Any] | None = None,
        retrieval_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """创建知识库

        Args:
            kb_id: 知识库ID（唯一标识）
            datasource_names: 要使用的数据源名称列表
            embedder_config: 嵌入模型配置 {"type": "mock", "params": {...}}
            datasource_roles: 数据源角色列表（可选，默认都是 "primary"）
            reranker_config: 重排序模型配置（可选）
            parser_config: Parser 配置（可选）
            chunker_config: Chunker 配置（可选）
            retrieval_config: 检索配置（可选）

        Returns:
            创建结果字典

        Raises:
            ValueError: KB ID已存在或数据源不存在
        """
        # 检查 KB ID 是否已存在
        if kb_id in self.knowledge_bases:
            return {
                "kb_id": kb_id,
                "status": "error",
                "error": f"Knowledge base '{kb_id}' already exists",
            }

        # 验证数据源存在
        missing_datasources = []
        for ds_name in datasource_names:
            if not self.store_manager.has_datasource(ds_name):
                missing_datasources.append(ds_name)

        if missing_datasources:
            return {
                "kb_id": kb_id,
                "status": "error",
                "error": f"Datasources not found: {missing_datasources}",
                "hint": "Create datasources first using create_datasource()",
            }

        # 处理角色
        if datasource_roles is None:
            datasource_roles = ["primary"] * len(datasource_names)

        if len(datasource_roles) != len(datasource_names):
            return {
                "kb_id": kb_id,
                "status": "error",
                "error": "datasource_roles length must match datasource_names",
            }

        try:
            # 构建数据源引用列表
            datasource_refs = [
                (name, StorageRole(role))
                for name, role in zip(datasource_names, datasource_roles, strict=True)
            ]

            # 创建或获取嵌入模型
            embedder = self._get_or_create_embedder(embedder_config)

            # 创建或获取重排序模型
            reranker = None
            if reranker_config:
                reranker = self._get_or_create_reranker(reranker_config)

            # 创建知识库
            kb = KnowledgeBase(
                kb_id=kb_id,
                store_manager=self.store_manager,
                datasource_refs=datasource_refs,
                embedder=embedder,
                reranker=reranker,
                parser_config=parser_config,
                chunker_config=chunker_config,
                retrieval_config=retrieval_config,
            )

            self.knowledge_bases[kb_id] = kb

            logger.info(
                f"Created knowledge base '{kb_id}' with {len(datasource_names)} datasource(s)"
            )

            return {
                "kb_id": kb_id,
                "status": "success",
                "datasources": datasource_names,
                "roles": datasource_roles,
                "embedder": embedder.__class__.__name__,
                "reranker": reranker.__class__.__name__ if reranker else None,
            }

        except Exception as e:
            logger.error(f"Failed to create knowledge base '{kb_id}': {e}")
            return {"kb_id": kb_id, "status": "error", "error": str(e)}

    def delete_knowledge_base(self, kb_id: str) -> dict[str, Any]:
        """删除知识库

        Args:
            kb_id: 知识库ID

        Returns:
            删除结果字典
        """
        if kb_id not in self.knowledge_bases:
            return {"kb_id": kb_id, "status": "error", "error": "Knowledge base not found"}

        del self.knowledge_bases[kb_id]
        logger.info(f"Deleted knowledge base '{kb_id}'")

        return {"kb_id": kb_id, "status": "success"}

    def list_knowledge_bases(self) -> list[dict[str, Any]]:
        """列出所有知识库

        Returns:
            知识库信息列表
        """
        return [
            {
                "kb_id": kb_id,
                "datasources": [name for name, _ in kb.datasource_refs],
                "num_datasources": len(kb.datasource_refs),
                "embedder": kb.embedder.__class__.__name__,
                "reranker": kb.reranker.__class__.__name__ if kb.reranker else None,
            }
            for kb_id, kb in self.knowledge_bases.items()
        ]

    def get_knowledge_base_info(self, kb_id: str) -> dict[str, Any] | None:
        """获取知识库详细信息

        Args:
            kb_id: 知识库ID

        Returns:
            知识库信息字典，不存在返回 None
        """
        kb = self.knowledge_bases.get(kb_id)
        if kb is None:
            return None

        return kb.get_stats()

    # ==================== 业务接口 ====================

    def index_document(self, kb_id: str, file_path: str | Path) -> dict[str, Any]:
        """向知识库索引文档

        Args:
            kb_id: 知识库ID
            file_path: 文档文件路径

        Returns:
            索引结果字典
        """
        kb = self.knowledge_bases.get(kb_id)
        if kb is None:
            return {"kb_id": kb_id, "status": "error", "error": "Knowledge base not found"}

        try:
            num_chunks = kb.index_file(file_path)
            return {
                "kb_id": kb_id,
                "file_path": str(file_path),
                "num_chunks": num_chunks,
                "status": "success",
            }
        except Exception as e:
            logger.error(f"Failed to index document in KB '{kb_id}': {e}")
            return {"kb_id": kb_id, "file_path": str(file_path), "status": "error", "error": str(e)}

    def index_documents(self, kb_id: str, file_paths: list[str | Path]) -> dict[str, Any]:
        """向知识库索引多个文档

        Args:
            kb_id: 知识库ID
            file_paths: 文档文件路径列表

        Returns:
            索引结果字典
        """
        kb = self.knowledge_bases.get(kb_id)
        if kb is None:
            return {"kb_id": kb_id, "status": "error", "error": "Knowledge base not found"}

        try:
            total_chunks = kb.index_files(file_paths)
            return {
                "kb_id": kb_id,
                "num_files": len(file_paths),
                "total_chunks": total_chunks,
                "status": "success",
            }
        except Exception as e:
            logger.error(f"Failed to index documents in KB '{kb_id}': {e}")
            return {
                "kb_id": kb_id,
                "num_files": len(file_paths),
                "status": "error",
                "error": str(e),
            }

    def search(self, kb_id: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """从知识库检索

        Args:
            kb_id: 知识库ID
            query: 查询文本
            top_k: 返回结果数

        Returns:
            检索结果列表（字典格式，便于序列化）
        """
        kb = self.knowledge_bases.get(kb_id)
        if kb is None:
            logger.error(f"Knowledge base '{kb_id}' not found")
            return []

        try:
            results: list[SearchResult] = kb.retrieve(query, top_k=top_k)

            # 转换为字典格式
            return [
                {
                    "chunk_id": r.chunk.id,
                    "content": r.chunk.content,
                    "score": r.score,
                    "metadata": r.chunk.metadata,
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Search failed in KB '{kb_id}': {e}")
            return []

    async def search_async(self, kb_id: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """异步检索（推荐使用）

        Args:
            kb_id: 知识库ID
            query: 查询文本
            top_k: 返回结果数

        Returns:
            检索结果列表
        """
        kb = self.knowledge_bases.get(kb_id)
        if kb is None:
            return []

        try:
            results = await kb.retrieve_async(query, top_k=top_k)
            return [
                {
                    "chunk_id": r.chunk.id,
                    "content": r.chunk.content,
                    "score": r.score,
                    "metadata": r.chunk.metadata,
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Async search failed in KB '{kb_id}': {e}")
            return []

    # ==================== 辅助方法 ====================

    def _get_or_create_embedder(self, config: dict[str, Any]) -> BaseEmbedder:
        """获取或创建嵌入模型（支持复用）

        Args:
            config: 嵌入模型配置

        Returns:
            BaseEmbedder 实例
        """
        cache_key = self._make_model_cache_key(config)

        if cache_key not in self._embedders:
            logger.info(f"Creating new embedder: {config['type']}")
            self._embedders[cache_key] = ComponentFactory.create_embedder(ComponentConfig(**config))
        else:
            logger.info(f"Reusing existing embedder: {config['type']}")

        return self._embedders[cache_key]

    def _get_or_create_reranker(self, config: dict[str, Any]) -> BaseReranker:
        """获取或创建重排序模型（支持复用）

        Args:
            config: 重排序模型配置

        Returns:
            BaseReranker 实例
        """
        cache_key = self._make_model_cache_key(config)

        if cache_key not in self._rerankers:
            logger.info(f"Creating new reranker: {config['type']}")
            self._rerankers[cache_key] = ComponentFactory.create_reranker(ComponentConfig(**config))
        else:
            logger.info(f"Reusing existing reranker: {config['type']}")

        return self._rerankers[cache_key]

    def _make_model_cache_key(self, config: dict[str, Any]) -> str:
        """生成模型缓存键

        Args:
            config: 模型配置

        Returns:
            缓存键字符串
        """
        model_type = config.get("type", "unknown")
        params = config.get("params", {})

        # 简单的键生成策略
        params_str = str(sorted(params.items()))
        return f"{model_type}:{hash(params_str)}"

    def get_stats(self) -> dict[str, Any]:
        """获取管理器统计信息

        Returns:
            统计信息字典
        """
        # 获取实例复用情况
        reused_instances = self.store_manager.get_reused_instances()

        return {
            "num_datasources": len(self.store_manager.list_datasources()),
            "num_knowledge_bases": len(self.knowledge_bases),
            "num_embedders": len(self._embedders),
            "num_rerankers": len(self._rerankers),
            "reused_store_instances": len(reused_instances),
            "reused_stores_detail": reused_instances,
        }

    def __repr__(self) -> str:
        return (
            f"KnowledgeBaseManager("
            f"datasources={len(self.store_manager.list_datasources())}, "
            f"knowledge_bases={len(self.knowledge_bases)})"
        )
