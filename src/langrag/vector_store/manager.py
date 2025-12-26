"""向量存储管理器 - 管理所有数据源实例"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger

from .base import BaseVectorStore
from .factory import VectorStoreFactory
from ..config.models import VectorStoreConfig, StorageRole


class VectorStoreManager:
    """向量存储数据源管理器
    
    负责管理所有向量存储实例，特性：
    - 数据源是全局共享的（多个 KB 可以引用同一个实例）
    - 支持动态创建/删除数据源
    - 实例复用（同类型+同配置 = 同一个实例）
    - 避免资源浪费
    
    示例：
        >>> manager = VectorStoreManager()
        >>> manager.create_datasource("chroma-1", "chroma", {"persist_directory": "./data"})
        >>> store = manager.get_store("chroma-1")
    """

    def __init__(self):
        """初始化数据源管理器"""
        # 数据源名称到实例的映射
        self._stores: Dict[str, BaseVectorStore] = {}
        
        # 数据源配置
        self._store_configs: Dict[str, VectorStoreConfig] = {}
        
        # 实例缓存：避免重复创建相同配置的实例
        # key: (store_type, frozen_params) -> instance
        self._instances_cache: Dict[Tuple, BaseVectorStore] = {}
        
        logger.info("VectorStoreManager initialized")

    def create_datasource(
        self,
        name: str,
        store_type: str,
        params: Dict[str, Any],
        role: StorageRole = StorageRole.PRIMARY,
        enabled: bool = True
    ) -> str:
        """创建数据源
        
        Args:
            name: 数据源名称（用户自定义，如 "chroma-main"）
            store_type: 存储类型（如 "chroma", "seekdb", "duckdb"）
            params: 存储参数字典
            role: 存储角色
            enabled: 是否启用
            
        Returns:
            数据源名称（用于后续引用）
            
        Raises:
            ValueError: 如果数据源名称已存在或类型不支持
        """
        if name in self._stores:
            raise ValueError(f"Datasource '{name}' already exists")
        
        # 生成缓存键（用于实例复用）
        cache_key = self._make_cache_key(store_type, params)
        
        # 检查是否可以复用已有实例
        if cache_key in self._instances_cache:
            logger.info(
                f"Reusing existing {store_type} instance for datasource '{name}' "
                f"(cached from previous creation)"
            )
            instance = self._instances_cache[cache_key]
        else:
            # 创建新实例
            logger.info(f"Creating new {store_type} instance for datasource '{name}'")
            try:
                instance = VectorStoreFactory.create(store_type, **params)
                self._instances_cache[cache_key] = instance
                logger.info(
                    f"Successfully created {store_type} instance, "
                    f"capabilities: {instance.capabilities}"
                )
            except Exception as e:
                logger.error(f"Failed to create datasource '{name}': {e}")
                raise
        
        # 注册数据源
        self._stores[name] = instance
        self._store_configs[name] = VectorStoreConfig(
            type=store_type,
            params=params,
            role=role,
            enabled=enabled
        )
        
        logger.info(
            f"Registered datasource '{name}' "
            f"(type={store_type}, role={role.value}, enabled={enabled})"
        )
        
        return name

    def get_store(self, name: str) -> Optional[BaseVectorStore]:
        """获取数据源实例
        
        Args:
            name: 数据源名称
            
        Returns:
            VectorStore 实例，不存在返回 None
        """
        return self._stores.get(name)

    def get_config(self, name: str) -> Optional[VectorStoreConfig]:
        """获取数据源配置
        
        Args:
            name: 数据源名称
            
        Returns:
            配置对象，不存在返回 None
        """
        return self._store_configs.get(name)

    def delete_datasource(self, name: str) -> bool:
        """删除数据源
        
        注意：不会删除实际的实例（可能被其他名称引用），
        只是取消这个名称的注册。
        
        Args:
            name: 数据源名称
            
        Returns:
            是否成功删除
        """
        if name not in self._stores:
            logger.warning(f"Datasource '{name}' not found")
            return False
        
        del self._stores[name]
        if name in self._store_configs:
            del self._store_configs[name]
        
        logger.info(f"Deleted datasource '{name}'")
        return True

    def list_datasources(self) -> List[Dict[str, Any]]:
        """列出所有数据源
        
        Returns:
            数据源信息列表
        """
        result = []
        for name, config in self._store_configs.items():
            store = self._stores[name]
            caps = store.capabilities
            
            result.append({
                "name": name,
                "type": config.type,
                "role": config.role.value,
                "enabled": config.enabled,
                "capabilities": {
                    "vector": caps.supports_vector,
                    "fulltext": caps.supports_fulltext,
                    "hybrid": caps.supports_hybrid
                },
                "instance_id": id(store)  # 用于判断是否复用实例
            })
        
        return result

    def has_datasource(self, name: str) -> bool:
        """检查数据源是否存在
        
        Args:
            name: 数据源名称
            
        Returns:
            是否存在
        """
        return name in self._stores

    def _make_cache_key(self, store_type: str, params: Dict[str, Any]) -> Tuple:
        """生成实例缓存键
        
        Args:
            store_type: 存储类型
            params: 参数字典
            
        Returns:
            缓存键元组
        """
        # 将参数字典转换为可哈希的元组
        try:
            frozen_params = frozenset(params.items())
        except TypeError:
            # 如果参数包含不可哈希的值（如列表、字典），转换为字符串
            frozen_params = frozenset(
                (k, str(v)) for k, v in params.items()
            )
        
        return (store_type, frozen_params)

    def get_reused_instances(self) -> Dict[str, List[str]]:
        """获取实例复用情况
        
        Returns:
            {instance_id: [datasource_names]} 映射
        """
        reuse_map: Dict[int, List[str]] = {}
        
        for name, store in self._stores.items():
            instance_id = id(store)
            if instance_id not in reuse_map:
                reuse_map[instance_id] = []
            reuse_map[instance_id].append(name)
        
        # 只返回被多个数据源共享的实例
        return {
            f"instance-{iid}": names
            for iid, names in reuse_map.items()
            if len(names) > 1
        }

    def __repr__(self) -> str:
        return (
            f"VectorStoreManager("
            f"datasources={len(self._stores)}, "
            f"unique_instances={len(self._instances_cache)})"
        )

