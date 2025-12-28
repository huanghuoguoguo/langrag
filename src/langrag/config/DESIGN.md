# Config Layer Design

## 1. 职责 (Responsibilities)
管理全局配置和环境变量。使用 Pydantic 进行强类型校验。

## 2. 核心配置对象

```python
class RAGConfig(BaseModel):
    # 全局开关
    enable_rerank: bool = False
    default_top_k: int = 4
    
    # 路径配置
    StorageRoot: str
    
    # 模型配置
    EmbeddingModel: str
    RerankModel: str
    
    # Dify 兼容性配置
    # Dify 倾向于将配置放在 Dataset 实例中，而不是全局 Config。
    # 这里我们定义 defaults。
```

## 3. Dify 对应关系
- **Dify Path**: `api/config.py` 或 `api/configs/`
