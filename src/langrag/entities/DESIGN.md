# Entities Layer Design

## 1. 职责 (Responsibilities)
本模块定义了 LangRAG 系统中流转的核心领域对象（Domain Objects）。
它的核心目标是**解耦**：防止业务逻辑层（Retrieval）直接依赖底层实现（Datasource）的具体数据格式。

## 2. 核心实体设计

### 2.1 Chunk / Document
系统中最基础的数据单元，代表一段被切分后的文本。

```python
class Document(BaseModel):
    """Class for storing a piece of text and associated metadata."""
    page_content: str
    
    # 向量表示 (Optional, 仅在存入时生成)
    vector: list[float] | None = None
    
    # 核心元数据
    metadata: dict[str, Any] = {
        "dataset_id": str,      # 归属的知识库ID
        "document_id": str,     # 归属的源文件ID
        "doc_id": str,          # 自身的分片ID (UUID)
        "doc_hash": str,        # 内容哈希，用于去重
        "position": int,        # 在原文中的位置
    }
```

### 2.2 Dataset (知识库)
代表一个逻辑上的文档集合。

```python
class Dataset(BaseModel):
    id: str
    name: str
    description: str
    
    # 索引策略配置
    indexing_technique: str = "high_quality" # "high_quality" (Vector) or "economy" (Keyword)
    
    # 底层存储配置
    collection_name_bound: str # 绑定的向量库 Collection Name
```

### 2.3 RetrievalContext
检索返回的中间结果。

```python
class RetrievalContext(BaseModel):
    document: Document
    score: float
    retrieval_method: str # "semantic", "keyword", "rerank"
```

## 3. Dify 对应关系
- **Dify Path**: `api/core/rag/models/document.py`
- **Dify Path**: `api/core/rag/models/dataset.py`

## 4. 演进说明
- **Old (LangRAG)**: 之前可能直接传递 dict 或简单的 dataclass。
- **New**: 统一使用 Pydantic 模型，确保类型安全。
