# Datasource Layer Design

## 1. 职责 (Responsibilities)
本模块是数据访问层 (DAL)，负责所有与持久化存储的交互。
它屏蔽了底层数据库（Weaviate, Chroma, PGVector, Elasticsearch）的差异，向通过提供统一的 CRUD 接口。

**关键原则：**
- **只存不切**：切分逻辑在 `index_processor`，这里只管存。
- **单库融合能力**：优先利用数据库原生的 Hybrid Search 能力，而不是在 Python 层做 RRF（除非数据库不支持）。

## 2. 模块结构

```text
datasource/
├── vdb/                 # 向量数据库适配器
│   ├── base.py         # BaseVector 抽象基类
│   ├── pgvector/       
│   ├── weaviate/
│   └── chroma/
├── keyword/             # 关键词/全文索引适配器 (Economy Mode使用)
│   ├── base.py
│   └── jieba/          # 基于 Jieba 的倒排索引实现
└── service.py          # [入口] RetrievalService
```

## 3. 核心接口设计

### 3.1 BaseVector
```python
class BaseVector(ABC):
    @abstractmethod
    def create(self, texts: list[Document], **kwargs) -> None:
        """批量写入文档"""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int, **kwargs) -> list[Document]:
        """执行检索 (包含 Semantic / Hybrid)"""
        pass
    
    @abstractmethod
    def delete_by_ids(self, ids: list[str]) -> None:
        pass
```

### 3.2 RetrievalService (Facade)
统一的服务入口，处理并发和异常。

```python
class RetrievalService:
    @staticmethod
    def retrieve(
        retrieval_method: str,
        dataset_id: str,
        query: str,
        top_k: int
    ) -> list[Document]:
        """
        根据 retrieval_method 调度 VDB 或 Keyword 模块。
        支持使用 ThreadPoolExecutor 并发执行多路查询。
        """
        pass
```

## 4. Dify 对应关系
- **Dify Path**: `api/core/rag/datasource/`
- **Dify Path**: `api/core/rag/datasource/vdb/vector_factory.py`
- **Dify Path**: `api/core/rag/datasource/retrieval_service.py`

## 5. 演进说明
- **Old (LangRAG)**: `vector_store` 目录混杂了检索逻辑。
- **New**: 
    - 将检索逻辑上移至 `retrieval/`。
    - 将 `vector_store` 重命名为 `datasource/vdb`，明确其作为“数据源”的定位。
    - 引入 `Keyword` 模块支持纯文本检索。
