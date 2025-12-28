# Datasource Layer Design

## 1. 职责 (Responsibilities)

本模块是 **数据访问层 (DAL)**，负责所有与数据持久化和底层检索相关的操作。
它的核心职责是屏蔽底层数据库（如 Chroma, SeekDB, DuckDB, PGVector）的接口差异，向业务层提供统一、易用的 CRUD 接口。

**核心原则：**
1.  **Storage Only**: 本层只负责“存”和“取”，不负责“处理”（如切分、清洗、Embedding 生成）。所有传入的数据必须已经是处理好的 `Document` 对象。
2.  **Native Power**: 优先利用数据库引擎的原生能力（如数据库自带的 Hybrid Search），减少 Python 层的性能开销。
3.  **Unified Interface**: 无论底层是何种数据库，对上层暴露的接口是一致的。

## 2. 模块结构

```text
datasource/
├── vdb/                 # 向量数据库适配器 (Vector Database)
│   ├── base.py          # [Abstract] 向量库基类
│   ├── chroma.py        # ChromaDB 实现 (Local/Server)
│   ├── seekdb.py        # SeekDB 实现 (AI-Native DB)
│   └── pgvector.py      # (Planned) Postgres pgvector 实现
├── keyword/             # 关键词搜索适配器 (Keyword Search)
│   ├── base.py          # [Abstract] 关键词库基类
│   └── duckdb.py        # 基于 DuckDB FTS 的实现
└── service.py           # [Facade] RetrievalService 服务入口
```

## 3. 核心接口设计

### 3.1 BaseVector (Abstract Class)

所有向量数据库必须实现的接口。

```python
class BaseVector(ABC):
    def __init__(self, dataset: Dataset):
        """初始化时必须绑定一个 Dataset 上下文"""
        pass

    @abstractmethod
    def create(self, texts: list[Document], **kwargs) -> None:
        """
        创建集合(如果不存在)并吸入数据。
        通常是 create_collection + add_texts 的组合原子操作。
        """
        pass

    @abstractmethod
    def add_texts(self, texts: list[Document], **kwargs) -> None:
        """向现有集合追加数据。"""
        pass

    @abstractmethod
    def search(
        self, 
        query: str, 
        query_vector: list[float] | None, 
        top_k: int = 4, 
        **kwargs
    ) -> list[Document]:
        """
        执行检索。
        kwargs 参数:
            - search_type: 'similarity' (default) | 'hybrid' | 'mmr'
            - filter: Metadata 过滤条件
        """
        pass
    
    @abstractmethod
    def delete_by_ids(self, ids: list[str]) -> None:
        """根据 ID 删除数据。"""
        pass
```

### 3.2 BaseKeyword (Abstract Class)

针对仅需关键词检索（Economy Mode）的场景，或者作为混合检索的一路召回。

```python
class BaseKeyword(ABC):
    # 接口与 BaseVector 类似，但通常不需要 query_vector 参数
    @abstractmethod
    def search(self, query: str, top_k: int = 4, **kwargs) -> list[Document]:
        pass
```

## 4. 实现策略详解

### 4.1 SeekDB (Vector + Hybrid)
- **特点**: AI-Native 数据库，原生支持 Vector 和 Keyword，且内置了 Hybrid Search (RRF)。
- **策略**: 
    - 当调用 `search(search_type='hybrid')` 时，直接透传调用 SeekDB 的 `hybrid_search` API。
    - 这样可以利用数据库端的优化，避免将大量数据拉回 Python 端做 RRF Fusion。

### 4.2 Chroma (Vector Only)
- **特点**: 流行且易用的向量库，但原生不支持混合检索（或者说支持较弱）。
- **策略**:
    - 主要作为纯向量检索后端。
    - 如果上层强行要求 Hybrid，LangRAG 的 `RetrievalService` 层需要介入（见下文）。

### 4.3 DuckDB (Keyword / Full-Text)
- **特点**: 嵌入式 OLAP 数据库，其 FTS (Full Text Search) 扩展基于 BM25，性能极佳且无需额外服务。
- **策略**:
    - 用作 `BaseKeyword` 的默认实现。
    - 存储时，只存 `content` 和 `metadata`，不存 `vector`（节省空间）。

### 4.4 混合检索 (Hybrid Search) 实现路径

LangRAG 支持两种混合检索实现路径：

1.  **Native (Push-down)**: 
    - 如果底层 DB (如 SeekDB, Weaviate, ElasticSearch) 支持 Hybrid，直接下推查询。
    - **优点**: 性能好，网络传输少。
    
2.  **Application-Side (RRF)**:
    - 如果底层 DB 不支持（如 Chroma），或者 Vector 和 Keyword 存在不同的库里（如 Chroma + DuckDB）。
    - **逻辑**: 
        1. 并行调用 Vector Store 查询 Top N。
        2. 并行调用 Keyword Store 查询 Top N。
        3. 在 `RetrievalService` 中使用 RRF (Reciprocal Rank Fusion) 算法合并结果。

## 5. 异常处理与连接管理

- **连接池**: 对于 Server 模式的 DB (Postgres, SeekDB Server)，实现连接池以复用 TCP 连接。
- **重试机制**: 对于网络波动导致的连接失败，应实现指数退避重试 (Exponential Backoff)。
- **隔离性**: 单个 Dataset 的查询失败不应影响整体服务。

## 6. 未来扩展示例

如果未来要支持 **PGVector**：
1. 在 `datasource/vdb/` 下新建 `pgvector.py`。
2. 继承 `BaseVector`。
3. 使用 `sqlalchemy` 或 `psycopg2` 实现 `search` 方法，构造 SQL 语句 `ORDER BY embedding <=> query_vector`。
4. 在 `Config` 中添加 PG 相关配置。
5. 业务代码无需任何修改。
