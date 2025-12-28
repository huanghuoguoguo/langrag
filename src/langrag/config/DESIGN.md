# Config Layer Design

## 1. 职责 (Responsibilities)

配置层负责管理整个 RAG 系统的运行时参数、环境变量依赖以及组件的默认行为。它是系统的中枢神经，确保所有模块在统一的上下文中运行。

核心目标：
- **集中管理**：将与业务逻辑分离的参数（如各种 Timeout、重试次数、模型默认名）集中管理。
- **强类型校验**：使用 Pydantic 对配置项进行严格的类型和逻辑校验（Validator），防止运行时错误。
- **环境隔离**：支持 `.env` 文件加载，便于在不同部署环境（Dev/Test/Prod）下切换配置。
- **分层设计**：支持从“系统默认” -> “配置文件” -> “环境变量” -> “运行时传参”的优先级覆盖。

## 2. 设计原则

1.  **Immutable (不可变)**: 配置一旦在启动时加载，原则上不应在运行时被随意修改，以保证系统状态的可预测性。
2.  **Explicit (显式)**: 尽量减少隐式默认值，关键配置若缺失应在启动时报错（Fail Fast）。
3.  **Secure (安全)**: 敏感信息（如 API Key）应通过 `SecretStr` 处理，避免在日志中明文打印。

## 3. 详细配置结构 (Configuration Structure)

配置对象将按照功能模块进行嵌套拆分，避免扁平化的大杂烩。

### 3.1 根配置对象 `RAGAppConfig`

```python
class RAGAppConfig(BaseSettings):
    """
    Root configuration object application.
    Loaded from environment variables or .env file.
    Prefix: LANGRAG_
    """
    # 基础设置
    log_level: str = "INFO"
    data_root: Path = Path("./data") # 持久化数据的根目录
    
    # 子模块配置
    vector_store: VectorStoreConfig
    keyword_store: KeywordStoreConfig
    retrieval: RetrievalConfig
    indexing: IndexingConfig
    models: ModelConfig
```

### 3.2 向量存储配置 `VectorStoreConfig`

管理与向量数据库连接及其默认行为相关的参数。

```python
class VectorStoreConfig(BaseModel):
    # 默认使用的向量库类型
    default_provider: str = "chroma"  # chroma, seekdb, pgvector
    
    # Chroma 特定配置
    chroma_path: Path = Path("./chroma_db")
    chroma_impl: str = "duckdb+parquet"
    
    # PGVector 特定配置
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_user: str = "postgres"
    pg_password: SecretStr
    pg_database: str = "langrag"
    
    # SeekDB 特定配置
    seekdb_host: str | None = None
    seekdb_port: int | None = None
    seekdb_path: Path = Path("./seekdb_data")
    
    # 通用连接池配置
    pool_size: int = 10
    timeout: float = 30.0
```

### 3.3 关键词存储配置 `KeywordStoreConfig`

```python
class KeywordStoreConfig(BaseModel):
    enabled: bool = True
    default_provider: str = "duckdb" # duckdb, elasticsearch
    
    duckdb_path: Path = Path("./duckdb_index.db")
```

### 3.4 索引流程配置 `IndexingConfig`

定义文档处理管道的默认参数。

```python
class IndexingConfig(BaseModel):
    # 切片策略默认值
    chunk_size: int = 500
    chunk_overlap: int = 50
    separator: str = "\n"
    
    # 清洗规则开关
    enable_clean_extra_whitespace: bool = True
    enable_remove_stopwords: bool = False
    
    # 并发控制
    max_indexing_threads: int = 4
    batch_size: int = 100
```

### 3.5 检索流程配置 `RetrievalConfig`

定义检索时的默认策略和阈值。

```python
class RetrievalConfig(BaseModel):
    # 默认召回数量
    default_top_k: int = 4
    
    # 分数阈值 (低于此分数的 Chunk 将被丢弃)
    score_threshold: float = 0.5
    
    # 混合检索权重 (Vector vs Keyword, 0-1)
    # 0.7 表示更偏重 Vector
    hybrid_alpha: float = 0.7 
    
    # 重排序配置
    enable_rerank: bool = False
    rerank_top_n: int = 4 # 重排后返回的数量
```

### 3.6 模型配置 `ModelConfig`

管理模型调用的相关参数，虽然具体的模型是外部注入的，但这里定义了一些系统级的默认行为。

```python
class ModelConfig(BaseModel):
    # 默认的模型标识符 (当用户未指定 Dataset 级别模型时使用)
    default_embedding_model: str = "text-embedding-3-small"
    default_chat_model: str = "gpt-3.5-turbo"
    default_rerank_model: str = "cohere-rerank-english-v3.0"
    
    # 接口调用设置
    openai_api_base: str | None = None
    openai_api_key: SecretStr | None = None
    
    # 超时与重试
    request_timeout: int = 60
    max_retries: int = 3
```

## 4. 实现细节 (Implementation Details)

### 加载逻辑
- 使用 `pydantic-settings` 库。
- 自动读取项目根目录下的 `.env` 文件。
- 环境变量前缀 `LANGRAG_`，例如 `LANGRAG_VECTOR_STORE__DEFAULT_PROVIDER=seekdb`。

### 访问方式
在代码中，应通过单例模式访问配置：

```python
from langrag.config import settings

def some_function():
    timeout = settings.models.request_timeout
    # ...
```

## 5. 扩展性设计

- **动态配置覆盖**：虽然 `settings` 是全局单例，但在处理特定 `Dataset` 的请求时，应允许通过 `Dataset` 对象的内部配置（存储在 metadata 或 DB 字段中）来覆盖全局默认值。例如，全局 `chunk_size` 是 500，但某个特定的 Dataset 可以配置为 1000。
- **热重载 (Hot Reload)**：当前设计暂不支持热重载。若需修改 DB 连接等核心配置，需重启服务。

