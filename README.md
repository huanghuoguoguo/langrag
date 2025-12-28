# LangRAG

LangRAG 是一个模块化的检索增强生成（RAG）框架，提供可插拔的组件和类型化的配置，支持从简单的 demo 到复杂的生产级应用。

## 功能特性

- **模块化架构**: 清晰分离索引（Indexing）、检索（Retrieval）、数据源（Datasource）和实体（Entities）。
- **类型化配置**: 使用 Pydantic 进行严格的配置验证。
- **灵活的数据流**: 支持自定义的解析器、分块器、嵌入器和向量存储。
- **完整的 RAG 流程**: 包含文档提取、清洗、切分、向量化、存储、检索、重排序等完整链路。
- **易于测试**: 提供 Mock 组件和内存存储，方便进行单元测试和端到端集成测试。

## 项目结构

```
src/langrag/
├── config/            # 配置管理 (Pydantic models)
├── datasource/        # 数据源抽象 (Vector DB, Keyword DB)
├── entities/          # 核心领域实体 (Document, Dataset, SearchResult)
├── index_processor/   # 索引处理管道 (Extractor, Splitter, Cleaner)
├── llm/               # LLM 适配层 (Embedder, Chat Model)
├── retrieval/         # 检索工作流 (Workflow, Router, Reranker)
└── utils/             # 通用工具
```

## 安装

```bash
# 克隆项目
git clone <repository-url>
cd langrag

# 安装依赖（使用 uv）
uv sync --dev
```

## 快速开始

### 1. 启动 Web 界面 (推荐)

LangRAG 提供了一个功能完整的 Web 管理界面，支持可视化的知识库管理、文档上传、配置和检索测试。

**功能亮点：**
- **可视化管理**: 直观的知识库列表和详情页。
- **多向量库支持**: 支持 ChromaDB, DuckDB (Persistent), SeekDB (Hybrid Search)。
- **灵活的 Embedder**: 支持 OpenAI 兼容接口及 SeekDB 本地内置模型。
- **混合检索**: 配合 SeekDB 实现向量+全文的混合检索。
- **持久化存储**: 所有业务数据和向量数据均持久化存储于 `web/data` 目录。

**启动方式：**

```bash
# 方式一：使用启动脚本
chmod +x web/start.sh
./web/start.sh

# 方式二：直接运行模块
uv run python -m web.app
```

启动后访问: [http://localhost:8000](http://localhost:8000)

### 2. 运行脚本 Demo (CLI)

LangRAG 提供了一个开箱即用的 `main.py` 演示脚本，它会使用内存向量存储演示完整的索引和检索流程。

```bash
uv run python main.py
```

### 2. 代码示例

以下是一个简化的使用示例：

```python
from langrag.entities.dataset import Dataset
from langrag.index_processor.extractor import SimpleTextParser
from langrag.index_processor.splitter import RecursiveCharacterChunker
from langrag.retrieval.workflow import RetrievalWorkflow
from tests.utils.in_memory_vector_store import InMemoryVectorStore

# 1. 准备数据源
dataset = Dataset(name="demo", collection_name="demo_collection")
store = InMemoryVectorStore(dataset)

# 2. 索引文档 (ETL)
parser = SimpleTextParser()
chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=20)

docs = parser.parse("README.md")
chunks = chunker.split(docs)

# 模拟 Embedding (实际使用中会自动调用 Embedder)
for chunk in chunks:
    chunk.vector = [0.1] * 384 

store.add_texts(chunks)

# 3. 检索 (Retrieval)
workflow = RetrievalWorkflow()
# 注意：实际使用中 RetrieveService 会自动连接 VectorStore
# 这里演示直接调用 Store 搜索
results = store.search("RAG架构", query_vector=[0.1]*384)

print(f"Top Result: {results[0].page_content}")
```

## 开发与测试

本项目包含完善的测试套件（Unit, Integration, Smoke, E2E）。

```bash
# 运行所有测试
uv run pytest tests/

# 运行冒烟测试（快速验证核心路径）
uv run pytest tests/smoke/

# 运行端到端测试
uv run pytest tests/e2e/

# 查看覆盖率
uv run pytest tests/ --cov=src/langrag --cov-report=term-missing
```

### 添加新组件

1. 在相应模块（如 `index_processor/extractor`）中继承基类（如 `BaseParser`）。
2. 实现核心方法（如 `parse`）。
3. 使用工厂模式注册（可选）。
4. 添加对应的单元测试。
