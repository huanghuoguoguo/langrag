# langrag

LangRAG 是一个模块化的检索增强生成（RAG）框架，提供可插拔的组件和类型化的配置。

## 功能特性

- **模块化架构**: 支持可插拔的解析器、分块器、嵌入器、向量存储和重排序器
- **类型化配置**: 使用 Pydantic 进行配置验证
- **完整的 RAG 流程**: 从文档索引到查询检索的完整实现
- **易于测试**: 提供 mock 组件用于测试

## 安装

```bash
# 克隆项目
git clone <repository-url>
cd langrag

# 安装依赖（包括测试依赖）
uv sync --dev
```

## 快速开始

```python
from langrag import RAGEngine, RAGConfig

# 配置 RAG 引擎
config = RAGConfig(
    parser={"type": "simple_text", "params": {"encoding": "utf-8"}},
    chunker={"type": "fixed_size", "params": {"chunk_size": 500, "overlap": 50}},
    embedder={"type": "mock", "params": {"dimension": 384, "seed": 42}},
    vector_store={"type": "in_memory", "params": {}},
    reranker={"type": "noop", "params": {}},
    retrieval_top_k=5,
    rerank_top_k=3
)

# 初始化引擎
engine = RAGEngine(config)

# 索引文档
num_chunks = engine.index("path/to/document.txt")

# 执行检索
results = engine.retrieve("你的查询")
```

## 运行演示

```bash
uv run python main.py
```

## 运行测试

```bash
# 运行所有测试
uv run pytest tests/

# 运行测试并查看覆盖率
uv run pytest tests/ --cov=src/langrag --cov-report=term-missing

# 运行特定测试文件
uv run pytest tests/test_integration.py -v
```

## 项目结构

```
src/langrag/
├── core/              # 核心数据模型
├── parser/            # 文档解析器
├── chunker/           # 文本分块器
├── embedder/          # 嵌入生成器
├── vector_store/      # 向量存储
├── reranker/          # 重排序器
├── llm/               # 大语言模型集成
├── pipeline/          # 处理管道
├── config/            # 配置管理
└── engine.py          # 主引擎

tests/
└── test_integration.py  # 集成测试
```

## 开发

该项目使用 [uv](https://github.com/astral-sh/uv) 进行依赖管理和构建。

### 添加新组件

1. 在相应模块中实现基类
2. 在工厂中注册新组件
3. 更新配置模型
4. 添加相应的测试

### 测试策略

项目包含全面的集成测试，覆盖：
- RAG 引擎初始化
- 文档索引和检索
- 批量处理
- 索引持久化
- 元数据验证
- 检索相关性
