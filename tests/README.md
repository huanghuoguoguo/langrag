# LangRAG 测试体系

## 📋 测试架构概览

LangRAG 采用**测试金字塔**架构，确保代码质量、可维护性和快速反馈：

```
        /\
       /  \      E2E Tests (端到端测试)
      /----\     ~10% - 完整业务流程
     /------\    Integration Tests (集成测试)
    /--------\   ~20% - 组件间交互
   /----------\  Unit Tests (单元测试)
  /------------\ ~70% - 单个组件逻辑
```

## 🗂️ 目录结构

```
tests/
├── unit/                   # 单元测试 - 快速、隔离
│   ├── core/              # 核心组件
│   ├── vector_store/      # 向量存储
│   ├── embedder/          # 嵌入器
│   ├── chunker/           # 分块器
│   ├── parser/            # 解析器
│   ├── retrieval/         # 检索组件
│   └── utils/             # 工具函数
│
├── integration/            # 集成测试 - 组件协作
│   ├── test_indexing_pipeline.py
│   ├── test_retrieval_pipeline.py
│   ├── test_multi_store.py
│   └── test_knowledge_base.py
│
├── e2e/                    # 端到端测试 - 完整流程
│   ├── test_rag_workflow.py
│   ├── test_multi_store_workflow.py
│   └── test_hybrid_search_workflow.py
│
├── smoke/                  # 冒烟测试 - 快速验证
│   └── test_critical_paths.py
│
├── fixtures/               # 测试数据和夹具
│   ├── __init__.py
│   ├── vector_stores.py
│   ├── documents.py
│   └── mock_data.py
│
├── utils/                  # 测试工具
│   ├── __init__.py
│   ├── assertions.py
│   ├── builders.py
│   └── helpers.py
│
├── conftest.py            # Pytest 全局配置
├── pytest.ini             # Pytest 配置
└── README.md              # 本文件
```

## 🎯 测试分层详解

### 1️⃣ 单元测试 (Unit Tests)

**目标：** 测试单个组件的逻辑正确性

**特点：**
- ⚡ 快速执行（< 100ms/test）
- 🔒 完全隔离（使用 mock/stub）
- 🎯 单一职责（一个测试一个行为）
- 📊 高覆盖率目标（> 80%）

**示例场景：**
```python
# tests/unit/chunker/test_recursive_chunker.py
def test_recursive_chunker_splits_text_correctly():
    chunker = RecursiveChunker(chunk_size=100, overlap=20)
    text = "A" * 250
    chunks = chunker.split([Document(content=text)])
    assert len(chunks) == 3  # 验证分块逻辑
```

**运行：**
```bash
pytest tests/unit/ -v                    # 运行所有单元测试
pytest tests/unit/chunker/ -v            # 运行特定模块
pytest tests/unit/ -k "chunker" -v       # 运行匹配模式的测试
```

---

### 2️⃣ 集成测试 (Integration Tests)

**目标：** 测试组件间的协作和数据流

**特点：**
- 🔗 测试真实集成（真实数据库、文件系统）
- ⏱️ 中等速度（100ms - 1s/test）
- 🎭 部分隔离（可以使用 in-memory 数据库）
- 🔄 验证数据流转

**示例场景：**
```python
# tests/integration/test_indexing_pipeline.py
def test_indexing_pipeline_end_to_end():
    pipeline = IndexingPipeline(parser, chunker, embedder, vector_store)
    num_chunks = pipeline.index_file("test.txt")

    # 验证整个流程：解析 -> 分块 -> 嵌入 -> 存储
    assert num_chunks > 0
    assert vector_store.count() == num_chunks
```

**运行：**
```bash
pytest tests/integration/ -v             # 运行所有集成测试
pytest tests/integration/ -v -s          # 显示日志输出
```

---

### 3️⃣ 端到端测试 (E2E Tests)

**目标：** 测试完整的用户场景和业务流程

**特点：**
- 🌐 完整系统测试
- 🐌 较慢（1s - 10s/test）
- 🎬 模拟真实用户场景
- 🔍 验证业务价值

**示例场景：**
```python
# tests/e2e/test_rag_workflow.py
def test_complete_rag_workflow():
    # 1. 初始化 RAG 引擎
    engine = RAGEngine(config)

    # 2. 索引文档
    engine.index_batch(["doc1.txt", "doc2.txt"])

    # 3. 检索
    results = engine.retrieve("What is RAG?")

    # 4. 生成回答
    answer = engine.query("What is RAG?", use_llm=True)

    # 验证完整流程
    assert len(results) > 0
    assert "retrieval" in answer.lower()
```

**运行：**
```bash
pytest tests/e2e/ -v                     # 运行所有 E2E 测试
pytest tests/e2e/ -v --slow              # 包含慢速测试
```

---

### 4️⃣ 冒烟测试 (Smoke Tests)

**目标：** 快速验证系统核心功能是否正常

**特点：**
- 🚀 超快（< 30s 全部）
- 🔥 核心路径优先
- 🚨 CI/CD 前置检查
- ✅ 快速失败反馈

**示例场景：**
```python
# tests/smoke/test_critical_paths.py
@pytest.mark.smoke
def test_can_import_core_modules():
    from langrag import RAGEngine
    from langrag.vector_store import InMemoryVectorStore
    assert RAGEngine is not None

@pytest.mark.smoke
def test_basic_indexing_works():
    # 最简单的索引流程
    engine = create_minimal_engine()
    assert engine.index("test.txt") > 0
```

**运行：**
```bash
pytest -m smoke -v                       # 只运行冒烟测试
pytest -m smoke --maxfail=1              # 遇到失败立即停止
```

---

## 🏷️ Pytest 标记 (Markers)

使用标记来组织和筛选测试：

```python
@pytest.mark.unit          # 单元测试
@pytest.mark.integration   # 集成测试
@pytest.mark.e2e           # 端到端测试
@pytest.mark.smoke         # 冒烟测试
@pytest.mark.slow          # 慢速测试（> 1s）
@pytest.mark.requires_gpu  # 需要 GPU
@pytest.mark.requires_api  # 需要外部 API
```

**示例：**
```python
@pytest.mark.unit
def test_chunk_size_validation():
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=-1)

@pytest.mark.integration
@pytest.mark.slow
def test_large_document_indexing():
    # 索引大文件
    pass
```

**运行特定标记：**
```bash
pytest -m unit                           # 只运行单元测试
pytest -m "not slow"                     # 排除慢速测试
pytest -m "integration and not slow"     # 快速集成测试
```

---

## 📊 测试覆盖率

**目标覆盖率：**
- 整体代码覆盖率：> 80%
- 核心模块覆盖率：> 90%
- 边界情况覆盖：重点关注

**生成覆盖率报告：**
```bash
# HTML 报告
pytest --cov=src/langrag --cov-report=html tests/

# 终端报告
pytest --cov=src/langrag --cov-report=term-missing tests/

# 只看缺失的行
pytest --cov=src/langrag --cov-report=term-missing:skip-covered tests/
```

**查看报告：**
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

---

## 🚀 运行测试

### 快速开始

```bash
# 安装测试依赖
pip install -e ".[dev]"

# 运行所有测试
pytest

# 详细模式
pytest -v

# 并行运行（使用 pytest-xdist）
pytest -n auto
```

### 常用命令

```bash
# 1. 冒烟测试（CI 入口）
pytest -m smoke -v --maxfail=1

# 2. 单元测试（开发时）
pytest tests/unit/ -v

# 3. 快速反馈（排除慢速）
pytest -m "not slow" -v

# 4. 完整测试套件（发布前）
pytest --cov=src/langrag --cov-report=html

# 5. 失败重试（flaky tests）
pytest --reruns 3 --reruns-delay 1

# 6. 只运行上次失败的测试
pytest --lf

# 7. 调试模式（显示 print 输出）
pytest -s -v

# 8. 匹配模式运行
pytest -k "chroma or duckdb" -v
```

### 性能分析

```bash
# 查找慢速测试
pytest --durations=10

# Profile 测试
pytest --profile

# 内存使用分析
pytest --memprof
```

---

## 🔧 测试工具和夹具

### 常用 Fixtures

```python
# tests/conftest.py 中定义的全局夹具

@pytest.fixture
def temp_dir():
    """临时目录夹具"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_documents():
    """示例文档夹具"""
    return [
        Document(content="RAG is cool", metadata={"source": "doc1"}),
        Document(content="Vector search", metadata={"source": "doc2"}),
    ]

@pytest.fixture
def mock_embedder():
    """Mock 嵌入器"""
    embedder = Mock(spec=BaseEmbedder)
    embedder.embed.return_value = [[0.1] * 384]
    return embedder
```

### 测试工具

```python
# tests/utils/assertions.py
def assert_search_results_valid(results):
    """验证搜索结果格式"""
    assert isinstance(results, list)
    for result in results:
        assert isinstance(result, SearchResult)
        assert 0 <= result.score <= 1
        assert result.chunk is not None

# tests/utils/builders.py
class RAGEngineBuilder:
    """构建器模式创建测试用 RAGEngine"""
    def with_in_memory_store(self):
        ...
    def with_mock_embedder(self):
        ...
    def build(self):
        ...
```

---

## 🔄 CI/CD 集成

### GitHub Actions 示例

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run smoke tests
        run: pytest -m smoke -v --maxfail=1

  unit:
    needs: smoke
    runs-on: ubuntu-latest
    steps:
      - name: Run unit tests
        run: pytest tests/unit/ -v --cov=src/langrag

  integration:
    needs: unit
    runs-on: ubuntu-latest
    steps:
      - name: Run integration tests
        run: pytest tests/integration/ -v

  e2e:
    needs: integration
    runs-on: ubuntu-latest
    steps:
      - name: Run E2E tests
        run: pytest tests/e2e/ -v
```

### Pre-commit Hook

```bash
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: smoke-tests
      name: smoke-tests
      entry: pytest -m smoke -v --maxfail=1
      language: system
      pass_filenames: false
```

---

## 📝 测试最佳实践

### ✅ DO（推荐）

1. **遵循 AAA 模式**
   ```python
   def test_example():
       # Arrange - 准备测试数据
       chunker = RecursiveChunker(chunk_size=100)

       # Act - 执行操作
       chunks = chunker.split(documents)

       # Assert - 验证结果
       assert len(chunks) > 0
   ```

2. **测试名称清晰描述行为**
   ```python
   # Good
   def test_chunker_raises_error_on_negative_chunk_size():
       pass

   # Bad
   def test_chunker():
       pass
   ```

3. **一个测试一个断言（概念）**
   ```python
   # Good
   def test_search_returns_sorted_results():
       results = vector_store.search(query, top_k=5)
       scores = [r.score for r in results]
       assert scores == sorted(scores, reverse=True)
   ```

4. **使用参数化减少重复**
   ```python
   @pytest.mark.parametrize("chunk_size,expected", [
       (100, 3),
       (200, 2),
       (500, 1),
   ])
   def test_chunking_with_different_sizes(chunk_size, expected):
       chunker = RecursiveChunker(chunk_size=chunk_size)
       chunks = chunker.split([Document(content="A" * 300)])
       assert len(chunks) == expected
   ```

### ❌ DON'T（避免）

1. ❌ 测试间有依赖关系
2. ❌ 使用 sleep() 等待异步操作
3. ❌ 硬编码路径和凭证
4. ❌ 测试实现细节而非行为
5. ❌ 忽略 flaky tests

---

## 📈 测试指标

跟踪这些指标以评估测试质量：

- **代码覆盖率**: > 80%
- **测试执行时间**: 单元测试 < 5min，全部 < 15min
- **测试通过率**: > 95%
- **Flaky 测试率**: < 1%
- **Bug 逃逸率**: < 5%

---

## 🆘 故障排查

### 常见问题

**Q: 测试很慢怎么办？**
```bash
# 1. 只运行快速测试
pytest -m "not slow"

# 2. 并行运行
pytest -n auto

# 3. 找出慢速测试
pytest --durations=10
```

**Q: 测试不稳定 (flaky)？**
```bash
# 启用重试
pytest --reruns 3

# 查看详细日志
pytest -vv -s --log-cli-level=DEBUG
```

**Q: 如何调试失败的测试？**
```bash
# 1. 进入 PDB 调试器
pytest --pdb

# 2. 在失败处停止
pytest -x

# 3. 显示局部变量
pytest -l
```

---

## 📚 参考资源

- [Pytest 官方文档](https://docs.pytest.org/)
- [测试金字塔理论](https://martinfowler.com/articles/practical-test-pyramid.html)
- [Python 测试最佳实践](https://docs.python-guide.org/writing/tests/)

---

## 🤝 贡献指南

提交 PR 前请确保：

1. ✅ 所有冒烟测试通过
2. ✅ 新代码有对应的单元测试
3. ✅ 代码覆盖率不降低
4. ✅ 运行 `pytest -m "not slow"` 全部通过
5. ✅ 更新相关文档

---

*最后更新: 2025-12-26*
