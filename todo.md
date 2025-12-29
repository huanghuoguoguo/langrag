# LangRAG Development Plan

## Phase 1: Core Functionality (核心闭环)
**目标**: 完成 RAG 最小闭环，确保系统在基础检索层面可用且稳定。
**周期**: 1 Week
**Branch**: `feat/phase1-core-loop`

| 任务 ID | 任务名称 | 范围 (Scope) | 详细描述 |
| :--- | :--- | :--- | :--- |
| **1.1** | ✅ **Post-Processing** | `src/langrag/retrieval/` | 实现 `post_processor.py`，包含 `ScoreThreshold` 和 `Deduplication` 逻辑。在 `Workflow` 中集成。 |
| **1.2** | **Config Implementation** | `src/langrag/config/` | 实现 `settings.py`，加载 `.env` 配置，支持 Chroma/DuckDB 路径配置。 |
| **1.3** | **Unit Testing** | `tests/` | 为 `Entities`, `ChromaVector`, `DuckDBVector`, `WORKFLOW` 编写单元测试，覆盖率 > 80%。 |

## Phase 2: Advanced Features (高级特性)
**目标**: 提升检索质量，支持更复杂的索引策略和混合检索。
**周期**: 2 Weeks
**Branch**: `feat/phase2-advanced-rag`

| 任务 ID | 任务名称 | 范围 (Scope) | 详细描述 |
| :--- | :--- | :--- | :--- |
| **2.1** | **Parent-Child Indexing** | `src/langrag/index_processor/` | 实现 `ParentChildProcessor`。需要引入 DocStore (Key-Value) 存储父文档。 |
| **2.2** | **Rerank Integration** | `src/langrag/retrieval/` | 实现 `Rerank` Adapter，对接 BGE/Cohere 模型。在 `Workflow` 中正式启用 Rerank 步骤。 |
| **2.3** | **Datasource Factory** | `src/langrag/datasource/` | 实现 `DatasourceFactory`，支持根据 Dataset 配置动态选择 VDB 后端。 |

## Phase 3: Ecosystem & Observability (生态与底座)
**目标**: 增强系统扩展性，支持更多文件格式，增加可观测性。
**周期**: 2 Weeks
**Branch**: `feat/phase3-ecosystem`

| 任务 ID | 任务名称 | 范围 (Scope) | 详细描述 |
| :--- | :--- | :--- | :--- |
| **3.1** | **ETL Extractors** | `src/langrag/index_processor/` | 移植 Dify 的 PDF, Word, Markdown 解析器。优化提取质量。 |
| **3.2** | **Observability** | `Global` | 引入 Callback 机制，在关键节点（Search, LLM Call）埋点，输出 Token 消耗和耗时日志。 |
| **3.3** | **QA Indexing** | `src/langrag/index_processor/` | (Optional) 实现 QA 对生成与索引策略。 |

## Phase 4: Agentic RAG (未来探索)
**目标**: 引入 LLM 在检索流程中的决策能力。
**周期**: TBD
**Branch**: `feat/agentic-rag`

| 任务 ID | 任务名称 | 范围 (Scope) | 详细描述 |
| :--- | :--- | :--- | :--- |
| **4.1** | **LLM Router** | `src/langrag/retrieval/router/` | 实现基于 Function Calling 的 Intent Router。 |
| **4.2** | **Query Rewrite** | `src/langrag/retrieval/` | 在 Workflow 入口增加 Query Rewrite 模块。 |

---
**执行建议**:
1. 每次只在一个 Feature Branch 上工作，完成后 Merge 到 `main`。
2. 每个任务完成后，必须通过对应的 Unit Test。
3. 随时保持设计文档 (`DESIGN.md`) 与代码同步。