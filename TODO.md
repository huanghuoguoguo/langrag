# LangRAG Development TODO

> 开发流程：每个任务创建独立分支 → 实现 → 等待用户确认 用户自己提交pr 并合并，然后告知可以进行下一步任务

---

## 优先级说明

- **P0**: 紧急 - 影响核心功能或稳定性
- **P1**: 高优 - 显著提升项目质量和性能
- **P2**: 中等 - 改进开发体验和可维护性
- **P3**: 低优 - 锦上添花

---

## 待办任务 (Code Review Findings)

### Phase 8: 关键 Bug 修复 (P0)

- [x] **Task 15**: 修复 DuckDB 连接资源泄漏
  - **分支**: `fix/duckdb-connection-leak`
  - **优先级**: P0
  - **状态**: ✅ 已完成
  - **文件**: `src/langrag/datasource/vdb/duckdb.py:449`
  - **解决方案**: 实现 Context Manager (`__enter__/__exit__`) + `close()` 方法
  - **测试**: 5 个新测试用例

- [x] **Task 16**: 修复 SQLiteKV 连接池问题
  - **分支**: `fix/sqlite-connection-pool`
  - **优先级**: P0
  - **状态**: ✅ 已完成
  - **文件**: `src/langrag/datasource/kv/sqlite.py:21-25`
  - **解决方案**: 连接复用 + RLock + WAL 模式 + 超时控制
  - **测试**: 8 个新测试用例

- [x] **Task 17**: 实现多数据集并行检索
  - **分支**: `feat/parallel-retrieval`
  - **优先级**: P0
  - **状态**: ✅ 已完成
  - **文件**: `src/langrag/retrieval/workflow.py:110`
  - **解决方案**: ThreadPoolExecutor 并行检索 (max_workers=5)
  - **测试**: 8 个新测试用例

---

### Phase 9: 性能优化 (P1)

- [x] **Task 18**: 优化 SemanticCache 相似度计算
  - **分支**: `perf/numpy-cosine-similarity`
  - **优先级**: P1
  - **状态**: ✅ 已完成
  - **文件**: `src/langrag/cache/semantic.py:34-58`
  - **解决方案**: NumPy 加速 + Python fallback，自动检测环境
  - **性能**: Python ~100ms vs NumPy ~1ms (1000 维向量 × 1000 次)
  - **测试**: 10 个新测试用例

- [ ] **Task 19**: 添加大文件处理保护
  - **分支**: `fix/large-file-protection`
  - **优先级**: P1
  - **文件**: `src/langrag/index_processor/extractor/providers/pdf.py`
  - **问题**: 大型 PDF 可能导致内存溢出
  - **影响**: OOM 风险
  - **解决方案**: 添加文件大小限制、分页处理
  - **预计工作量**: 1 小时

---

### Phase 10: 安全加固 (P1)

- [ ] **Task 20**: 添加 SQL 表名验证
  - **分支**: `fix/sql-table-name-validation`
  - **优先级**: P1
  - **文件**: `src/langrag/datasource/vdb/duckdb.py:84`
  - **问题**: 表名未经验证，存在 SQL 注入风险 (虽然是内部使用)
  - **解决方案**: 正则验证表名 `^[a-zA-Z0-9_]+$`
  - **预计工作量**: 30 分钟

- [ ] **Task 21**: 细化异常捕获类型
  - **分支**: `fix/specific-exception-handling`
  - **优先级**: P1
  - **文件**: 全局 (3 处 `except Exception:`)
  - **问题**: 过度捕获异常，可能掩盖系统错误
  - **涉及文件**:
    - `src/langrag/retrieval/rerank/providers/cohere.py:55`
    - `src/langrag/datasource/vdb/duckdb.py:152`
    - `src/langrag/datasource/service.py`
  - **解决方案**: 改为捕获具体异常类型
  - **预计工作量**: 2 小时

---

### Phase 11: 健壮性增强 (P2)

- [ ] **Task 22**: 添加向量维度验证
  - **分支**: `fix/vector-dimension-validation`
  - **优先级**: P2
  - **文件**: `src/langrag/datasource/vdb/duckdb.py:183-185`
  - **问题**: 假设所有文档向量维度相同，未验证
  - **解决方案**: 添加断言确保维度一致
  - **预计工作量**: 15 分钟

- [ ] **Task 23**: 处理加密 PDF 文件
  - **分支**: `fix/encrypted-pdf-handling`
  - **优先级**: P2
  - **文件**: `src/langrag/index_processor/extractor/providers/pdf.py:129`
  - **问题**: 未特别处理密码保护的 PDF
  - **解决方案**: 捕获 `pypdf.errors.FileNotDecryptedError` 并给出友好提示
  - **预计工作量**: 30 分钟

- [ ] **Task 24**: 移除空 Pass 异常处理
  - **分支**: `fix/empty-exception-handlers`
  - **优先级**: P2
  - **文件**: `src/langrag/datasource/vdb/duckdb.py:152`
  - **问题**: `except Exception: pass` 掩盖错误
  - **解决方案**: 明确捕获特定异常并记录日志
  - **预计工作量**: 15 分钟

---

### Phase 12: 容器化 (P3)

- [ ] **Task 25**: 添加 Dockerfile 和 docker-compose
  - **分支**: `feat/docker-support`
  - **描述**: 官方 Docker 镜像和 Compose 编排文件
  - **优先级**: P3
  - **预计改动**: 新建 `Dockerfile`, `docker-compose.yml`

---

### Future (待规划)

- [ ] Graph RAG 集成
- [ ] Multi-Tenant 支持
- [ ] Multi-Modal 文档支持
- [ ] Cloud Connectors (S3, GCS, Azure Blob)
- [ ] Prompt 模板管理独立模块

---

## 已完成任务 ✅

### Phase 1: 文档与规范 (P0-P1)

- [x] **Task 1**: 更新 README 中 DuckDB 全文搜索说明
  - **分支**: `docs/duckdb-fts-clarification`
  - **状态**: ✅ 已合并
  - **说明**: DuckDB FTS 已实现，README 已更新

- [x] **Task 2**: 添加 CHANGELOG.md
  - **分支**: `docs/add-changelog`
  - **状态**: ✅ 已合并

- [x] **Task 3**: 添加 CONTRIBUTING.md
  - **分支**: `docs/add-contributing`
  - **状态**: ✅ 已合并

---

### Phase 2: 可观测性 (P1)

- [x] **Task 4**: 集成 OpenTelemetry 基础设施
  - **分支**: `feat/opentelemetry-tracing`
  - **状态**: ✅ 已合并
  - **实现**: `src/langrag/observability/` 模块

---

### Phase 3: 测试覆盖率提升 (P1)

- [x] **Task 5**: 提升核心模块测试覆盖率至 80%
  - **分支**: `test/improve-coverage`
  - **状态**: ✅ 已合并
  - **结果**: 357 tests, 覆盖率显著提升

- [x] **Task 6**: 添加 web/ 目录测试
  - **分支**: `test/web-module-tests`
  - **状态**: ✅ 已合并
  - **实现**: `tests/unit/web/` 完整测试套件

---

### Phase 4: 代码重构 (P2)

- [x] **Task 7**: 拆分 rag_kernel.py
  - **分支**: `refactor/split-rag-kernel`
  - **状态**: ✅ 已合并
  - **实现**: 拆分为 `retrieval_service.py`, `vdb_manager.py` 等模块

- [x] **Task 8**: 统一配置管理
  - **分支**: `refactor/unified-config`
  - **状态**: ✅ 已合并
  - **实现**: `web/config.py` 使用 Pydantic Settings

- [x] **Task 9**: 统一文档注释语言为英文
  - **分支**: `refactor/unify-docstrings`
  - **状态**: ✅ 已合并

---

### Phase 5: 功能增强 (P1-P2)

- [x] **Task 10**: 实现 DuckDB 全文搜索
  - **分支**: `feat/duckdb-fts`
  - **状态**: ✅ 已合并
  - **实现**: DuckDB FTS 支持，RRF 混合搜索

- [x] **Task 11**: 添加 Semantic Cache 层
  - **分支**: `feat/semantic-cache`
  - **状态**: ✅ 已合并
  - **实现**: `src/langrag/cache/` 模块，支持相似度缓存、TTL、LRU 淘汰

- [x] **Task 12**: 添加 Batch Processing 支持
  - **分支**: `feat/batch-indexing`
  - **状态**: ✅ 已合并
  - **实现**: `src/langrag/batch/` 模块，支持批量嵌入和存储

---

### Phase 6: LLM Judge (P1)

- [x] **Task 13**: 实现 LLM Judge 评估框架
  - **分支**: `feat/llm-judge`
  - **状态**: ✅ 已合并
  - **实现**: `src/langrag/evaluation/` 模块
  - **指标**: Faithfulness, Answer Relevancy, Context Relevancy

---

### Phase 7: API 文档 (P2)

- [x] **Task 14**: 使用 MkDocs 生成 API 文档
  - **分支**: `docs/api-reference`
  - **状态**: ✅ 已合并
  - **实现**: MkDocs + Material 主题，完整 API 参考文档

---

## 进度追踪

| Task | 分支 | 优先级 | 状态 | 合并日期 |
|------|------|--------|------|----------|
| Task 1 | `docs/duckdb-fts-clarification` | P0 | ✅ 已完成 | 2026-01 |
| Task 2 | `docs/add-changelog` | P1 | ✅ 已完成 | 2026-01 |
| Task 3 | `docs/add-contributing` | P1 | ✅ 已完成 | 2026-01 |
| Task 4 | `feat/opentelemetry-tracing` | P1 | ✅ 已完成 | 2026-01 |
| Task 5 | `test/improve-coverage` | P1 | ✅ 已完成 | 2026-01 |
| Task 6 | `test/web-module-tests` | P1 | ✅ 已完成 | 2026-01 |
| Task 7 | `refactor/split-rag-kernel` | P2 | ✅ 已完成 | 2026-01 |
| Task 8 | `refactor/unified-config` | P2 | ✅ 已完成 | 2026-01 |
| Task 9 | `refactor/unify-docstrings` | P2 | ✅ 已完成 | 2026-01 |
| Task 10 | `feat/duckdb-fts` | P1 | ✅ 已完成 | 2026-01 |
| Task 11 | `feat/semantic-cache` | P1 | ✅ 已完成 | 2026-01 |
| Task 12 | `feat/batch-indexing` | P2 | ✅ 已完成 | 2026-01 |
| Task 13 | `feat/llm-judge` | P1 | ✅ 已完成 | 2026-01 |
| Task 14 | `docs/api-reference` | P2 | ✅ 已完成 | 2026-01 |
| Task 15 | `fix/duckdb-connection-leak` | P0 | ✅ 已完成 | 2026-01 |
| Task 16 | `fix/sqlite-connection-pool` | P0 | ✅ 已完成 | 2026-01 |
| Task 17 | `feat/parallel-retrieval` | P0 | ✅ 已完成 | 2026-01 |
| Task 18 | `perf/numpy-cosine-similarity` | P1 | ✅ 已完成 | 2026-01 |
| Task 19 | `fix/large-file-protection` | P1 | ⏳ 待开始 | - |
| Task 20 | `fix/sql-table-name-validation` | P1 | ⏳ 待开始 | - |
| Task 21 | `fix/specific-exception-handling` | P1 | ⏳ 待开始 | - |
| Task 22 | `fix/vector-dimension-validation` | P2 | ⏳ 待开始 | - |
| Task 23 | `fix/encrypted-pdf-handling` | P2 | ⏳ 待开始 | - |
| Task 24 | `fix/empty-exception-handlers` | P2 | ⏳ 待开始 | - |
| Task 25 | `feat/docker-support` | P3 | ⏳ 待开始 | - |

---

## 开发规范

1. **分支命名**: `<type>/<description>`
   - `feat/` - 新功能
   - `fix/` - Bug 修复
   - `perf/` - 性能优化
   - `docs/` - 文档
   - `refactor/` - 重构
   - `test/` - 测试

2. **Commit 规范**: 使用 Conventional Commits
   ```
   feat: add OpenTelemetry tracing support
   fix: resolve DuckDB connection leak
   perf: optimize cosine similarity with NumPy
   docs: update README with FTS clarification
   ```

3. **PR 模板**:
   ```markdown
   ## Summary
   - 简述改动内容

   ## Changes
   - 具体修改列表

   ## Test Plan
   - 测试验证方式
   ```

---

## 推荐开发顺序

### Week 1 (P0 - 必须修复)
1. Task 15: 修复 DuckDB 连接泄漏
2. Task 16: 修复 SQLiteKV 连接池
3. Task 17: 实现并行检索

### Week 2 (P1 - 性能和安全)
4. Task 18: NumPy 优化相似度计算
5. Task 19: 大文件保护
6. Task 20: SQL 表名验证
7. Task 21: 细化异常处理

### Week 3 (P2 - 健壮性)
8. Task 22-24: 边界情况处理

---

*最后更新: 2026-01-10 (Code Review)*
