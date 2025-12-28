# Index Processor Layer Design

## 1. 职责 (Responsibilities)
本模块负责 **索引构建工作流 (Indexing Pipeline)**。
它处理从原始文件到存储层的所有中间步骤：提取 -> 清洗 -> 切分 -> 增强 -> 入库。

## 2. 模块结构

```text
index_processor/
├── extractor/           # 提取器 (PDF/Word/HTML -> Text)
├── cleaner/             # 清洗器 (Remove artifacts, normalize)
├── splitter/            # 切分器 (TextSplitter)
└── processor/           # [核心] 索引编排器
    ├── base.py
    ├── paragraph.py     # 普通段落索引逻辑
    └── parent_child.py  # 父子索引逻辑
```

## 3. 核心策略设计

### 3.1 Paragraph Indexing (普通模式)
最基础的 RAG 索引方式。
1. **Load**: 读取文件。
2. **Clean**: 清洗文本。
3. **Split**: 按固定字符数 (e.g. 500 chars) 切分。
4. **Index**: 调用 `datasource.vdb.create()` 存入向量库。

### 3.2 Parent-Child Indexing (高级模式)
Dify 的高级特性，解决“切太细丢失上下文”和“切太粗检索不准”的矛盾。
1. **Split Parent**: 先切分为较大的块 (e.g. 2000 chars)。 -> 存入 **DocStore** (Postgres/Redis)。
2. **Split Child**: 将每个 Parent 切分为多个小块 (e.g. 400 chars)。 -> 存入 **VectorDB**。
3. **Mapping**: 记录 Child -> Parent 的 ID 映射。
4. **Retrieval**: 检索时搜 Child，也就是命中 Parent，最后返回 Parent 的全文给 LLM。

## 4. Dify 对应关系
- **Dify Path**: `api/core/rag/index_processor/`
- **Dify Path**: `api/core/rag/index_processor/processor/paragraph_index_processor.py`

## 5. 演进说明
- **Old (LangRAG)**: `indexing` 模块逻辑比较简单，通常是一条龙脚本。
- **New**: 
    - 拆分为 `Paragraph` 和 `ParentChild` 两种 Processor。
    - 引入 `Extractor` 和 `Cleaner` 的独立抽象。
