# Retrieval Layer Design

## 1. 职责 (Responsibilities)
本模块是 **业务逻辑层**。
它编排整个检索过程：从接收用户 Query，到路由，再到调用 Datasource，最后 Rerank 输出。

## 2. 模块结构

```text
retrieval/
├── router/              # [Agentic] 路由模块
│   ├── base.py
│   └── llm_router.py    # 使用 LLM 判断查哪个 Dataset
├── rerank/              # 重排序模块
│   ├── base.py
│   └── cohere.py        # Cohere / BGE 适配器
└── workflow.py          # [入口] RetrievalWorkflow (Orchestrator)
```

## 3. 核心流程 (Retrieval Workflow)

### 3.1 DatasetRetrieval
这是主入口类。

```python
class DatasetRetrieval:
    def retrieve(self, query: str, context: AppContext) -> str:
        # 1. Router: 决定查询哪些 Dataset (Single / Multi)
        selected_datasets = self.router.route(query, available_datasets)
        
        # 2. Parallel Retrieve: 并发调用底层 Service
        all_docs = []
        for dataset in selected_datasets:
            docs = RetrievalService.retrieve(dataset, query)
            all_docs.extend(docs)
            
        # 3. Rerank: 对汇总结果进行重排
        if self.reranker:
            all_docs = self.reranker.rerank(query, all_docs)
            
        # 4. Format: 格式化为最终上下文 String
        return self._format_results(all_docs)
```

### 3.2 Router (Agentic RAG)
LangRAG 进化的重点。
- **Naive Router**: 简单的规则匹配（如根据用户权限）。
- **LLM Router**: 将 Dataset 作为 Tool 描述提供给 LLM (Function Calling)，让 LLM 决定调用哪个库。

## 4. Dify 对应关系
- **Dify Path**: `api/core/rag/retrieval/dataset_retrieval.py` (核心编排)
- **Dify Path**: `api/core/rag/retrieval/router/`

## 5. 演进说明
- **Old (LangRAG)**: `Retriever` 类承担了太多职责（实际上在做 Datasource 的活）。
- **New**: 
    - `Retriever` 更名为 `RetrievalWorkflow` 或保持 `DatasetRetrieval`，专注于**流程编排**。
    - 具体的数据库查询逻辑下沉到 `datasource`。
    - 增加 `router` 子模块。
