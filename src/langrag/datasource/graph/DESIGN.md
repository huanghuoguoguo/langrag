# Graph Storage Layer Design

## 1. 职责 (Responsibilities)

本模块是 **图数据访问层**，负责知识图谱的存储和检索操作。
它是 GraphRAG 功能的基础设施层，为上层的 `GraphIndexProcessor` 和 `GraphRetriever` 提供统一接口。

**核心原则：**
1. **Storage Only**: 只负责图数据的存取，不负责实体抽取（属于 IndexProcessor 层）
2. **Native Power**: 优先利用图数据库的原生能力（如 Neo4j Cypher、图遍历算法）
3. **Unified Interface**: 无论底层是 NetworkX、Neo4j 还是 NebulaGraph，对上层暴露一致的接口

## 2. 模块结构

```text
datasource/graph/
├── base.py           # [Abstract] BaseGraphStore 抽象基类
├── networkx.py       # NetworkX 内存实现 (开发/测试)
├── neo4j.py          # (Planned) Neo4j 生产实现
├── factory.py        # GraphStoreFactory 工厂
└── DESIGN.md         # 本设计文档
```

## 3. 核心实体

### 3.1 Entity (实体节点)

```python
class Entity(BaseModel):
    id: str                           # 唯一标识
    name: str                         # 实体名称
    type: str                         # 实体类型 (Person, Organization, Concept...)
    properties: dict[str, Any]        # 附加属性
    embedding: list[float] | None     # 向量表示 (用于相似度搜索)
    source_chunk_ids: list[str]       # 来源 Chunk IDs (溯源)
```

### 3.2 Relationship (关系边)

```python
class Relationship(BaseModel):
    id: str                           # 唯一标识
    source_id: str                    # 源实体 ID
    target_id: str                    # 目标实体 ID
    type: str                         # 关系类型 (WORKS_AT, KNOWS, RELATED_TO...)
    properties: dict[str, Any]        # 附加属性
    weight: float                     # 关系权重/强度
    source_chunk_ids: list[str]       # 来源 Chunk IDs
```

### 3.3 Subgraph (子图)

```python
class Subgraph(BaseModel):
    entities: list[Entity]            # 实体列表
    relationships: list[Relationship] # 关系列表

    def to_context() -> str:          # 转换为 LLM 可消费的文本
```

## 4. 核心接口设计

### 4.1 BaseGraphStore

```python
class BaseGraphStore(ABC):
    # ===== 写入操作 =====
    async def add_entities(entities: list[Entity]) -> None
    async def add_relationships(relationships: list[Relationship]) -> None

    # ===== 读取操作 =====
    async def get_entity(entity_id: str) -> Entity | None
    async def get_entities(entity_ids: list[str]) -> list[Entity]
    async def get_relationship(relationship_id: str) -> Relationship | None

    # ===== 图遍历 =====
    async def get_neighbors(
        entity_ids: list[str],
        depth: int = 1,
        relationship_types: list[str] | None = None,
        direction: str = "both"  # "in", "out", "both"
    ) -> Subgraph

    # ===== 搜索 =====
    async def search_entities(
        query: str | None = None,           # 文本搜索
        query_vector: list[float] | None,   # 向量搜索
        top_k: int = 10,
        entity_types: list[str] | None = None,
        threshold: float = 0.0
    ) -> list[Entity]

    # ===== 删除操作 =====
    async def delete_entities(entity_ids: list[str]) -> int
    async def delete_relationships(relationship_ids: list[str]) -> int
    async def clear() -> None

    # ===== 统计 =====
    async def get_stats() -> dict[str, Any]

    # ===== 原生查询 (可选) =====
    async def execute_query(query: str, params: dict) -> list[dict]
```

## 5. 实现策略

### 5.1 NetworkX (内存实现)

**适用场景**: 开发、测试、小规模生产 (<100K 节点)

**特点**:
- 纯 Python 实现，无外部依赖
- 数据存储在内存，重启后丢失
- 完整的图算法支持 (最短路径、连通分量等)
- 向量搜索通过线性扫描实现

**使用示例**:
```python
from langrag.datasource.graph import GraphStoreFactory

store = GraphStoreFactory.create("networkx")
await store.add_entities([entity1, entity2])
await store.add_relationships([rel1])

# 图遍历
subgraph = await store.get_neighbors(["entity-1"], depth=2)
context = subgraph.to_context()
```

### 5.2 Neo4j (生产实现) - Planned

**适用场景**: 大规模生产部署

**特点**:
- 原生图数据库，高性能遍历
- 持久化存储
- Cypher 查询语言
- 支持向量索引 (Neo4j 5.x+)

**规划接口**:
```python
store = GraphStoreFactory.create(
    "neo4j",
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)
```

## 6. 与其他层的协作

### 6.1 与 IndexProcessor 层

```
文档 → GraphIndexProcessor → [LLM 抽取] → Entity/Relationship → GraphStore
```

GraphStore 只负责存储，实体抽取逻辑在 `GraphIndexProcessor` 中实现。

### 6.2 与 Retrieval 层

```
Query → GraphRetriever → [实体识别] → GraphStore.get_neighbors → Subgraph → Context
```

GraphRetriever 调用 GraphStore 的遍历和搜索接口，构建上下文。

## 7. 搜索策略

### 7.1 文本搜索

在实体的 `name` 和 `properties` 中进行子字符串匹配：

```python
results = await store.search_entities(query="Apple")
```

### 7.2 向量搜索

使用实体的 `embedding` 字段进行余弦相似度搜索：

```python
query_vec = embedder.embed(["Apple Inc"])
results = await store.search_entities(query_vector=query_vec[0], top_k=5)
```

### 7.3 混合搜索

同时使用文本和向量，取最高分：

```python
results = await store.search_entities(
    query="Apple",
    query_vector=query_vec[0],
    top_k=10
)
```

## 8. 图遍历

### 8.1 BFS 遍历

从起始节点开始，按层扩展：

```python
# 获取 entity-1 的 2 跳邻居
subgraph = await store.get_neighbors(
    entity_ids=["entity-1"],
    depth=2,
    direction="both"
)
```

### 8.2 方向控制

- `direction="out"`: 只沿出边遍历
- `direction="in"`: 只沿入边遍历
- `direction="both"`: 双向遍历

### 8.3 关系类型过滤

```python
# 只遍历 WORKS_AT 和 KNOWS 关系
subgraph = await store.get_neighbors(
    entity_ids=["person-1"],
    depth=1,
    relationship_types=["WORKS_AT", "KNOWS"]
)
```

## 9. 依赖配置

在 `pyproject.toml` 中添加可选依赖：

```toml
[project.optional-dependencies]
graph = [
    "networkx>=3.2.0",
]
```

安装：
```bash
uv pip install langrag[graph]
```

## 10. 未来扩展

### 10.1 社区检测

用于 Global Search 的社区摘要：

```python
communities = await store.detect_communities(algorithm="louvain")
```

### 10.2 路径查询

查找两个实体之间的路径：

```python
paths = await store.find_paths(
    source_id="entity-1",
    target_id="entity-2",
    max_depth=3
)
```

### 10.3 子图导出

导出为标准格式：

```python
data = await store.export(format="graphml")
```

---

*Last updated: 2026-02-09*
*Related Issue: #69*
