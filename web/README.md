# LangRAG Web Application

业务层应用，使用 `langrag` 核心库作为 RAG 引擎。

## 架构设计

```
web/
├── app.py              # FastAPI 应用入口
├── core/               # 核心组件
│   ├── database.py     # 数据库连接与初始化
│   └── rag_kernel.py   # RAG 内核封装（与 langrag 交互）
├── models/             # 数据模型
│   └── database.py     # SQLModel 业务实体
├── services/           # 业务逻辑层
│   ├── kb_service.py       # 知识库服务
│   ├── document_service.py # 文档处理服务
│   └── embedder_service.py # 模型配置服务
├── routers/            # API 路由
│   ├── kb.py           # 知识库 API
│   ├── document.py     # 文档上传 API
│   ├── search.py       # 检索 API
│   └── config.py       # 配置 API
├── static/             # 前端静态文件
│   ├── index.html
│   ├── style.css
│   └── script.js
└── data/               # SQLite 数据库文件（自动创建）
    └── app.db
```

## 分层说明

### 1. langrag 核心层
- 提供 RAG 基础能力：文档解析、分块、向量检索
- 不管理业务数据和模型配置
- 通过依赖注入接收外部组件（Embedder、VectorStore）

### 2. 业务层（本应用）
- 管理知识库元数据（名称、描述、配置）
- 管理文档记录（文件名、状态、处理时间）
- 管理模型配置（API Key、Base URL）
- 使用 SQLite 持久化业务数据

## 运行

```bash
# 从项目根目录运行 (使用 -m 以确保模块路径正确)
uv run python -m web.app
```

访问：http://localhost:8000

## API 文档

启动后访问：http://localhost:8000/docs
