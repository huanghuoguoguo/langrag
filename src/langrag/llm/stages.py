"""LLM使用阶段定义 (LLM Usage Stages)

定义LangRAG中所有可能使用LLM的不同阶段。
这些阶段可以独立配置不同的LLM模型。
"""

from typing import List


class LLMStage:
    """LLM使用阶段定义

    定义LangRAG中所有可能使用LLM的不同阶段。
    每个阶段都可以独立配置不同的LLM模型。
    """

    # 核心RAG阶段
    CHAT = "chat"              # 对话生成 - 最终的答案生成
    ROUTER = "router"          # 知识库路由 - 决定使用哪些知识库
    REWRITER = "rewriter"      # 查询重写 - 优化用户查询
    RERANKER = "reranker"      # 重排序 - LLM模板重排序

    # 索引阶段
    QA_INDEXING = "qa_indexing"  # QA索引生成 - 生成问答对用于索引

    # 所有阶段列表
    ALL_STAGES: List[str] = [CHAT, ROUTER, REWRITER, RERANKER, QA_INDEXING]

    # 阶段描述
    STAGE_DESCRIPTIONS = {
        CHAT: "对话生成 - 为检索结果生成最终答案",
        ROUTER: "知识库路由 - 决定查询应路由到哪些知识库",
        REWRITER: "查询重写 - 优化和改写用户查询",
        RERANKER: "重排序 - 使用LLM模板对检索结果重排序",
        QA_INDEXING: "QA索引生成 - 为文档生成问答对用于索引",
    }

    @classmethod
    def get_description(cls, stage: str) -> str:
        """获取阶段描述"""
        return cls.STAGE_DESCRIPTIONS.get(stage, f"未知阶段: {stage}")

    @classmethod
    def is_valid_stage(cls, stage: str) -> bool:
        """检查是否为有效的阶段"""
        return stage in cls.ALL_STAGES

    @classmethod
    def get_required_stages(cls) -> List[str]:
        """获取必需的阶段（RAG功能的基础）"""
        return [cls.CHAT]

    @classmethod
    def get_optional_stages(cls) -> List[str]:
        """获取可选的阶段（可以增强但非必需）"""
        return [cls.ROUTER, cls.REWRITER, cls.RERANKER, cls.QA_INDEXING]