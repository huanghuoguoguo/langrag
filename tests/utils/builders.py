"""Test builders - 使用构建器模式创建测试对象"""

from pathlib import Path
from langrag.config.models import RAGConfig, ComponentConfig
from langrag.core.document import Document
from langrag.core.chunk import Chunk


class RAGConfigBuilder:
    """RAG 配置构建器"""

    def __init__(self):
        self._parser = ComponentConfig(type="simple_text", params={})
        self._chunker = ComponentConfig(
            type="recursive",
            params={"chunk_size": 500, "chunk_overlap": 50}
        )
        self._embedder = ComponentConfig(type="simple", params={"dimension": 384})
        self._vector_store = ComponentConfig(type="in_memory", params={})
        self._reranker = None
        self._llm = None

    def with_parser(self, parser_type: str, **params):
        """设置解析器"""
        self._parser = ComponentConfig(type=parser_type, params=params)
        return self

    def with_chunker(self, chunker_type: str, **params):
        """设置分块器"""
        self._chunker = ComponentConfig(type=chunker_type, params=params)
        return self

    def with_embedder(self, embedder_type: str, **params):
        """设置嵌入器"""
        self._embedder = ComponentConfig(type=embedder_type, params=params)
        return self

    def with_vector_store(self, store_type: str, **params):
        """设置向量存储"""
        self._vector_store = ComponentConfig(type=store_type, params=params)
        return self

    def with_reranker(self, reranker_type: str, **params):
        """设置重排序器"""
        self._reranker = ComponentConfig(type=reranker_type, params=params)
        return self

    def with_llm(self, llm_type: str, **params):
        """设置 LLM"""
        self._llm = ComponentConfig(type=llm_type, params=params)
        return self

    def build(self) -> RAGConfig:
        """构建 RAG 配置"""
        return RAGConfig(
            parser=self._parser,
            chunker=self._chunker,
            embedder=self._embedder,
            vector_store=self._vector_store,
            reranker=self._reranker,
            llm=self._llm
        )


class DocumentBuilder:
    """文档构建器"""

    def __init__(self):
        self._content = "Sample document content"
        self._metadata = {}

    def with_content(self, content: str):
        """设置内容"""
        self._content = content
        return self

    def with_metadata(self, **metadata):
        """设置元数据"""
        self._metadata.update(metadata)
        return self

    def build(self) -> Document:
        """构建文档"""
        return Document(content=self._content, metadata=self._metadata)


class ChunkBuilder:
    """Chunk 构建器"""

    def __init__(self):
        self._id = "test-chunk-1"
        self._content = "Sample chunk content"
        self._embedding = [0.1] * 384
        self._source_doc_id = "test-doc"
        self._metadata = {}

    def with_id(self, chunk_id: str):
        """设置 ID"""
        self._id = chunk_id
        return self

    def with_content(self, content: str):
        """设置内容"""
        self._content = content
        return self

    def with_embedding(self, embedding: list):
        """设置 embedding"""
        self._embedding = embedding
        return self

    def without_embedding(self):
        """移除 embedding"""
        self._embedding = None
        return self

    def with_source_doc_id(self, doc_id: str):
        """设置源文档 ID"""
        self._source_doc_id = doc_id
        return self

    def with_metadata(self, **metadata):
        """设置元数据"""
        self._metadata.update(metadata)
        return self

    def build(self) -> Chunk:
        """构建 Chunk"""
        return Chunk(
            id=self._id,
            content=self._content,
            embedding=self._embedding,
            source_doc_id=self._source_doc_id,
            metadata=self._metadata
        )


class TestFileBuilder:
    """测试文件构建器"""

    def __init__(self, temp_dir: Path):
        self._temp_dir = Path(temp_dir)
        self._filename = "test_file.txt"
        self._content = "Sample file content for testing"

    def with_filename(self, filename: str):
        """设置文件名"""
        self._filename = filename
        return self

    def with_content(self, content: str):
        """设置内容"""
        self._content = content
        return self

    def build(self) -> Path:
        """创建文件并返回路径"""
        file_path = self._temp_dir / self._filename
        file_path.write_text(self._content, encoding="utf-8")
        return file_path
