"""Indexing pipeline for document processing."""

from pathlib import Path
from loguru import logger
from typing import TYPE_CHECKING

from ..parser import BaseParser
from ..chunker import BaseChunker
from ..embedder import BaseEmbedder
from ..vector_store import BaseVectorStore
from ..config.models import StorageRole
from ..utils.performance import timer

if TYPE_CHECKING:
    pass


class IndexingPipeline:
    """Pipeline for indexing documents into a vector store.

    This pipeline orchestrates the complete indexing workflow:
    1. Parse files into documents
    2. Chunk documents into smaller pieces
    3. Generate embeddings for chunks
    4. Store chunks with embeddings in vector store(s)

    支持多数据源写入：
    - 单一存储：直接写入
    - 多个存储：根据角色决定写入内容，避免重复

    Attributes:
        parser: Document parser
        chunker: Text chunker
        embedder: Embedding generator
        vector_stores: 向量存储列表（包含角色信息）
    """

    def __init__(
        self,
        parser: BaseParser,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        vector_stores: list[tuple[BaseVectorStore, StorageRole]] | BaseVectorStore,
    ):
        """Initialize the indexing pipeline.

        Args:
            parser: Parser for reading documents
            chunker: Chunker for splitting text
            embedder: Embedder for generating vectors
            vector_stores: 存储配置
                - 单个 BaseVectorStore: 向后兼容
                - [(store, role), ...]: 多存储配置
        """
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        
        # 统一处理为列表格式
        if isinstance(vector_stores, BaseVectorStore):
            # 向后兼容：单一存储，默认为 PRIMARY 角色
            from ..config.models import StorageRole
            self.vector_stores = [(vector_stores, StorageRole.PRIMARY)]
        else:
            self.vector_stores = vector_stores
        
        logger.info(
            f"IndexingPipeline initialized with {len(self.vector_stores)} store(s): "
            f"{[(s.__class__.__name__, r.value) for s, r in self.vector_stores]}"
        )

    def index_file(self, file_path: str | Path) -> int:
        """Index a single file.

        Args:
            file_path: Path to file to index

        Returns:
            Number of chunks indexed

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid or chunks lack embeddings
        """
        file_path = Path(file_path)
        logger.info(f"Indexing file: {file_path}")

        # Validate file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        try:
            # 1. Parse
            with timer(f"Parsing {file_path.name}", threshold_ms=100):
                documents = self.parser.parse(file_path)
            logger.debug(f"Parsed {len(documents)} documents from {file_path}")

            if not documents:
                logger.warning(f"No documents parsed from {file_path}")
                return 0

        except Exception as e:
            logger.error(f"Failed to parse file {file_path}: {e}")
            raise ValueError(f"Failed to parse file {file_path}: {e}") from e

        try:
            # 2. Chunk
            with timer(f"Chunking {len(documents)} documents", threshold_ms=100):
                chunks = self.chunker.split(documents)
            logger.debug(f"Created {len(chunks)} chunks from {file_path}")

            if not chunks:
                logger.warning(f"No chunks created from {file_path}")
                return 0

        except Exception as e:
            logger.error(f"Failed to chunk documents from {file_path}: {e}")
            raise ValueError(f"Failed to chunk documents: {e}") from e

        try:
            # 3. Embed
            texts = [chunk.content for chunk in chunks]

            with timer(f"Embedding {len(chunks)} chunks", threshold_ms=500):
                embeddings = self.embedder.embed(texts)

            # Validate embeddings
            if len(embeddings) != len(chunks):
                raise ValueError(
                    f"Embedding count mismatch: {len(embeddings)} embeddings "
                    f"for {len(chunks)} chunks"
                )

            # Attach embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding

        except Exception as e:
            logger.error(f"Failed to generate embeddings for {file_path}: {e}")
            raise ValueError(f"Failed to generate embeddings: {e}") from e

        try:
            # 4. Store - 写入所有配置的存储
            with timer(f"Storing {len(chunks)} chunks to vector stores", threshold_ms=200):
                self._store_chunks(chunks)

        except Exception as e:
            logger.error(f"Failed to store chunks from {file_path}: {e}")
            raise ValueError(f"Failed to store chunks: {e}") from e

        logger.info(f"Successfully indexed {len(chunks)} chunks from {file_path}")
        return len(chunks)
    
    def _store_chunks(self, chunks):
        """将 chunks 写入所有配置的存储
        
        根据存储角色决定写入策略：
        - PRIMARY / BACKUP: 写入完整数据
        - VECTOR_ONLY: 只需要 embedding（文本也会写，但主要用于向量检索）
        - FULLTEXT_ONLY: 只需要文本内容（不需要 embedding）
        
        Args:
            chunks: 要存储的 chunk 列表
        """
        from ..config.models import StorageRole
        
        for store, role in self.vector_stores:
            logger.debug(f"Storing to {store.__class__.__name__} (role={role.value})")
            
            # 检查存储能力
            caps = store.capabilities
            
            # 根据角色和能力决定是否写入
            if role == StorageRole.FULLTEXT_ONLY:
                # 全文存储：检查是否支持全文
                if not caps.supports_fulltext:
                    logger.warning(
                        f"Store {store.__class__.__name__} marked as FULLTEXT_ONLY "
                        f"but doesn't support fulltext search, skipping"
                    )
                    continue
                # 全文存储也需要完整 chunks（因为 add 方法需要）
                # VDB 内部会根据自己的能力决定存储什么
                
            elif role == StorageRole.VECTOR_ONLY:
                # 向量存储：检查是否支持向量
                if not caps.supports_vector:
                    logger.warning(
                        f"Store {store.__class__.__name__} marked as VECTOR_ONLY "
                        f"but doesn't support vector search, skipping"
                    )
                    continue
                # 确保所有 chunks 都有 embedding
                if any(c.embedding is None for c in chunks):
                    logger.error(f"Some chunks lack embeddings, cannot store to {store.__class__.__name__}")
                    continue
            
            # 写入存储
            try:
                store.add(chunks)
                logger.info(
                    f"✓ Stored {len(chunks)} chunks to "
                    f"{store.__class__.__name__} ({role.value})"
                )
            except Exception as e:
                logger.error(
                    f"✗ Failed to store chunks to {store.__class__.__name__}: {e}"
                )
                # 继续写入其他存储，不中断流程
                continue

    def index_files(self, file_paths: list[str | Path]) -> int:
        """Index multiple files.

        Args:
            file_paths: List of file paths to index

        Returns:
            Total number of chunks indexed across all successful files

        Note:
            This method continues processing remaining files even if some fail.
            Check logs for individual file failures.
        """
        total = 0
        successful = 0
        failed = []

        logger.info(f"Starting batch indexing of {len(file_paths)} files")

        for path in file_paths:
            try:
                num_chunks = self.index_file(path)
                total += num_chunks
                successful += 1
                logger.debug(f"✓ Indexed {path}: {num_chunks} chunks")

            except Exception as e:
                failed.append((path, str(e)))
                logger.error(f"✗ Failed to index {path}: {e}")
                # Continue processing remaining files

        # Summary logging
        logger.info(
            f"Batch indexing completed: "
            f"{successful}/{len(file_paths)} files successful, "
            f"{len(failed)} failed, "
            f"{total} total chunks"
        )

        if failed:
            logger.warning(
                f"Failed files: {[str(path) for path, _ in failed]}"
            )

        return total
