"""
RAPTOR Index Processor - Recursive Abstractive Processing for Tree-Organized Retrieval.

This implementation is adapted from RAGFlow/powerrag's RAPTOR implementation.
See: https://arxiv.org/abs/2401.18059
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from langrag.entities.dataset import Dataset
from langrag.entities.document import Document, DocumentType
from langrag.index_processor.cleaner.cleaner import Cleaner
from langrag.index_processor.processor.base import BaseIndexProcessor
from langrag.llm.base import BaseLLM

logger = logging.getLogger(__name__)


@dataclass
class RaptorConfig:
    """Configuration for RAPTOR processing."""

    max_cluster: int = 64
    """Maximum number of clusters per layer."""

    threshold: float = 0.1
    """Probability threshold for GMM cluster assignment."""

    max_token: int = 512
    """Maximum tokens for summary generation."""

    max_errors: int = 3
    """Maximum errors before aborting RAPTOR processing."""

    random_state: int = 42
    """Random state for reproducibility."""

    summarize_prompt: str = field(default_factory=lambda: (
        "Please summarize the following content concisely, "
        "preserving the key information and main ideas:\n\n{cluster_content}"
    ))
    """Prompt template for summarizing clusters. Must contain {cluster_content} placeholder."""

    umap_n_components: int = 12
    """Number of dimensions for UMAP reduction."""

    umap_metric: str = "cosine"
    """Distance metric for UMAP."""


class RaptorProcessor:
    """
    Core RAPTOR algorithm implementation.

    Recursively clusters document chunks, generates summaries for each cluster,
    and builds a hierarchical tree structure for improved retrieval.
    """

    def __init__(
        self,
        llm: BaseLLM,
        config: RaptorConfig | None = None,
    ):
        self._llm = llm
        self._config = config or RaptorConfig()
        self._error_count = 0

    def _get_optimal_clusters(
        self,
        embeddings: np.ndarray,
        random_state: int,
    ) -> int:
        """Determine optimal number of clusters using BIC criterion."""
        from sklearn.mixture import GaussianMixture

        max_clusters = min(self._config.max_cluster, len(embeddings))
        n_clusters = np.arange(1, max_clusters)
        bics = []

        for n in n_clusters:
            gm = GaussianMixture(n_components=n, random_state=random_state)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))

        optimal_clusters = n_clusters[np.argmin(bics)]
        return int(optimal_clusters)

    async def _summarize_cluster(
        self,
        texts: list[str],
    ) -> tuple[str, list[float]] | None:
        """Generate summary for a cluster and compute its embedding."""
        # Truncate texts if needed to fit context window
        max_len_per_text = max(1, (16000 - self._config.max_token) // len(texts))
        truncated_texts = [t[:max_len_per_text] for t in texts]
        cluster_content = "\n\n---\n\n".join(truncated_texts)

        try:
            # Generate summary using LLM
            prompt = self._config.summarize_prompt.format(cluster_content=cluster_content)
            summary = self._llm.chat([{"role": "user", "content": prompt}])

            # Generate embedding for the summary
            embedding = self._llm.embed_query(summary)

            return summary, embedding

        except Exception as exc:
            self._error_count += 1
            logger.warning(f"[RAPTOR] Cluster summarization failed: {exc}")

            if self._error_count >= self._config.max_errors:
                raise RuntimeError(
                    f"RAPTOR aborted after {self._error_count} errors. Last: {exc}"
                ) from exc

            return None

    async def process(
        self,
        chunks: list[tuple[str, list[float]]],
        callback: Callable[[str], None] | None = None,
    ) -> list[tuple[str, list[float], int]]:
        """
        Process chunks through RAPTOR algorithm.

        Args:
            chunks: List of (text, embedding) tuples.
            callback: Optional callback for progress updates.

        Returns:
            List of (text, embedding, layer) tuples including original chunks
            and generated summaries.
        """
        import umap
        from sklearn.mixture import GaussianMixture

        if len(chunks) <= 1:
            return [(text, emb, 0) for text, emb in chunks]

        # Filter out invalid chunks
        chunks = [(s, e) for s, e in chunks if s and e is not None and len(e) > 0]

        if len(chunks) <= 1:
            return [(text, emb, 0) for text, emb in chunks]

        # Initialize: all original chunks are layer 0
        result = [(text, emb, 0) for text, emb in chunks]
        current_layer_chunks = list(chunks)
        current_layer = 0

        while len(current_layer_chunks) > 1:
            current_layer += 1
            embeddings = np.array([emb for _, emb in current_layer_chunks])

            if len(embeddings) == 2:
                # Special case: only 2 chunks, summarize directly
                texts = [text for text, _ in current_layer_chunks]
                summary_result = await self._summarize_cluster(texts)
                if summary_result:
                    summary_text, summary_emb = summary_result
                    result.append((summary_text, summary_emb, current_layer))
                    current_layer_chunks = [(summary_text, summary_emb)]
                    if callback:
                        callback(f"Layer {current_layer}: 2 -> 1")
                else:
                    break
                continue

            # UMAP dimensionality reduction
            n_neighbors = int((len(embeddings) - 1) ** 0.8)
            n_components = min(self._config.umap_n_components, len(embeddings) - 2)

            try:
                reduced_embeddings = umap.UMAP(
                    n_neighbors=max(2, n_neighbors),
                    n_components=n_components,
                    metric=self._config.umap_metric,
                ).fit_transform(embeddings)
            except Exception as e:
                logger.warning(f"[RAPTOR] UMAP failed: {e}, stopping at layer {current_layer - 1}")
                break

            # Determine optimal clusters
            n_clusters = self._get_optimal_clusters(
                reduced_embeddings,
                self._config.random_state,
            )

            if n_clusters == 1:
                # All chunks belong to one cluster
                labels = [0] * len(reduced_embeddings)
            else:
                gm = GaussianMixture(
                    n_components=n_clusters,
                    random_state=self._config.random_state,
                )
                gm.fit(reduced_embeddings)
                probs = gm.predict_proba(reduced_embeddings)
                labels = []
                for prob in probs:
                    assigned = np.where(prob > self._config.threshold)[0]
                    labels.append(assigned[0] if len(assigned) > 0 else 0)

            # Summarize each cluster
            next_layer_chunks = []
            tasks = []

            for cluster_id in range(n_clusters):
                cluster_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
                if not cluster_indices:
                    continue

                cluster_texts = [current_layer_chunks[i][0] for i in cluster_indices]
                tasks.append(self._summarize_cluster(cluster_texts))

            # Execute summarization tasks
            summaries = await asyncio.gather(*tasks, return_exceptions=True)

            for summary_result in summaries:
                if isinstance(summary_result, Exception):
                    logger.warning(f"[RAPTOR] Cluster failed: {summary_result}")
                    continue
                if summary_result is not None:
                    summary_text, summary_emb = summary_result
                    result.append((summary_text, summary_emb, current_layer))
                    next_layer_chunks.append((summary_text, summary_emb))

            if callback:
                callback(
                    f"Layer {current_layer}: {len(current_layer_chunks)} -> {len(next_layer_chunks)}"
                )

            if not next_layer_chunks:
                break

            current_layer_chunks = next_layer_chunks

        return result


class RaptorIndexProcessor(BaseIndexProcessor):
    """
    RAPTOR Index Processor.

    Processes documents using RAPTOR algorithm:
    1. Clean -> Split -> Embed (standard flow)
    2. Apply RAPTOR clustering and summarization
    3. Store all chunks (original + summaries) in vector store

    The hierarchical summaries improve retrieval for complex queries
    that require understanding document structure.
    """

    def __init__(
        self,
        llm: BaseLLM,
        embedder: Any = None,
        vector_manager: Any = None,
        splitter: Any = None,
        cleaner: Cleaner | None = None,
        config: RaptorConfig | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ):
        """
        Initialize RAPTOR Index Processor.

        Args:
            llm: LLM instance for chat (summarization) and embedding.
            embedder: Optional separate embedder. If None, uses llm.embed_documents.
            vector_manager: Vector store manager.
            splitter: Text splitter for chunking documents.
            cleaner: Text cleaner.
            config: RAPTOR configuration.
            progress_callback: Optional callback for progress updates.
        """
        self.llm = llm
        self.embedder = embedder
        self.vector_manager = vector_manager
        self.splitter = splitter
        self.cleaner = cleaner or Cleaner()
        self.config = config or RaptorConfig()
        self.progress_callback = progress_callback

        self._raptor = RaptorProcessor(llm, self.config)

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using embedder or LLM."""
        if self.embedder is not None:
            return self.embedder.embed(texts)
        return self.llm.embed_documents(texts)

    def process(self, dataset: Dataset, documents: list[Document], **kwargs) -> None:
        """
        Process documents through RAPTOR pipeline.

        Args:
            dataset: Target dataset.
            documents: Documents to process.
        """
        all_chunks = []

        # Step 1: Clean and Split
        for doc in documents:
            cleaned_content = self.cleaner.clean(doc.page_content)
            doc.page_content = cleaned_content

            if self.splitter:
                chunks = self.splitter.split_documents([doc])
            else:
                chunks = [doc]

            for chunk in chunks:
                chunk.metadata["dataset_id"] = dataset.id
                if "document_id" in doc.metadata:
                    chunk.metadata["document_id"] = doc.metadata["document_id"]

            all_chunks.extend(chunks)

        if not all_chunks:
            return

        # Step 2: Initial Embedding
        texts = [c.page_content for c in all_chunks]
        embeddings = self._embed_texts(texts)

        for i, chunk in enumerate(all_chunks):
            chunk.vector = embeddings[i]

        # Step 3: Apply RAPTOR
        chunk_tuples = [(c.page_content, c.vector) for c in all_chunks]

        try:
            raptor_results = asyncio.run(
                self._raptor.process(chunk_tuples, callback=self.progress_callback)
            )
        except Exception as e:
            logger.error(f"[RAPTOR] Processing failed: {e}, falling back to standard indexing")
            raptor_results = [(text, emb, 0) for text, emb in chunk_tuples]

        # Step 4: Create documents from RAPTOR results
        final_chunks = []

        # Map original chunks (layer 0)
        original_idx = 0
        for text, embedding, layer in raptor_results:
            if layer == 0 and original_idx < len(all_chunks):
                # Use original chunk with its metadata
                chunk = all_chunks[original_idx]
                chunk.metadata["raptor_layer"] = 0
                final_chunks.append(chunk)
                original_idx += 1
            else:
                # Create new document for summary
                summary_doc = Document(
                    page_content=text,
                    vector=embedding,
                    type=DocumentType.CHUNK,
                    metadata={
                        "dataset_id": dataset.id,
                        "raptor_layer": layer,
                        "is_raptor_summary": True,
                    },
                )
                final_chunks.append(summary_doc)

        # Step 5: Save to Vector Store
        manager = self.vector_manager
        if manager is None:
            from langrag.datasource.vdb.global_manager import get_vector_manager
            manager = get_vector_manager()

        manager.add_texts(dataset, final_chunks)

        if self.progress_callback:
            self.progress_callback(
                f"RAPTOR complete: {len(all_chunks)} original + "
                f"{len(final_chunks) - len(all_chunks)} summaries"
            )
