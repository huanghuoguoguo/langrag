import chromadb
from chromadb.config import Settings as ChromaSettings

from langrag.datasource.vdb.base import BaseVector
from langrag.entities.dataset import Dataset
from langrag.entities.document import Document


class ChromaVector(BaseVector):
    """
    Chroma Vector Store Implementation.
    """

    def __init__(
        self,
        dataset: Dataset,
        persist_directory: str | None = None,
        host: str | None = None,
        port: int | None = None,
        ssl: bool = False,
        headers: dict[str, str] | None = None,
        **kwargs
    ):
        super().__init__(dataset)

        if host and port:
            # Remote connection
            self._client = chromadb.HttpClient(
                host=host,
                port=port,
                ssl=ssl,
                headers=headers,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
        else:
            # Local persistence
            path = persist_directory or "./chroma_db"
            self._client = chromadb.PersistentClient(
                path=path,
                settings=ChromaSettings(anonymized_telemetry=False)
            )

        # Get or create collection
        # We enforce cosine distance for consistency
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def create(self, texts: list[Document], **kwargs) -> None:
        """Create collection and add texts."""
        # Chroma `get_or_create_collection` handles creation.
        # So we just add texts.
        self.add_texts(texts, **kwargs)

    def add_texts(self, texts: list[Document], **kwargs) -> None:
        """Add texts to existing collection."""
        if not texts:
            return

        ids = [doc.id for doc in texts]
        documents = [doc.page_content for doc in texts]

        # Chroma requires embeddings. If not present, it will try to use its default EF.
        # But our architecture assumes embeddings are generated before calling this.
        # We should check if vectors are present.
        embeddings = [doc.vector for doc in texts]
        if any(e is None for e in embeddings):
             # Fallback or Error?
             # In our architecture, embedding happens in IndexProcessor.
             # So we expect vectors here.
             # However, Chroma CAN generate embeddings if we don't provide them
             # (using its default lightweight embedder).
             # Let's strictly require them or handle None based on config.
             # For now, let's assume if vector is None, we pass None to Chroma
             # (Chroma will calculate it if it has an embedding function set, otherwise error)
             pass

        # Metadata processing
        metadatas = []
        for doc in texts:
            # Flatten metadata and convert non-primitives to str (Chroma limitation)
            meta = {}
            for k, v in doc.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                else:
                    meta[k] = str(v)
            metadatas.append(meta)

        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings if embeddings[0] is not None else None,
            metadatas=metadatas
        )

    def search(
        self,
        query: str,
        query_vector: list[float] | None,
        top_k: int = 4,
        **kwargs
    ) -> list[Document]:
        """
        Search for documents.
        """

        # Hybrid support: Chroma is vector only (mostly).
        # So we ignore search_type="hybrid" request unless we implement client-side RRF (which we explicitly avoid in VDB layer now).
        # We behave as semantic search.

        if query_vector is None:
            # If no vector provided, Chroma can't search unless it has internal embedding function.
            # Our service layer should ensure query_vector is present if using semantic search.
            # But let's check if we can query by text (only works if collection has EF).
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
        else:
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

        # Parse results
        if not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        out_docs = []
        for i, doc_id in enumerate(ids):
            # Calculate score from distance (Cosine distance 0..2)
            # score = 1 - distance (approx)
            distance = dists[i]
            score = 1 - distance

            meta = metas[i] if metas[i] else {}
            # Inject score into metadata so Service can read it
            meta['score'] = score

            doc = Document(
                page_content=docs[i],
                metadata=meta,
                id=doc_id
            )
            out_docs.append(doc)

        return out_docs

    def delete_by_ids(self, ids: list[str]) -> None:
        if not ids:
            return
        self._collection.delete(ids=ids)

    def delete(self) -> None:
        self._client.delete_collection(self.collection_name)
