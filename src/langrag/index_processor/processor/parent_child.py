from typing import Any

from langrag.datasource.kv.base import BaseKVStore
from langrag.datasource.vdb.base import BaseVector
from langrag.entities.dataset import Dataset
from langrag.entities.document import Document, DocumentType
from langrag.index_processor.cleaner.cleaner import Cleaner
from langrag.index_processor.processor.base import BaseIndexProcessor


class ParentChildIndexProcessor(BaseIndexProcessor):
    """
    Parent-Child Indexing Strategy.
    Splits documents into Parent chunks (stored in KV) and Child chunks (embedded in VDB).
    Children link back to Parent via metadata 'parent_id'.
    """

    def __init__(
        self,
        vector_store: BaseVector,
        kv_store: BaseKVStore,
        embedder: Any,
        parent_splitter: Any,
        child_splitter: Any,
        cleaner: Cleaner = None
    ):
        self.vector_store = vector_store
        self.kv_store = kv_store
        self.embedder = embedder
        self.parent_splitter = parent_splitter
        self.child_splitter = child_splitter
        self.cleaner = cleaner or Cleaner()

    def process(self, dataset: Dataset, documents: list[Document], **kwargs) -> None:
        parents_data = {}
        child_chunks = []

        for doc in documents:
            # 1. Clean
            doc.page_content = self.cleaner.clean(doc.page_content)

            # 2. Split into Parents
            # Assuming parent_splitter.split_documents returns list[Document]
            parents = self.parent_splitter.split_documents([doc])

            for parent in parents:
                parent.type = DocumentType.PARENT
                # Store parent to KV
                # key strategy: unique id. parent.id should be unique.
                parents_data[parent.id] = parent.page_content

                # 3. Split Parent into Children
                # Children splitter should ideally handle small chunks
                children = self.child_splitter.split_documents([parent])

                for child in children:
                    child.type = DocumentType.CHUNK
                    child.metadata['parent_id'] = parent.id
                    child.metadata['dataset_id'] = dataset.id
                    if 'document_id' in doc.metadata:
                        # Propagate origin document ID
                        child.metadata['document_id'] = doc.metadata['document_id']

                child_chunks.extend(children)

        # 4. Save Parents to KV
        if parents_data:
            self.kv_store.mset(parents_data)

        # 5. Embed Children
        if child_chunks:
            texts = [c.page_content for c in child_chunks]
            embeddings = self.embedder.embed(texts)
            for i, chunk in enumerate(child_chunks):
                chunk.vector = embeddings[i]

            # 6. Save Children to VDB
            self.vector_store.create(child_chunks)
