from typing import Any

from langrag.entities.dataset import Dataset
from langrag.entities.document import Document
from langrag.index_processor.processor.base import BaseIndexProcessor
from langrag.index_processor.cleaner.cleaner import Cleaner
# from langrag.index_processor.splitter.factory import SplitterFactory # Assuming existence
# from langrag.llm.embedder.base import BaseEmbedder # Assuming existence
from langrag.datasource.vdb.base import BaseVector

class ParagraphIndexProcessor(BaseIndexProcessor):
    """
    Standard Paragraph Indexer.
    Flow: Clean -> Split -> Embed -> Save to VectorDB.
    """

    def __init__(
        self, 
        vector_store: BaseVector, 
        embedder: Any, # BaseEmbedder
        splitter: Any = None,
        cleaner: Cleaner = None
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.splitter = splitter # TODO: Use Factory to get default splitter if None
        self.cleaner = cleaner or Cleaner()

    def process(self, dataset: Dataset, documents: list[Document], **kwargs) -> None:
        """
        Process the documents pipeline.
        """
        all_chunks = []
        
        for doc in documents:
            # 1. Clean
            cleaned_content = self.cleaner.clean(doc.page_content)
            doc.page_content = cleaned_content
            
            # 2. Split
            # Assuming splitter.split_documents takes [Document] and returns [Document] (Chunks)
            # If splitter is None, we treat the whole doc as one chunk (not recommended usually)
            if self.splitter:
                # We need to adapt our Document entity to what the splitter expects 
                # or ensure splitter uses our entity. 
                # For now assuming compatible interface or list of texts.
                chunks = self.splitter.split_documents([doc])
            else:
                chunks = [doc]
            
            # 3. Add Metadata
            for chunk in chunks:
                chunk.metadata['dataset_id'] = dataset.id
                # Ensure doc_id is set (it's auto generated in our Entity)
                # Propagate original document_id if present
                if 'document_id' in doc.metadata:
                    chunk.metadata['document_id'] = doc.metadata['document_id']

            all_chunks.extend(chunks)

        if not all_chunks:
            return

        # 4. Embed (Batch embedding)
        texts_to_embed = [c.page_content for c in all_chunks]
        embeddings = self.embedder.embed_documents(texts_to_embed)
        
        for i, chunk in enumerate(all_chunks):
            chunk.vector = embeddings[i]

        # 5. Save to VDB
        self.vector_store.create(all_chunks)
        
        # 6. Save to Keyword Store (Optional / Economy Mode)
        # if dataset.indexing_technique == 'economy': ...
