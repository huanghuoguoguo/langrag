"""GraphRAG Index Processor for extracting knowledge graphs from documents."""
import json
from typing import Any

from loguru import logger

from langrag.datasource.graph.base import BaseGraphStore
from langrag.datasource.vdb.base import BaseVector
from langrag.entities.dataset import Dataset
from langrag.entities.document import Document
from langrag.entities.graph import Entity, Relationship
from langrag.index_processor.processor.base import BaseIndexProcessor
from langrag.llm.base import BaseLLM
from langrag.llm.embedder.base import BaseEmbedder


ENTITY_EXTRACTION_PROMPT = """You are an expert at extracting structured information from text.

Extract all entities and relationships from the following text.

Entity types to extract: {entity_types}
Relationship types to extract: {relationship_types}

Text:
{text}

Return a JSON object with the following structure:
{{
  "entities": [
    {{"name": "entity name", "type": "entity type", "properties": {{"key": "value"}}}}
  ],
  "relationships": [
    {{"source": "source entity name", "target": "target entity name", "type": "relationship type", "properties": {{}}}}
  ]
}}

Important:
- Extract ALL relevant entities and relationships
- Use the exact entity names consistently across relationships
- Only use entity types and relationship types from the provided lists
- If no entities or relationships are found, return empty arrays
- Return ONLY valid JSON, no other text

JSON:"""


class GraphProcessingError(Exception):
    """Raised when graph processing encounters errors."""
    pass


class GraphIndexProcessor(BaseIndexProcessor):
    """
    GraphRAG Index Processor.

    Extracts entities and relationships from document chunks using LLM,
    then stores them in both a graph store and a vector store for hybrid retrieval.

    Flow:
    1. Split documents into chunks
    2. Extract entities and relationships using LLM
    3. Deduplicate entities by name
    4. Generate embeddings for entities
    5. Store in GraphStore (graph structure) and VectorStore (entity vectors)
    """

    def __init__(
        self,
        graph_store: BaseGraphStore,
        vector_store: BaseVector | None = None,
        llm: BaseLLM | None = None,
        embedder: BaseEmbedder | None = None,
        splitter: Any = None,
        entity_types: list[str] | None = None,
        relationship_types: list[str] | None = None,
    ):
        """
        Initialize the GraphIndexProcessor.

        Args:
            graph_store: Graph storage backend for entities and relationships
            vector_store: Optional vector store for entity embeddings (enables vector search)
            llm: LLM for entity extraction
            embedder: Embedder for entity vectors
            splitter: Optional text splitter for chunking documents
            entity_types: Allowed entity types (None = any type)
            relationship_types: Allowed relationship types (None = any type)
        """
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.llm = llm
        self.embedder = embedder
        self.splitter = splitter
        self.entity_types = entity_types or [
            "Person", "Organization", "Location", "Event", "Concept", "Product", "Technology"
        ]
        self.relationship_types = relationship_types or [
            "WORKS_AT", "LOCATED_IN", "RELATED_TO", "PART_OF", "CREATED_BY",
            "OWNS", "KNOWS", "USES", "CAUSES", "DEPENDS_ON"
        ]

    def process(self, dataset: Dataset, documents: list[Document], **kwargs) -> dict:
        """
        Process documents to extract and index knowledge graph.

        Args:
            dataset: Dataset configuration
            documents: List of documents to process
            **kwargs: Additional arguments

        Returns:
            dict: Processing statistics
        """
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.process_async(dataset, documents, **kwargs)
        )

    async def process_async(
        self,
        dataset: Dataset,
        documents: list[Document],
        **kwargs
    ) -> dict:
        """
        Process documents asynchronously.

        Args:
            dataset: Dataset configuration
            documents: List of documents to process
            **kwargs: Additional arguments

        Returns:
            dict: Processing statistics including:
                - total_chunks: Total number of chunks processed
                - total_entities: Number of entities extracted
                - total_relationships: Number of relationships extracted
                - failed_chunks: Number of failed chunk extractions
        """
        if not self.llm:
            raise GraphProcessingError("LLM is required for entity extraction")

        # 1. Split documents if splitter is provided
        chunks = self.splitter.split_documents(documents) if self.splitter else documents

        stats = {
            "total_chunks": len(chunks),
            "total_entities": 0,
            "total_relationships": 0,
            "failed_chunks": 0,
            "failed_chunk_ids": [],
        }

        # 2. Extract entities and relationships from each chunk
        all_entities: dict[str, Entity] = {}  # name -> Entity (for deduplication)
        all_relationships: list[Relationship] = []

        for idx, chunk in enumerate(chunks):
            chunk_id = chunk.id or f"chunk_{idx}"
            try:
                entities, relationships = await self._extract_from_chunk(chunk, chunk_id)

                # Deduplicate entities by name (merge properties)
                for entity in entities:
                    key = f"{entity.type}:{entity.name}".lower()
                    if key in all_entities:
                        # Merge source chunk IDs
                        existing = all_entities[key]
                        existing.source_chunk_ids.extend(entity.source_chunk_ids)
                        # Merge properties (new values override)
                        existing.properties.update(entity.properties)
                    else:
                        all_entities[key] = entity

                all_relationships.extend(relationships)
                stats["total_entities"] = len(all_entities)
                stats["total_relationships"] += len(relationships)

            except Exception as e:
                stats["failed_chunks"] += 1
                stats["failed_chunk_ids"].append(chunk_id)
                logger.error(f"Failed to extract from chunk '{chunk_id}': {type(e).__name__}: {e}")

        if not all_entities:
            logger.warning("No entities extracted, skipping indexing")
            return stats

        # 3. Generate embeddings for entities
        entity_list = list(all_entities.values())
        if self.embedder:
            await self._embed_entities(entity_list)

        # 4. Resolve relationship entity references (name -> id)
        entity_name_to_id = {f"{e.type}:{e.name}".lower(): e.id for e in entity_list}
        valid_relationships = self._resolve_relationships(all_relationships, entity_name_to_id)

        # 5. Store in graph store
        await self.graph_store.add_entities(entity_list)
        if valid_relationships:
            await self.graph_store.add_relationships(valid_relationships)

        # 6. Store entity embeddings in vector store (optional)
        if self.vector_store and self.embedder:
            await self._store_entity_vectors(entity_list, dataset)

        logger.info(
            f"Graph indexing completed: {len(entity_list)} entities, "
            f"{len(valid_relationships)} relationships indexed"
        )

        return stats

    async def _extract_from_chunk(
        self,
        chunk: Document,
        chunk_id: str
    ) -> tuple[list[Entity], list[Relationship]]:
        """
        Extract entities and relationships from a single chunk.

        Args:
            chunk: Document chunk
            chunk_id: Chunk identifier

        Returns:
            Tuple of (entities, relationships)
        """
        prompt = ENTITY_EXTRACTION_PROMPT.format(
            entity_types=", ".join(self.entity_types),
            relationship_types=", ".join(self.relationship_types),
            text=chunk.page_content
        )

        # Call LLM
        if hasattr(self.llm, 'chat_async'):
            response = await self.llm.chat_async([{"role": "user", "content": prompt}])
        else:
            response = self.llm.chat([{"role": "user", "content": prompt}])

        # Parse JSON response
        entities, relationships = self._parse_extraction_response(response, chunk_id)

        return entities, relationships

    def _parse_extraction_response(
        self,
        response: str,
        chunk_id: str
    ) -> tuple[list[Entity], list[Relationship]]:
        """
        Parse LLM response into Entity and Relationship objects.

        Args:
            response: LLM response string (expected JSON)
            chunk_id: Source chunk ID for tracking

        Returns:
            Tuple of (entities, relationships)
        """
        entities: list[Entity] = []
        relationships: list[Relationship] = []

        try:
            # Try to extract JSON from response
            response = response.strip()
            if response.startswith("```"):
                # Remove markdown code blocks
                lines = response.split("\n")
                response = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

            data = json.loads(response)

            # Parse entities
            for e in data.get("entities", []):
                if not e.get("name"):
                    continue
                entity = Entity(
                    name=e["name"],
                    type=e.get("type", "UNKNOWN"),
                    properties=e.get("properties", {}),
                    source_chunk_ids=[chunk_id],
                )
                entities.append(entity)

            # Parse relationships (keep source/target as names for now)
            for r in data.get("relationships", []):
                if not r.get("source") or not r.get("target") or not r.get("type"):
                    continue
                rel = Relationship(
                    source_id=r["source"],  # Temporarily store name, resolve later
                    target_id=r["target"],
                    type=r["type"],
                    properties=r.get("properties", {}),
                    source_chunk_ids=[chunk_id],
                )
                relationships.append(rel)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from LLM response for chunk '{chunk_id}': {e}")
        except Exception as e:
            logger.warning(f"Error parsing extraction response for chunk '{chunk_id}': {e}")

        return entities, relationships

    async def _embed_entities(self, entities: list[Entity]) -> None:
        """
        Generate embeddings for entities.

        Args:
            entities: List of entities to embed
        """
        texts = [entity.to_text() for entity in entities]

        if hasattr(self.embedder, 'embed_async'):
            embeddings = await self.embedder.embed_async(texts)
        else:
            embeddings = self.embedder.embed(texts)

        for entity, embedding in zip(entities, embeddings):
            entity.embedding = embedding

    def _resolve_relationships(
        self,
        relationships: list[Relationship],
        entity_name_to_id: dict[str, str]
    ) -> list[Relationship]:
        """
        Resolve relationship entity names to IDs.

        Args:
            relationships: Relationships with entity names in source_id/target_id
            entity_name_to_id: Mapping from "type:name" to entity ID

        Returns:
            List of relationships with resolved entity IDs
        """
        valid_relationships: list[Relationship] = []

        for rel in relationships:
            # Try to find source entity
            source_key = None
            target_key = None

            # Search for matching entities (case-insensitive)
            for key, entity_id in entity_name_to_id.items():
                entity_name = key.split(":", 1)[1] if ":" in key else key
                if entity_name == rel.source_id.lower():
                    source_key = entity_id
                if entity_name == rel.target_id.lower():
                    target_key = entity_id

            if source_key and target_key:
                rel.source_id = source_key
                rel.target_id = target_key
                valid_relationships.append(rel)
            else:
                logger.debug(
                    f"Skipping relationship {rel.source_id} -> {rel.target_id}: "
                    "entity not found"
                )

        return valid_relationships

    async def _store_entity_vectors(
        self,
        entities: list[Entity],
        dataset: Dataset
    ) -> None:
        """
        Store entity embeddings in vector store for similarity search.

        Args:
            entities: Entities with embeddings
            dataset: Dataset configuration
        """
        # Convert entities to Documents for vector store
        docs = []
        for entity in entities:
            if entity.embedding:
                doc = Document(
                    id=entity.id,
                    page_content=entity.to_text(),
                    vector=entity.embedding,
                    metadata={
                        "entity_id": entity.id,
                        "entity_name": entity.name,
                        "entity_type": entity.type,
                        "is_entity": True,
                        "dataset_id": dataset.id,
                        **entity.properties,
                    }
                )
                docs.append(doc)

        if docs:
            if hasattr(self.vector_store, 'create_async'):
                await self.vector_store.create_async(docs)
            else:
                self.vector_store.create(docs)
