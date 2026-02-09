"""GraphRetriever for knowledge graph-based retrieval.

This module implements the retrieval component of GraphRAG, which:
1. Identifies entities in the query using LLM or vector similarity
2. Traverses the knowledge graph to find related entities
3. Constructs context from the subgraph for LLM consumption
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

from langrag.datasource.graph.base import BaseGraphStore
from langrag.entities.dataset import RetrievalContext
from langrag.entities.graph import Entity, Subgraph
from langrag.llm.base import BaseLLM
from langrag.llm.embedder.base import BaseEmbedder


logger = logging.getLogger(__name__)


ENTITY_RECOGNITION_PROMPT = """You are an expert at identifying entities in text.

Given the following query, extract all entities that should be looked up in a knowledge graph.

Query: {query}

Return a JSON array of entities with their types:
[
  {{"name": "entity name", "type": "entity type"}}
]

Entity types: {entity_types}

Important:
- Extract only entities that are explicitly mentioned or strongly implied
- Use the exact entity names as they appear in the query
- Return ONLY valid JSON, no other text

JSON:"""


class GraphRetrieverConfig(BaseModel):
    """Configuration for GraphRetriever."""

    # Entity recognition
    use_llm_recognition: bool = Field(
        default=True,
        description="Use LLM for entity recognition (more accurate but slower)"
    )
    use_vector_recognition: bool = Field(
        default=True,
        description="Use vector similarity for entity recognition (faster fallback)"
    )
    entity_types: list[str] = Field(
        default_factory=lambda: [
            "Person", "Organization", "Location", "Event",
            "Concept", "Product", "Technology"
        ],
        description="Entity types to recognize"
    )

    # Graph traversal
    traversal_depth: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum depth for graph traversal"
    )
    max_entities: int = Field(
        default=20,
        ge=1,
        description="Maximum entities to include in context"
    )
    max_relationships: int = Field(
        default=30,
        ge=1,
        description="Maximum relationships to include in context"
    )

    # Search settings
    vector_top_k: int = Field(
        default=5,
        ge=1,
        description="Number of entities to retrieve via vector search"
    )
    similarity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for vector search"
    )

    # Relationship filtering
    relationship_types: list[str] | None = Field(
        default=None,
        description="Filter traversal by relationship types (None = all)"
    )
    direction: str = Field(
        default="both",
        description="Traversal direction: 'in', 'out', or 'both'"
    )

    model_config = {"frozen": False}


class GraphRetriever:
    """
    GraphRAG Retriever.

    Retrieves relevant context from a knowledge graph based on the query.
    Combines entity recognition (LLM or vector-based) with graph traversal
    to build rich contextual information for the LLM.

    Flow:
    1. Query → Entity Recognition (LLM/Vector)
    2. Entities → Graph Traversal (BFS)
    3. Subgraph → Context Formatting

    Example:
        retriever = GraphRetriever(
            graph_store=networkx_store,
            embedder=openai_embedder,
            llm=openai_llm,
            config=GraphRetrieverConfig(traversal_depth=2)
        )

        # Retrieve context for a query
        contexts = await retriever.retrieve_async(
            query="What is the relationship between Apple and Steve Jobs?",
            top_k=5
        )
    """

    def __init__(
        self,
        graph_store: BaseGraphStore,
        embedder: BaseEmbedder | None = None,
        llm: BaseLLM | None = None,
        config: GraphRetrieverConfig | None = None,
    ):
        """
        Initialize the GraphRetriever.

        Args:
            graph_store: Graph storage backend
            embedder: Embedder for vector-based entity recognition
            llm: LLM for entity recognition from query
            config: Retriever configuration
        """
        self.graph_store = graph_store
        self.embedder = embedder
        self.llm = llm
        self.config = config or GraphRetrieverConfig()

        # Validate configuration
        if self.config.use_llm_recognition and not self.llm:
            logger.warning(
                "LLM recognition enabled but no LLM provided, "
                "falling back to vector recognition"
            )
            self.config.use_llm_recognition = False

        if self.config.use_vector_recognition and not self.embedder:
            logger.warning(
                "Vector recognition enabled but no embedder provided, "
                "disabling vector recognition"
            )
            self.config.use_vector_recognition = False

        if not self.config.use_llm_recognition and not self.config.use_vector_recognition:
            raise ValueError(
                "At least one recognition method (LLM or vector) must be enabled "
                "with the corresponding component provided"
            )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalContext]:
        """
        Retrieve context from the knowledge graph (sync wrapper).

        Args:
            query: User query
            top_k: Maximum number of context items to return

        Returns:
            List of RetrievalContext with graph-based context
        """
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.retrieve_async(query, top_k)
        )

    async def retrieve_async(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalContext]:
        """
        Retrieve context from the knowledge graph asynchronously.

        Args:
            query: User query
            top_k: Maximum number of context items to return

        Returns:
            List of RetrievalContext with graph-based context
        """
        logger.info(f"GraphRetriever: Processing query '{query[:50]}...'")

        # Step 1: Recognize entities in the query
        recognized_entities = await self._recognize_entities(query)

        if not recognized_entities:
            logger.warning("No entities recognized in query")
            return []

        logger.info(
            f"Recognized {len(recognized_entities)} entities: "
            f"{[e.name for e in recognized_entities]}"
        )

        # Step 2: Traverse the graph from recognized entities
        entity_ids = [e.id for e in recognized_entities]
        subgraph = await self.graph_store.get_neighbors(
            entity_ids=entity_ids,
            depth=self.config.traversal_depth,
            relationship_types=self.config.relationship_types,
            direction=self.config.direction,
        )

        if subgraph.is_empty():
            logger.warning("Graph traversal returned empty subgraph")
            return []

        logger.info(
            f"Graph traversal found {len(subgraph.entities)} entities, "
            f"{len(subgraph.relationships)} relationships"
        )

        # Step 3: Format subgraph as context
        contexts = self._format_subgraph_contexts(
            subgraph, recognized_entities, top_k
        )

        return contexts

    async def retrieve_subgraph(
        self,
        query: str,
    ) -> Subgraph:
        """
        Retrieve the raw subgraph for a query.

        Useful when you need direct access to entities and relationships
        rather than formatted context.

        Args:
            query: User query

        Returns:
            Subgraph containing relevant entities and relationships
        """
        recognized_entities = await self._recognize_entities(query)

        if not recognized_entities:
            return Subgraph(entities=[], relationships=[])

        entity_ids = [e.id for e in recognized_entities]
        return await self.graph_store.get_neighbors(
            entity_ids=entity_ids,
            depth=self.config.traversal_depth,
            relationship_types=self.config.relationship_types,
            direction=self.config.direction,
        )

    async def _recognize_entities(self, query: str) -> list[Entity]:
        """
        Recognize entities in the query.

        Tries LLM recognition first (if enabled), then falls back to
        vector similarity search.

        Args:
            query: User query

        Returns:
            List of recognized entities
        """
        entities: list[Entity] = []

        # Try LLM-based recognition
        if self.config.use_llm_recognition and self.llm:
            try:
                entities = await self._recognize_entities_llm(query)
                if entities:
                    return entities
            except Exception as e:
                logger.warning(f"LLM entity recognition failed: {e}")

        # Fall back to vector-based recognition
        if self.config.use_vector_recognition and self.embedder:
            try:
                entities = await self._recognize_entities_vector(query)
            except Exception as e:
                logger.warning(f"Vector entity recognition failed: {e}")

        return entities

    async def _recognize_entities_llm(self, query: str) -> list[Entity]:
        """
        Recognize entities using LLM.

        Args:
            query: User query

        Returns:
            List of entities found in the graph that match LLM output
        """
        import json

        prompt = ENTITY_RECOGNITION_PROMPT.format(
            query=query,
            entity_types=", ".join(self.config.entity_types)
        )

        # Call LLM
        if hasattr(self.llm, 'chat_async'):
            response = await self.llm.chat_async([{"role": "user", "content": prompt}])
        else:
            response = self.llm.chat([{"role": "user", "content": prompt}])

        # Parse response
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            entity_data = json.loads(response)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM entity recognition response")
            return []

        # Look up entities in the graph
        entities = []
        for item in entity_data:
            name = item.get("name", "")
            if not name:
                continue

            # Search for entity by name in graph
            found = await self.graph_store.search_entities(
                query=name,
                top_k=1,
                threshold=0.7
            )
            if found:
                entities.append(found[0])

        return entities

    async def _recognize_entities_vector(self, query: str) -> list[Entity]:
        """
        Recognize entities using vector similarity.

        Args:
            query: User query

        Returns:
            List of similar entities from the graph
        """
        # Embed the query
        if hasattr(self.embedder, 'embed_async'):
            embeddings = await self.embedder.embed_async([query])
        else:
            embeddings = self.embedder.embed([query])

        if not embeddings:
            return []

        query_vector = embeddings[0]

        # Search for similar entities
        entities = await self.graph_store.search_entities(
            query_vector=query_vector,
            top_k=self.config.vector_top_k,
            threshold=self.config.similarity_threshold
        )

        return entities

    def _format_subgraph_contexts(
        self,
        subgraph: Subgraph,
        seed_entities: list[Entity],
        top_k: int,
    ) -> list[RetrievalContext]:
        """
        Format subgraph as RetrievalContext objects.

        Creates context items prioritizing:
        1. Seed entities (directly matched from query)
        2. Entities with most connections
        3. Relationships involving seed entities

        Args:
            subgraph: Retrieved subgraph
            seed_entities: Entities directly matched from query
            top_k: Maximum contexts to return

        Returns:
            List of RetrievalContext objects
        """
        contexts: list[RetrievalContext] = []
        seed_ids = {e.id for e in seed_entities}

        # Build entity lookup and connection counts
        entity_map = {e.id: e for e in subgraph.entities}
        connection_counts: dict[str, int] = {}
        for rel in subgraph.relationships:
            connection_counts[rel.source_id] = connection_counts.get(rel.source_id, 0) + 1
            connection_counts[rel.target_id] = connection_counts.get(rel.target_id, 0) + 1

        # Score and sort entities
        scored_entities: list[tuple[float, Entity]] = []
        for entity in subgraph.entities:
            score = 0.0
            # Seed entities get highest priority
            if entity.id in seed_ids:
                score += 10.0
            # Connection count contributes to score
            score += connection_counts.get(entity.id, 0) * 0.5
            scored_entities.append((score, entity))

        scored_entities.sort(key=lambda x: x[0], reverse=True)

        # Create context from top entities
        top_entities = scored_entities[:self.config.max_entities]

        # Create a focused subgraph with top entities
        top_entity_ids = {e.id for _, e in top_entities}
        relevant_relationships = [
            r for r in subgraph.relationships
            if r.source_id in top_entity_ids and r.target_id in top_entity_ids
        ][:self.config.max_relationships]

        focused_subgraph = Subgraph(
            entities=[e for _, e in top_entities],
            relationships=relevant_relationships
        )

        # Format as single comprehensive context
        context_text = focused_subgraph.to_context()

        if context_text:
            # Primary context: full subgraph
            contexts.append(RetrievalContext(
                document_id="graph_context",
                content=context_text,
                score=1.0,
                metadata={
                    "source": "graph_rag",
                    "entity_count": len(focused_subgraph.entities),
                    "relationship_count": len(focused_subgraph.relationships),
                    "seed_entities": [e.name for e in seed_entities],
                }
            ))

        # Additional contexts: individual seed entity details
        for i, entity in enumerate(seed_entities[:top_k - 1]):
            if i >= top_k - 1:
                break

            # Find relationships for this entity
            entity_rels = [
                r for r in subgraph.relationships
                if r.source_id == entity.id or r.target_id == entity.id
            ][:5]

            entity_subgraph = Subgraph(
                entities=[entity],
                relationships=entity_rels
            )

            entity_context = entity_subgraph.to_context()
            if entity_context:
                contexts.append(RetrievalContext(
                    document_id=f"entity_{entity.id}",
                    content=entity_context,
                    score=0.8 - (i * 0.1),
                    metadata={
                        "source": "graph_rag",
                        "entity_id": entity.id,
                        "entity_name": entity.name,
                        "entity_type": entity.type,
                        "source_chunk_ids": entity.source_chunk_ids,
                    }
                ))

        return contexts[:top_k]

    async def get_entity_context(
        self,
        entity_id: str,
        depth: int = 1,
    ) -> RetrievalContext | None:
        """
        Get context for a specific entity.

        Args:
            entity_id: Entity ID to look up
            depth: Traversal depth from entity

        Returns:
            RetrievalContext for the entity, or None if not found
        """
        entity = await self.graph_store.get_entity(entity_id)
        if not entity:
            return None

        subgraph = await self.graph_store.get_neighbors(
            entity_ids=[entity_id],
            depth=depth,
            direction=self.config.direction,
        )

        context_text = subgraph.to_context()

        return RetrievalContext(
            document_id=f"entity_{entity_id}",
            content=context_text,
            score=1.0,
            metadata={
                "source": "graph_rag",
                "entity_id": entity.id,
                "entity_name": entity.name,
                "entity_type": entity.type,
                "entity_count": len(subgraph.entities),
                "relationship_count": len(subgraph.relationships),
            }
        )
