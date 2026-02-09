"""Graph entities for GraphRAG support."""
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """
    Represents a node in the knowledge graph.

    Entities are extracted from documents and represent real-world concepts
    like people, organizations, locations, or abstract concepts.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., min_length=1, description="Entity name")
    type: str = Field(default="UNKNOWN", description="Entity type (Person, Organization, Concept...)")
    properties: dict[str, Any] = Field(default_factory=dict, description="Additional properties")
    embedding: list[float] | None = Field(default=None, description="Vector representation")

    # Source tracking
    source_chunk_ids: list[str] = Field(
        default_factory=list,
        description="IDs of chunks this entity was extracted from"
    )

    def get_property(self, key: str, default: Any = None) -> Any:
        return self.properties.get(key, default)

    def set_property(self, key: str, value: Any) -> None:
        self.properties[key] = value

    def to_text(self) -> str:
        """Convert entity to text representation for embedding."""
        props_str = ", ".join(f"{k}: {v}" for k, v in self.properties.items())
        if props_str:
            return f"{self.name} ({self.type}): {props_str}"
        return f"{self.name} ({self.type})"

    model_config = {
        "frozen": False,
        "arbitrary_types_allowed": True
    }


class Relationship(BaseModel):
    """
    Represents an edge in the knowledge graph.

    Relationships connect two entities and describe the nature of their connection.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    source_id: str = Field(..., description="Source entity ID")
    target_id: str = Field(..., description="Target entity ID")
    type: str = Field(..., min_length=1, description="Relationship type (WORKS_AT, RELATED_TO...)")
    properties: dict[str, Any] = Field(default_factory=dict, description="Additional properties")
    weight: float = Field(default=1.0, ge=0.0, description="Relationship weight/strength")

    # Source tracking
    source_chunk_ids: list[str] = Field(
        default_factory=list,
        description="IDs of chunks this relationship was extracted from"
    )

    def get_property(self, key: str, default: Any = None) -> Any:
        return self.properties.get(key, default)

    def set_property(self, key: str, value: Any) -> None:
        self.properties[key] = value

    def to_text(self, source_name: str = "", target_name: str = "") -> str:
        """Convert relationship to text representation."""
        src = source_name or self.source_id
        tgt = target_name or self.target_id
        return f"{src} --[{self.type}]--> {tgt}"

    model_config = {
        "frozen": False,
        "arbitrary_types_allowed": True
    }


class Subgraph(BaseModel):
    """
    Represents a subgraph containing entities and relationships.

    Used as the return type for graph traversal operations.
    """
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)

    def to_context(self, max_entities: int | None = None, max_relationships: int | None = None) -> str:
        """
        Convert subgraph to text context for LLM consumption.

        Args:
            max_entities: Maximum number of entities to include
            max_relationships: Maximum number of relationships to include

        Returns:
            Formatted text representation of the subgraph
        """
        lines = []

        # Build entity lookup
        entity_map = {e.id: e for e in self.entities}

        # Entities section
        entities = self.entities[:max_entities] if max_entities else self.entities
        if entities:
            lines.append("## Entities")
            for entity in entities:
                lines.append(f"- {entity.to_text()}")

        # Relationships section
        relationships = self.relationships[:max_relationships] if max_relationships else self.relationships
        if relationships:
            lines.append("\n## Relationships")
            for rel in relationships:
                src_name = entity_map.get(rel.source_id, Entity(id=rel.source_id, name=rel.source_id)).name
                tgt_name = entity_map.get(rel.target_id, Entity(id=rel.target_id, name=rel.target_id)).name
                lines.append(f"- {rel.to_text(src_name, tgt_name)}")

        return "\n".join(lines)

    def is_empty(self) -> bool:
        return len(self.entities) == 0 and len(self.relationships) == 0

    model_config = {
        "frozen": False,
        "arbitrary_types_allowed": True
    }
