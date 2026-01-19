"""
DAG Node and Edge Definitions.

Defines the graph structure for DAG-based pipeline execution.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from langrag.core.component.base import Component


class NodeType(str, Enum):
    """Types of nodes in the pipeline graph."""
    COMPONENT = "component"   # Normal component node
    START = "start"           # Pipeline entry point
    END = "end"               # Pipeline exit point
    CONDITION = "condition"   # Conditional branching node
    PARALLEL = "parallel"     # Parallel execution node


@dataclass
class Node:
    """
    A node in the pipeline DAG.
    
    Each node wraps a Component and maintains graph connectivity.
    
    Attributes:
        id: Unique identifier for this node.
        component: The Component instance to execute.
        node_type: Type of node (component, start, end, etc).
        upstream: List of node IDs that must complete before this.
        downstream: List of node IDs to execute after this.
        config: Additional configuration for execution.
    """
    id: str
    component: Optional[Component] = None
    node_type: NodeType = NodeType.COMPONENT
    upstream: List[str] = field(default_factory=list)
    downstream: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def name(self) -> str:
        """Human-readable name for the node."""
        if self.component:
            return self.component.name
        return self.id
    
    @property
    def is_start(self) -> bool:
        return self.node_type == NodeType.START
    
    @property
    def is_end(self) -> bool:
        return self.node_type == NodeType.END
    
    @property
    def is_condition(self) -> bool:
        return self.node_type == NodeType.CONDITION
    
    def add_upstream(self, node_id: str) -> None:
        """Add an upstream dependency."""
        if node_id not in self.upstream:
            self.upstream.append(node_id)
    
    def add_downstream(self, node_id: str) -> None:
        """Add a downstream node."""
        if node_id not in self.downstream:
            self.downstream.append(node_id)


@dataclass
class Edge:
    """
    An edge connecting two nodes in the pipeline.
    
    Attributes:
        source: Source node ID.
        target: Target node ID.
        label: Optional label for the edge (useful for visualization).
    """
    source: str
    target: str
    label: Optional[str] = None
    
    def __hash__(self):
        return hash((self.source, self.target))
    
    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return self.source == other.source and self.target == other.target


@dataclass
class ConditionalEdge(Edge):
    """
    A conditional edge that only activates when a condition is met.
    
    Attributes:
        condition: A callable that takes the context and returns bool.
        condition_key: Alternatively, a key in context to check for truthiness.
        condition_value: Expected value for the condition_key.
    """
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    condition_key: Optional[str] = None
    condition_value: Any = True
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """
        Evaluate whether this edge should be taken.
        
        Args:
            context: The current execution context data.
            
        Returns:
            True if the edge should be followed.
        """
        if self.condition is not None:
            return self.condition(context)
        
        if self.condition_key is not None:
            value = context.get(self.condition_key)
            return value == self.condition_value
        
        # Default: always take the edge
        return True


def create_start_node() -> Node:
    """Create a START node for the pipeline."""
    return Node(
        id="__start__",
        node_type=NodeType.START,
    )


def create_end_node() -> Node:
    """Create an END node for the pipeline."""
    return Node(
        id="__end__",
        node_type=NodeType.END,
    )
