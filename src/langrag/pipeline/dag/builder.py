"""
Pipeline Builder - Fluent API for constructing DAG pipelines.

Provides a user-friendly interface for building complex pipelines
without manually managing nodes and edges.
"""

from typing import Any, Callable, Dict, List, Optional
import logging

from langrag.core.component.base import Component
from langrag.core.callbacks.base import BaseCallbackHandler
from .executor import DAGPipeline
from .node import Node, Edge, ConditionalEdge, NodeType

logger = logging.getLogger(__name__)


class PipelineBuilder:
    """
    Fluent API for building DAG pipelines.
    
    Example:
        builder = PipelineBuilder()
        builder.add_step("parse", parser_component) \\
               .add_step("chunk", chunker_component) \\
               .connect("parse", "chunk") \\
               .add_step("embed", embedder_component) \\
               .connect("chunk", "embed")
        
        pipeline = builder.build()
        result = await pipeline.run(file_path="doc.txt")
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize the builder.
        
        Args:
            name: Optional name for the pipeline (for logging).
        """
        self.name = name or "pipeline"
        self._nodes: Dict[str, Node] = {}
        self._edges: List[Edge] = []
        self._callbacks: List[BaseCallbackHandler] = []
        self._last_added_node_id: Optional[str] = None
    
    def add_step(
        self,
        node_id: str,
        component: Component,
        config: Optional[Dict[str, Any]] = None,
    ) -> "PipelineBuilder":
        """
        Add a component as a step in the pipeline.
        
        Args:
            node_id: Unique identifier for this step.
            component: Component instance to execute.
            config: Additional configuration for this node.
            
        Returns:
            Self for chaining.
        """
        if node_id in self._nodes:
            raise ValueError(f"Node '{node_id}' already exists")
        
        node = Node(
            id=node_id,
            component=component,
            node_type=NodeType.COMPONENT,
            config=config or {},
        )
        
        self._nodes[node_id] = node
        self._last_added_node_id = node_id
        
        logger.debug(f"Added step '{node_id}': {component.__class__.__name__}")
        return self
    
    def connect(
        self,
        source: str,
        target: str,
        label: Optional[str] = None,
    ) -> "PipelineBuilder":
        """
        Connect two nodes with an edge.
        
        Args:
            source: Source node ID.
            target: Target node ID.
            label: Optional label for the edge.
            
        Returns:
            Self for chaining.
        """
        edge = Edge(source=source, target=target, label=label)
        self._edges.append(edge)
        
        logger.debug(f"Connected '{source}' -> '{target}'")
        return self
    
    def connect_conditional(
        self,
        source: str,
        target: str,
        condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
        condition_key: Optional[str] = None,
        condition_value: Any = True,
        label: Optional[str] = None,
    ) -> "PipelineBuilder":
        """
        Connect two nodes with a conditional edge.
        
        The edge will only be followed if the condition evaluates to True.
        
        Args:
            source: Source node ID.
            target: Target node ID.
            condition: Callable that takes context dict and returns bool.
            condition_key: Alternatively, key in context to check.
            condition_value: Expected value for condition_key.
            label: Optional label for the edge.
            
        Returns:
            Self for chaining.
        """
        edge = ConditionalEdge(
            source=source,
            target=target,
            condition=condition,
            condition_key=condition_key,
            condition_value=condition_value,
            label=label,
        )
        self._edges.append(edge)
        
        logger.debug(f"Connected '{source}' -> '{target}' (conditional)")
        return self
    
    def then(self, node_id: str, component: Component) -> "PipelineBuilder":
        """
        Add a step and automatically connect it to the last added step.
        
        Args:
            node_id: ID for the new step.
            component: Component to execute.
            
        Returns:
            Self for chaining.
        """
        self.add_step(node_id, component)
        
        if self._last_added_node_id and self._last_added_node_id != node_id:
            # Get the previous node (the one before the current last)
            nodes_list = list(self._nodes.keys())
            current_idx = nodes_list.index(node_id)
            if current_idx > 0:
                prev_node_id = nodes_list[current_idx - 1]
                self.connect(prev_node_id, node_id)
        
        return self
    
    def branch(
        self,
        branches: Dict[str, str],
        condition_key: str,
    ) -> "PipelineBuilder":
        """
        Create conditional branches from the last added node.
        
        Args:
            branches: Dict of {condition_value: target_node_id}.
            condition_key: Key in context to check for branching.
            
        Returns:
            Self for chaining.
        """
        if not self._last_added_node_id:
            raise ValueError("No previous node to branch from")
        
        for value, target_id in branches.items():
            self.connect_conditional(
                source=self._last_added_node_id,
                target=target_id,
                condition_key=condition_key,
                condition_value=value,
            )
        
        return self
    
    def add_callback(self, callback: BaseCallbackHandler) -> "PipelineBuilder":
        """
        Add a global callback handler.
        
        Args:
            callback: Callback handler instance.
            
        Returns:
            Self for chaining.
        """
        self._callbacks.append(callback)
        return self
    
    def build(self, max_parallel: int = 5) -> DAGPipeline:
        """
        Build the final DAG pipeline.
        
        Args:
            max_parallel: Maximum concurrent node executions.
            
        Returns:
            Configured DAGPipeline instance.
        """
        pipeline = DAGPipeline(
            callbacks=self._callbacks,
            max_parallel=max_parallel,
        )
        
        # Add all nodes
        for node in self._nodes.values():
            pipeline.add_node(node)
        
        # Add all edges
        for edge in self._edges:
            pipeline.add_edge(edge)
        
        logger.info(f"Built pipeline '{self.name}' with {len(self._nodes)} nodes")
        return pipeline
    
    def visualize(self) -> str:
        """
        Generate a text visualization of the pipeline being built.
        
        Returns:
            String representation.
        """
        lines = [f"Pipeline: {self.name}", ""]
        
        for node_id, node in self._nodes.items():
            comp_name = node.component.__class__.__name__ if node.component else "N/A"
            lines.append(f"  {node_id}: {comp_name}")
        
        lines.append("\nConnections:")
        for edge in self._edges:
            edge_type = "conditional" if isinstance(edge, ConditionalEdge) else "direct"
            label = f" ({edge.label})" if edge.label else ""
            lines.append(f"  {edge.source} -> {edge.target} [{edge_type}]{label}")
        
        return "\n".join(lines)


# === Helper functions for common pipeline patterns ===

def create_linear_pipeline(
    steps: List[tuple[str, Component]],
    name: str = "linear_pipeline",
) -> DAGPipeline:
    """
    Create a simple linear pipeline (step1 -> step2 -> step3 -> ...).
    
    Args:
        steps: List of (node_id, component) tuples.
        name: Pipeline name.
        
    Returns:
        Configured DAGPipeline.
    """
    builder = PipelineBuilder(name=name)
    
    for node_id, component in steps:
        builder.add_step(node_id, component)
    
    # Connect sequentially
    for i in range(len(steps) - 1):
        builder.connect(steps[i][0], steps[i + 1][0])
    
    return builder.build()
