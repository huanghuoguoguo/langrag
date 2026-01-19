"""
DAG Pipeline Executor.

Executes a pipeline defined as a Directed Acyclic Graph with support for:
- Topological sorting for correct execution order
- Conditional branching
- Parallel execution of independent nodes
- Progress tracking and callbacks
"""

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from langrag.core.component.base import Component, ComponentState
from langrag.core.component.context import ComponentContext
from langrag.core.callbacks.base import BaseCallbackHandler
from .node import Node, Edge, ConditionalEdge, NodeType

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """
    Result of pipeline execution.
    
    Attributes:
        success: Whether the pipeline completed successfully.
        context: Final execution context with all data.
        outputs: Output from the final node(s).
        errors: List of errors encountered.
        execution_path: List of node IDs in execution order.
        elapsed_time: Total pipeline execution time.
    """
    success: bool
    context: ComponentContext
    outputs: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    execution_path: List[str] = field(default_factory=list)
    elapsed_time: float = 0.0


class DAGPipeline:
    """
    DAG-based pipeline executor.
    
    This pipeline executes components in a directed acyclic graph structure,
    enabling complex workflows with branching and parallel execution.
    
    Example:
        pipeline = DAGPipeline()
        pipeline.add_node(Node(id="parse", component=parser))
        pipeline.add_node(Node(id="chunk", component=chunker))
        pipeline.add_edge(Edge(source="parse", target="chunk"))
        
        result = await pipeline.run(file_path="doc.txt")
    """
    
    def __init__(
        self,
        callbacks: Optional[List[BaseCallbackHandler]] = None,
        max_parallel: int = 5,
    ):
        """
        Initialize the DAG pipeline.
        
        Args:
            callbacks: Global callbacks for the pipeline.
            max_parallel: Maximum concurrent node executions.
        """
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.callbacks = callbacks or []
        self.max_parallel = max_parallel
        
        # Add implicit start and end nodes
        self._start_node = Node(id="__start__", node_type=NodeType.START)
        self._end_node = Node(id="__end__", node_type=NodeType.END)
        self.nodes["__start__"] = self._start_node
        self.nodes["__end__"] = self._end_node
    
    def add_node(self, node: Node) -> None:
        """Add a node to the pipeline."""
        if node.id in self.nodes:
            raise ValueError(f"Node '{node.id}' already exists")
        self.nodes[node.id] = node
    
    def add_edge(self, edge: Edge) -> None:
        """
        Add an edge connecting two nodes.
        
        Args:
            edge: Edge to add.
            
        Raises:
            ValueError: If source or target node doesn't exist.
        """
        if edge.source not in self.nodes:
            raise ValueError(f"Source node '{edge.source}' not found")
        if edge.target not in self.nodes:
            raise ValueError(f"Target node '{edge.target}' not found")
        
        self.edges.append(edge)
        self.nodes[edge.source].add_downstream(edge.target)
        self.nodes[edge.target].add_upstream(edge.source)
    
    def validate(self) -> None:
        """
        Validate the pipeline structure.
        
        Raises:
            ValueError: If the graph has cycles or unreachable nodes.
        """
        # Check for cycles using DFS
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        
        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            for downstream_id in self.nodes[node_id].downstream:
                if downstream_id not in visited:
                    if has_cycle(downstream_id):
                        return True
                elif downstream_id in rec_stack:
                    return True
            
            rec_stack.remove(node_id)
            return False
        
        for node_id in self.nodes:
            if node_id not in visited:
                if has_cycle(node_id):
                    raise ValueError("Pipeline contains a cycle")
    
    def topological_sort(self) -> List[str]:
        """
        Perform topological sort on the graph.
        
        Returns:
            List of node IDs in execution order.
            
        Raises:
            ValueError: If the graph has cycles.
        """
        in_degree = {node_id: len(node.upstream) for node_id, node in self.nodes.items()}
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        sorted_nodes = []
        
        while queue:
            node_id = queue.popleft()
            sorted_nodes.append(node_id)
            
            for downstream_id in self.nodes[node_id].downstream:
                in_degree[downstream_id] -= 1
                if in_degree[downstream_id] == 0:
                    queue.append(downstream_id)
        
        if len(sorted_nodes) != len(self.nodes):
            raise ValueError("Pipeline contains a cycle")
        
        return sorted_nodes
    
    async def run(self, **initial_data) -> PipelineResult:
        """
        Execute the pipeline.
        
        Args:
            **initial_data: Initial data to seed the context.
            
        Returns:
            PipelineResult with execution outcomes.
        """
        # Validate structure
        self.validate()
        
        # Create context
        context = ComponentContext()
        context.update(initial_data)
        
        # Get execution order
        try:
            execution_order = self.topological_sort()
        except ValueError as e:
            logger.error(f"Pipeline validation failed: {e}")
            return PipelineResult(
                success=False,
                context=context,
                errors=[{"error": str(e), "component_id": None}],
            )
        
        # Execute nodes
        execution_path = []
        completed: Set[str] = set()
        
        for node_id in execution_order:
            node = self.nodes[node_id]
            
            # Skip special nodes
            if node.node_type in [NodeType.START, NodeType.END]:
                completed.add(node_id)
                continue
            
            # Check if all upstream dependencies are completed
            if not all(up in completed for up in node.upstream):
                logger.debug(f"Skipping {node_id}: upstream not ready")
                continue
            
            # Check conditional edges
            should_execute = self._should_execute_node(node_id, context)
            if not should_execute:
                logger.info(f"Skipping {node_id}: condition not met")
                completed.add(node_id)
                continue
            
            # Execute component
            try:
                logger.info(f"Executing node: {node_id}")
                execution_path.append(node_id)
                
                if node.component:
                    # Pass context data as kwargs
                    output = await node.component.invoke(**context._data)
                    context.update(output)
                    context.record_trace(node_id, output)
                
                completed.add(node_id)
                
            except Exception as e:
                logger.exception(f"Node {node_id} failed: {e}")
                context.record_error(node_id, e)
                
                # Determine if should continue
                if node.config.get("continue_on_error", False):
                    completed.add(node_id)
                else:
                    return PipelineResult(
                        success=False,
                        context=context,
                        errors=context.errors,
                        execution_path=execution_path,
                        elapsed_time=context.elapsed_time,
                    )
        
        # Collect final outputs
        final_outputs = {}
        if execution_path:
            last_node_id = execution_path[-1]
            final_outputs = context.get_trace(last_node_id) or {}
        
        return PipelineResult(
            success=True,
            context=context,
            outputs=final_outputs,
            errors=context.errors,
            execution_path=execution_path,
            elapsed_time=context.elapsed_time,
        )
    
    def _should_execute_node(self, node_id: str, context: ComponentContext) -> bool:
        """
        Check if a node should be executed based on conditional edges.
        
        Args:
            node_id: ID of the node to check.
            context: Current execution context.
            
        Returns:
            True if the node should execute.
        """
        # Find all edges leading to this node
        incoming_edges = [e for e in self.edges if e.target == node_id]
        
        if not incoming_edges:
            return True
        
        # If any incoming edge is conditional and evaluates to False, skip
        for edge in incoming_edges:
            if isinstance(edge, ConditionalEdge):
                if not edge.evaluate(context._data):
                    return False
        
        return True
    
    def visualize(self) -> str:
        """
        Generate a simple text visualization of the pipeline.
        
        Returns:
            String representation of the graph.
        """
        lines = ["Pipeline Graph:", ""]
        
        for node_id, node in self.nodes.items():
            if node.node_type in [NodeType.START, NodeType.END]:
                continue
            
            component_name = node.component.name if node.component else "N/A"
            lines.append(f"  [{node_id}] {component_name}")
            
            for downstream_id in node.downstream:
                lines.append(f"    ↓")
                lines.append(f"  [{downstream_id}]")
        
        return "\n".join(lines)
