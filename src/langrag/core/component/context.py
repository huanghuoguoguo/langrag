"""
Component Execution Context.

Provides a shared context object that flows through the pipeline,
carrying data, configuration, and state between components.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
from dataclasses import dataclass, field
import time


@dataclass
class ComponentContext:
    """
    Execution context that flows through a pipeline.
    
    The context carries:
    - Shared data between components
    - Pipeline-level configuration
    - Execution metadata (run_id, timing)
    - Accumulated results
    
    Components read from and write to this context, enabling
    data flow without tight coupling between components.
    
    Example:
        ctx = ComponentContext()
        ctx.set("query", "What is RAG?")
        ctx.set("documents", [doc1, doc2])
        
        query = ctx.get("query")
        docs = ctx.get("documents", default=[])
    """
    
    # Unique identifier for this pipeline run
    run_id: UUID = field(default_factory=uuid4)
    
    # Parent run ID (for nested pipelines)
    parent_run_id: Optional[UUID] = None
    
    # Start time of the pipeline
    start_time: float = field(default_factory=time.perf_counter)
    
    # Shared data store
    _data: Dict[str, Any] = field(default_factory=dict)
    
    # Execution trace (component_id -> output)
    _trace: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Error log
    _errors: List[Dict[str, Any]] = field(default_factory=list)
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the context."""
        self._data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the context."""
        return self._data.get(key, default)
    
    def has(self, key: str) -> bool:
        """Check if a key exists in the context."""
        return key in self._data
    
    def update(self, data: Dict[str, Any]) -> None:
        """Update context with multiple values."""
        self._data.update(data)
    
    def keys(self) -> List[str]:
        """Get all keys in the context."""
        return list(self._data.keys())
    
    def record_trace(self, component_id: str, output: Dict[str, Any]) -> None:
        """Record a component's output in the trace."""
        self._trace[component_id] = {
            "output": output,
            "timestamp": time.perf_counter() - self.start_time,
        }
    
    def get_trace(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get a component's trace entry."""
        return self._trace.get(component_id)
    
    def record_error(self, component_id: str, error: Exception) -> None:
        """Record an error."""
        self._errors.append({
            "component_id": component_id,
            "error": str(error),
            "error_type": type(error).__name__,
            "timestamp": time.perf_counter() - self.start_time,
        })
    
    @property
    def errors(self) -> List[Dict[str, Any]]:
        """Get all recorded errors."""
        return self._errors.copy()
    
    @property
    def trace(self) -> Dict[str, Dict[str, Any]]:
        """Get the full execution trace."""
        return self._trace.copy()
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time since pipeline start."""
        return time.perf_counter() - self.start_time
    
    def fork(self) -> "ComponentContext":
        """
        Create a child context for nested pipeline execution.
        
        The child shares the same data but has its own run_id and trace.
        """
        child = ComponentContext(
            parent_run_id=self.run_id,
        )
        # Share the data reference (changes propagate to parent)
        child._data = self._data
        return child
    
    def to_dict(self) -> Dict[str, Any]:
        """Export context state as a dictionary."""
        return {
            "run_id": str(self.run_id),
            "parent_run_id": str(self.parent_run_id) if self.parent_run_id else None,
            "elapsed_time": self.elapsed_time,
            "data": self._data.copy(),
            "trace": self._trace.copy(),
            "errors": self._errors.copy(),
        }
