"""
Component Base Classes.

Defines the core abstractions for pipeline components:
- ComponentState: Enum for component lifecycle states
- Component: Abstract base class for all pipeline components
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID, uuid4
import time
import logging

from langrag.core.callbacks.base import BaseCallbackHandler

logger = logging.getLogger(__name__)


class ComponentState(str, Enum):
    """Lifecycle states of a component."""
    PENDING = "pending"      # Not yet started
    RUNNING = "running"      # Currently executing
    COMPLETED = "completed"  # Finished successfully
    FAILED = "failed"        # Finished with error
    SKIPPED = "skipped"      # Skipped due to condition


class Component(ABC):
    """
    Abstract base class for all pipeline components.
    
    A Component is a single unit of work in a pipeline. It has:
    - A unique ID
    - Input/Output contracts
    - Lifecycle hooks (before/after invoke)
    - Error handling
    - Callback integration for observability
    
    Subclasses must implement `_invoke()` which contains the actual logic.
    
    Example:
        class MyParser(Component):
            component_type = "parser"
            
            def _invoke(self, file_path: str, **kwargs) -> Dict[str, Any]:
                content = Path(file_path).read_text()
                return {"documents": [Document(page_content=content)]}
    """
    
    # Class-level component type identifier
    component_type: str = "base"
    
    def __init__(
        self,
        component_id: Optional[str] = None,
        name: Optional[str] = None,
        callbacks: Optional[list[BaseCallbackHandler]] = None,
    ):
        """
        Initialize the component.
        
        Args:
            component_id: Unique identifier (auto-generated if None).
            name: Human-readable name for logging/display.
            callbacks: List of callback handlers for observability.
        """
        self.id = component_id or f"{self.component_type}_{uuid4().hex[:8]}"
        self.name = name or self.__class__.__name__
        self.callbacks = callbacks or []
        
        # Runtime state
        self._state = ComponentState.PENDING
        self._output: Dict[str, Any] = {}
        self._error: Optional[Exception] = None
        self._run_id: Optional[UUID] = None
        self._elapsed_time: float = 0.0
    
    @property
    def state(self) -> ComponentState:
        """Current state of the component."""
        return self._state
    
    @property
    def output(self) -> Dict[str, Any]:
        """Output data produced by this component after execution."""
        return self._output
    
    @property
    def error(self) -> Optional[Exception]:
        """Error if the component failed."""
        return self._error
    
    @property
    def elapsed_time(self) -> float:
        """Execution time in seconds."""
        return self._elapsed_time
    
    def set_output(self, key: str, value: Any) -> None:
        """Set an output value."""
        self._output[key] = value
    
    def get_output(self, key: str, default: Any = None) -> Any:
        """Get an output value."""
        return self._output.get(key, default)
    
    def reset(self) -> None:
        """Reset the component state for re-execution."""
        self._state = ComponentState.PENDING
        self._output = {}
        self._error = None
        self._run_id = None
        self._elapsed_time = 0.0
    
    async def invoke(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the component.
        
        This is the main entry point. It handles:
        1. State management
        2. Timing
        3. Callback notifications
        4. Error handling
        
        Args:
            **kwargs: Input parameters for the component.
            
        Returns:
            Dict containing the component's output.
        """
        self._run_id = uuid4()
        self._state = ComponentState.RUNNING
        start_time = time.perf_counter()
        
        # Notify callbacks
        self._notify_start(**kwargs)
        
        try:
            # Execute the actual logic
            result = await self._invoke(**kwargs)
            
            # Store output
            if isinstance(result, dict):
                self._output.update(result)
            else:
                self._output["result"] = result
            
            self._state = ComponentState.COMPLETED
            self._notify_end()
            
        except Exception as e:
            self._error = e
            self._state = ComponentState.FAILED
            self._notify_error(e)
            logger.exception(f"Component {self.name} failed: {e}")
            raise
        
        finally:
            self._elapsed_time = time.perf_counter() - start_time
        
        return self._output
    
    @abstractmethod
    async def _invoke(self, **kwargs) -> Dict[str, Any]:
        """
        Internal invoke method to be implemented by subclasses.
        
        This method contains the actual component logic.
        
        Args:
            **kwargs: Input parameters.
            
        Returns:
            Dict containing output data.
        """
        pass
    
    def _notify_start(self, **kwargs) -> None:
        """Notify callbacks that component is starting."""
        for cb in self.callbacks:
            try:
                if hasattr(cb, "on_component_start"):
                    cb.on_component_start(
                        component_id=self.id,
                        component_name=self.name,
                        component_type=self.component_type,
                        run_id=self._run_id,
                        inputs=kwargs,
                    )
            except Exception as e:
                logger.warning(f"Callback error on start: {e}")
    
    def _notify_end(self) -> None:
        """Notify callbacks that component completed."""
        for cb in self.callbacks:
            try:
                if hasattr(cb, "on_component_end"):
                    cb.on_component_end(
                        component_id=self.id,
                        component_name=self.name,
                        component_type=self.component_type,
                        run_id=self._run_id,
                        outputs=self._output,
                        elapsed_time=self._elapsed_time,
                    )
            except Exception as e:
                logger.warning(f"Callback error on end: {e}")
    
    def _notify_error(self, error: Exception) -> None:
        """Notify callbacks that component failed."""
        for cb in self.callbacks:
            try:
                if hasattr(cb, "on_error"):
                    cb.on_error(
                        error=error,
                        run_id=self._run_id,
                    )
            except Exception as e:
                logger.warning(f"Callback error on error: {e}")
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.id}, state={self.state.value})>"
