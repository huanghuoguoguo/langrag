from abc import ABC, abstractmethod
from typing import Any

class BasePipeline(ABC):
    """
    Abstract base class for all pipelines.
    Pipelines orchestrate multiple components to achieve a high-level goal.
    """
    
    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Execute the pipeline."""
        pass
