"""
Retrieval Workflow Configuration.

This module contains configuration classes and constants for the retrieval workflow.
"""

from dataclasses import dataclass


# Default max workers for parallel retrieval
DEFAULT_MAX_WORKERS = 5

# Default timeout for retrieval operations (seconds)
DEFAULT_RETRIEVAL_TIMEOUT = 30.0

# Default timeout for router operations (seconds)
DEFAULT_ROUTER_TIMEOUT = 10.0


@dataclass
class WorkflowConfig:
    """
    Configuration for RetrievalWorkflow.

    Attributes:
        max_workers: Maximum parallel retrieval threads
        retrieval_timeout: Timeout for each dataset retrieval (seconds)
        router_timeout: Timeout for router operations (seconds)
        enable_router_retry: Whether to retry router on failures
        router_max_retries: Max retry attempts for router
        enable_retrieval_retry: Whether to retry failed retrievals
        retrieval_max_retries: Max retry attempts per retrieval
    """

    max_workers: int = DEFAULT_MAX_WORKERS
    retrieval_timeout: float = DEFAULT_RETRIEVAL_TIMEOUT
    router_timeout: float = DEFAULT_ROUTER_TIMEOUT
    enable_router_retry: bool = True
    router_max_retries: int = 2
    enable_retrieval_retry: bool = True
    retrieval_max_retries: int = 2

    def __post_init__(self):
        """Validate configuration values."""
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if self.retrieval_timeout <= 0:
            raise ValueError("retrieval_timeout must be positive")
        if self.router_timeout <= 0:
            raise ValueError("router_timeout must be positive")
        if self.router_max_retries < 0:
            raise ValueError("router_max_retries cannot be negative")
        if self.retrieval_max_retries < 0:
            raise ValueError("retrieval_max_retries cannot be negative")
