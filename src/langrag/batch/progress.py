"""
Progress tracking for batch operations.

This module provides progress tracking utilities for long-running
batch processing operations, allowing applications to display
progress bars, logs, or other feedback to users.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol


class BatchStage(Enum):
    """Stages of batch processing."""
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STORING = "storing"
    COMPLETE = "complete"


@dataclass
class BatchProgress:
    """
    Progress information for batch operations.

    This class encapsulates all progress information that can be
    reported to callbacks during batch processing.

    Attributes:
        stage: Current processing stage
        current: Number of items processed so far
        total: Total number of items to process
        percent: Completion percentage (0.0 to 1.0)
        message: Human-readable progress message
        errors: Number of errors encountered
        batch_num: Current batch number (1-indexed)
        total_batches: Total number of batches

    Example:
        >>> progress = BatchProgress(
        ...     stage=BatchStage.EMBEDDING,
        ...     current=50,
        ...     total=100,
        ...     percent=0.5,
        ...     message="Embedding batch 2/4"
        ... )
        >>> print(f"{progress.percent:.0%} - {progress.message}")
        50% - Embedding batch 2/4
    """

    stage: BatchStage
    current: int
    total: int
    percent: float
    message: str
    errors: int = 0
    batch_num: int = 0
    total_batches: int = 0

    @classmethod
    def create(
        cls,
        stage: BatchStage,
        current: int,
        total: int,
        message: str = "",
        errors: int = 0,
        batch_num: int = 0,
        total_batches: int = 0
    ) -> "BatchProgress":
        """
        Create a BatchProgress instance with auto-calculated percent.

        Args:
            stage: Current processing stage
            current: Items processed so far
            total: Total items to process
            message: Progress message
            errors: Error count
            batch_num: Current batch number
            total_batches: Total batches

        Returns:
            BatchProgress instance
        """
        percent = current / total if total > 0 else 0.0
        return cls(
            stage=stage,
            current=current,
            total=total,
            percent=percent,
            message=message or f"{stage.value}: {current}/{total}",
            errors=errors,
            batch_num=batch_num,
            total_batches=total_batches
        )


# Type alias for progress callback functions
ProgressCallback = Callable[[BatchProgress], None]


class ProgressReporter(Protocol):
    """Protocol for progress reporting implementations."""

    def report(self, progress: BatchProgress) -> None:
        """Report progress to the user/system."""
        ...


class LoggingProgressReporter:
    """
    Progress reporter that logs to Python logging.

    Useful for CLI applications or background processing where
    a progress bar isn't appropriate.

    Example:
        >>> reporter = LoggingProgressReporter()
        >>> reporter.report(progress)
        # Logs: "embedding: 50/100 (50.0%)"
    """

    def __init__(self, logger_name: str = "langrag.batch"):
        """
        Initialize with a logger.

        Args:
            logger_name: Name of the logger to use
        """
        import logging
        self.logger = logging.getLogger(logger_name)

    def report(self, progress: BatchProgress) -> None:
        """Log progress information."""
        self.logger.info(
            f"{progress.stage.value}: {progress.current}/{progress.total} "
            f"({progress.percent:.1%})"
            + (f" [errors: {progress.errors}]" if progress.errors else "")
        )


class CallbackProgressReporter:
    """
    Progress reporter that calls a user-provided callback.

    This is the default reporter used by BatchProcessor when
    an on_progress callback is provided.

    Example:
        >>> def my_callback(p: BatchProgress):
        ...     print(f"{p.percent:.0%} complete")
        >>> reporter = CallbackProgressReporter(my_callback)
        >>> reporter.report(progress)
        50% complete
    """

    def __init__(self, callback: ProgressCallback):
        """
        Initialize with a callback function.

        Args:
            callback: Function to call with progress updates
        """
        self.callback = callback

    def report(self, progress: BatchProgress) -> None:
        """Call the callback with progress."""
        self.callback(progress)
