"""
Data types for RAG evaluation.

This module defines the core data structures used throughout the evaluation
framework, including evaluation samples and results.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationSample:
    """
    A single sample for RAG evaluation.

    This represents one evaluation case containing all the information
    needed to evaluate a RAG system's response.

    Attributes:
        question: The user's input question
        answer: The generated answer from the RAG system
        contexts: List of retrieved context passages
        ground_truth: Optional ground truth answer for reference-based metrics
        metadata: Optional additional metadata for the sample
    """

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """
    Result of evaluating a single sample with one metric.

    Attributes:
        score: The evaluation score (typically 0.0 to 1.0)
        metric_name: Name of the metric that produced this result
        reason: Optional explanation for the score
        details: Optional additional details from the evaluation
    """

    score: float
    metric_name: str
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate score is in valid range."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {self.score}")


@dataclass
class BatchEvaluationResult:
    """
    Results from evaluating multiple samples.

    Attributes:
        results: List of individual evaluation results
        metric_name: Name of the metric used
        mean_score: Average score across all samples
        sample_count: Number of samples evaluated
        error_count: Number of samples that failed evaluation
    """

    results: list[EvaluationResult]
    metric_name: str
    mean_score: float
    sample_count: int
    error_count: int = 0

    @classmethod
    def from_results(
        cls,
        results: list[EvaluationResult],
        metric_name: str,
        error_count: int = 0
    ) -> "BatchEvaluationResult":
        """
        Create a BatchEvaluationResult from a list of individual results.

        Args:
            results: List of individual evaluation results
            metric_name: Name of the metric
            error_count: Number of samples that failed

        Returns:
            BatchEvaluationResult with computed statistics
        """
        if not results:
            return cls(
                results=[],
                metric_name=metric_name,
                mean_score=0.0,
                sample_count=0,
                error_count=error_count
            )

        total_score = sum(r.score for r in results)
        mean_score = total_score / len(results)

        return cls(
            results=results,
            metric_name=metric_name,
            mean_score=mean_score,
            sample_count=len(results),
            error_count=error_count
        )
