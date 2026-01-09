"""
Evaluation runner for batch RAG evaluation.

This module provides the EvaluationRunner class that orchestrates running
multiple evaluation metrics across multiple samples, with support for
progress tracking and result aggregation.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable

from langrag.evaluation.base import BaseEvaluator
from langrag.evaluation.types import (
    EvaluationSample,
    EvaluationResult,
    BatchEvaluationResult,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """
    Complete evaluation report with results from all metrics.

    Attributes:
        metric_results: Dict mapping metric name to batch results
        sample_count: Number of samples evaluated
        summary: Dict with summary statistics per metric
    """

    metric_results: dict[str, BatchEvaluationResult]
    sample_count: int
    summary: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self):
        """Compute summary statistics."""
        for metric_name, batch_result in self.metric_results.items():
            self.summary[metric_name] = {
                "mean": batch_result.mean_score,
                "sample_count": batch_result.sample_count,
                "error_count": batch_result.error_count,
            }

    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            "sample_count": self.sample_count,
            "metrics": {
                name: {
                    "mean_score": result.mean_score,
                    "sample_count": result.sample_count,
                    "error_count": result.error_count,
                    "scores": [r.score for r in result.results]
                }
                for name, result in self.metric_results.items()
            },
            "summary": self.summary
        }


# Progress callback type
EvaluationProgressCallback = Callable[[str, int, int], None]


class EvaluationRunner:
    """
    Runner for executing multiple evaluation metrics on sample datasets.

    This class orchestrates the evaluation process, running each configured
    metric on all samples and aggregating results into a comprehensive report.

    Features:
    - Multiple metrics in a single run
    - Progress tracking via callbacks
    - Error handling with configurable behavior
    - Comprehensive result aggregation

    Example:
        >>> runner = EvaluationRunner(
        ...     evaluators=[
        ...         FaithfulnessEvaluator(llm),
        ...         AnswerRelevancyEvaluator(llm),
        ...         ContextRelevancyEvaluator(llm)
        ...     ]
        ... )
        >>> samples = [
        ...     EvaluationSample(
        ...         question="What is Python?",
        ...         answer="Python is a programming language.",
        ...         contexts=["Python is a high-level programming language."]
        ...     )
        ... ]
        >>> report = runner.run(samples)
        >>> print(f"Faithfulness: {report.summary['faithfulness']['mean']:.2f}")
    """

    def __init__(
        self,
        evaluators: list[BaseEvaluator],
        continue_on_error: bool = True
    ):
        """
        Initialize the evaluation runner.

        Args:
            evaluators: List of evaluator instances to run
            continue_on_error: If True, continue on individual failures
        """
        self.evaluators = evaluators
        self.continue_on_error = continue_on_error

    def run(
        self,
        samples: list[EvaluationSample],
        on_progress: EvaluationProgressCallback | None = None
    ) -> EvaluationReport:
        """
        Run all evaluators on all samples.

        Args:
            samples: List of samples to evaluate
            on_progress: Optional callback for progress updates
                        Called with (metric_name, current_sample, total_samples)

        Returns:
            EvaluationReport with all results and statistics
        """
        if not samples:
            logger.warning("No samples provided for evaluation")
            return EvaluationReport(
                metric_results={},
                sample_count=0
            )

        metric_results: dict[str, BatchEvaluationResult] = {}
        total_metrics = len(self.evaluators)

        logger.info(
            f"Starting evaluation: {len(samples)} samples, "
            f"{total_metrics} metrics"
        )

        for metric_idx, evaluator in enumerate(self.evaluators):
            metric_name = evaluator.name
            logger.info(
                f"Running metric {metric_idx + 1}/{total_metrics}: {metric_name}"
            )

            results: list[EvaluationResult] = []
            error_count = 0

            for sample_idx, sample in enumerate(samples):
                if on_progress:
                    on_progress(metric_name, sample_idx + 1, len(samples))

                try:
                    result = evaluator.evaluate(sample)
                    results.append(result)

                    logger.debug(
                        f"[{metric_name}] Sample {sample_idx + 1}/{len(samples)}: "
                        f"score={result.score:.2f}"
                    )

                except Exception as e:
                    error_count += 1
                    logger.error(
                        f"[{metric_name}] Sample {sample_idx + 1} failed: {e}"
                    )
                    if not self.continue_on_error:
                        raise

            batch_result = BatchEvaluationResult.from_results(
                results=results,
                metric_name=metric_name,
                error_count=error_count
            )
            metric_results[metric_name] = batch_result

            logger.info(
                f"[{metric_name}] Complete: mean={batch_result.mean_score:.2f}, "
                f"errors={error_count}"
            )

        report = EvaluationReport(
            metric_results=metric_results,
            sample_count=len(samples)
        )

        summary_str = ", ".join(
            f"{k}={v['mean']:.2f}" for k, v in report.summary.items()
        )
        logger.info(f"Evaluation complete. Summary: {summary_str}")

        return report

    def run_single_metric(
        self,
        evaluator: BaseEvaluator,
        samples: list[EvaluationSample],
        on_progress: EvaluationProgressCallback | None = None
    ) -> BatchEvaluationResult:
        """
        Run a single metric on all samples.

        Convenience method for running just one metric.

        Args:
            evaluator: The evaluator to run
            samples: Samples to evaluate
            on_progress: Optional progress callback

        Returns:
            BatchEvaluationResult for the metric
        """
        results: list[EvaluationResult] = []
        error_count = 0

        for idx, sample in enumerate(samples):
            if on_progress:
                on_progress(evaluator.name, idx + 1, len(samples))

            try:
                result = evaluator.evaluate(sample)
                results.append(result)
            except Exception as e:
                error_count += 1
                logger.error(f"Sample {idx + 1} failed: {e}")
                if not self.continue_on_error:
                    raise

        return BatchEvaluationResult.from_results(
            results=results,
            metric_name=evaluator.name,
            error_count=error_count
        )


def evaluate_rag(
    samples: list[EvaluationSample],
    evaluators: list[BaseEvaluator],
    on_progress: EvaluationProgressCallback | None = None
) -> EvaluationReport:
    """
    Convenience function to evaluate RAG samples with multiple metrics.

    This is a simple wrapper around EvaluationRunner for one-off evaluations.

    Args:
        samples: List of samples to evaluate
        evaluators: List of evaluators to run
        on_progress: Optional progress callback

    Returns:
        EvaluationReport with all results

    Example:
        >>> from langrag.evaluation import evaluate_rag
        >>> from langrag.evaluation.metrics import FaithfulnessEvaluator
        >>> report = evaluate_rag(
        ...     samples=[...],
        ...     evaluators=[FaithfulnessEvaluator(llm)]
        ... )
    """
    runner = EvaluationRunner(evaluators)
    return runner.run(samples, on_progress)
