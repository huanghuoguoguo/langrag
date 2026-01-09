"""
Base evaluator interface for RAG evaluation metrics.

This module defines the abstract base class that all evaluation metrics
must implement. It provides a consistent interface for evaluating RAG
system outputs using various metrics like Faithfulness, Relevancy, etc.
"""

from abc import ABC, abstractmethod
import logging

from langrag.evaluation.types import (
    EvaluationSample,
    EvaluationResult,
    BatchEvaluationResult,
)
from langrag.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """
    Abstract base class for RAG evaluation metrics.

    This class defines the interface that all evaluation metrics must implement.
    Evaluators use an LLM as a judge to assess the quality of RAG system outputs
    based on specific criteria.

    The evaluation pattern follows the LLM-as-a-Judge approach:
    1. Construct a prompt with the sample data and evaluation criteria
    2. Send to LLM for assessment
    3. Parse the LLM response to extract score and reasoning

    Subclasses must implement:
    - name: Property returning the metric name
    - evaluate: Method to evaluate a single sample

    Attributes:
        llm: The LLM instance used for judging
        temperature: Temperature for LLM calls (default 0.0 for determinism)

    Example:
        >>> evaluator = FaithfulnessEvaluator(llm)
        >>> sample = EvaluationSample(
        ...     question="What is Python?",
        ...     answer="Python is a programming language.",
        ...     contexts=["Python is a high-level programming language."]
        ... )
        >>> result = evaluator.evaluate(sample)
        >>> print(f"Faithfulness: {result.score:.2f}")
    """

    def __init__(self, llm: BaseLLM, temperature: float = 0.0):
        """
        Initialize the evaluator with an LLM.

        Args:
            llm: LLM instance implementing BaseLLM interface
            temperature: Temperature for LLM inference (default 0.0 for consistency)
        """
        self.llm = llm
        self.temperature = temperature

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this evaluation metric."""
        pass

    @abstractmethod
    def evaluate(self, sample: EvaluationSample) -> EvaluationResult:
        """
        Evaluate a single sample.

        Args:
            sample: The evaluation sample containing question, answer, and contexts

        Returns:
            EvaluationResult with score, metric name, and optional reasoning

        Raises:
            Exception: If evaluation fails (e.g., LLM error)
        """
        pass

    def evaluate_batch(
        self,
        samples: list[EvaluationSample],
        continue_on_error: bool = True
    ) -> BatchEvaluationResult:
        """
        Evaluate multiple samples.

        This is a convenience method that evaluates each sample individually.
        Subclasses may override this for more efficient batch processing.

        Args:
            samples: List of samples to evaluate
            continue_on_error: If True, continue on individual failures

        Returns:
            BatchEvaluationResult with all results and statistics
        """
        results = []
        error_count = 0

        for i, sample in enumerate(samples):
            try:
                result = self.evaluate(sample)
                results.append(result)
                logger.debug(
                    f"Evaluated sample {i+1}/{len(samples)}: "
                    f"{self.name}={result.score:.2f}"
                )
            except Exception as e:
                error_count += 1
                logger.error(f"Failed to evaluate sample {i+1}: {e}")
                if not continue_on_error:
                    raise

        return BatchEvaluationResult.from_results(
            results=results,
            metric_name=self.name,
            error_count=error_count
        )

    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM with a prompt.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The LLM's response content
        """
        messages = [{"role": "user", "content": prompt}]
        return self.llm.chat(messages, temperature=self.temperature)

    def _parse_score_from_response(
        self,
        response: str,
        score_prefix: str = "Score:"
    ) -> tuple[float, str | None]:
        """
        Parse score and reasoning from LLM response.

        Expected format:
        ```
        Reason: <explanation>
        Score: <0.0-1.0>
        ```

        Args:
            response: The LLM response text
            score_prefix: Prefix before the score (default "Score:")

        Returns:
            Tuple of (score, reason) where reason may be None

        Raises:
            ValueError: If score cannot be parsed
        """
        lines = response.strip().split("\n")
        score = None
        reason_lines = []

        for line in lines:
            line = line.strip()
            if line.lower().startswith(score_prefix.lower()):
                try:
                    score_str = line[len(score_prefix):].strip()
                    # Handle formats like "0.8" or "0.8/1.0" or "8/10"
                    if "/" in score_str:
                        parts = score_str.split("/")
                        score = float(parts[0]) / float(parts[1])
                    else:
                        score = float(score_str)
                    # Clamp to valid range
                    score = max(0.0, min(1.0, score))
                except (ValueError, IndexError):
                    pass
            elif line.lower().startswith("reason:"):
                reason_lines.append(line[7:].strip())
            elif reason_lines:
                reason_lines.append(line)

        if score is None:
            raise ValueError(f"Could not parse score from response: {response}")

        reason = " ".join(reason_lines) if reason_lines else None
        return score, reason
