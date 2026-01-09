"""
Tests for evaluation types.

These tests verify:
1. EvaluationSample creation
2. EvaluationResult validation
3. BatchEvaluationResult statistics
"""

import pytest

from langrag.evaluation import (
    EvaluationSample,
    EvaluationResult,
)
from langrag.evaluation.types import BatchEvaluationResult


class TestEvaluationSample:
    """Tests for EvaluationSample class."""

    def test_create_sample(self):
        """Test creating a basic evaluation sample."""
        sample = EvaluationSample(
            question="What is Python?",
            answer="Python is a programming language.",
            contexts=["Python is a high-level programming language."]
        )

        assert sample.question == "What is Python?"
        assert sample.answer == "Python is a programming language."
        assert len(sample.contexts) == 1
        assert sample.ground_truth is None
        assert sample.metadata == {}

    def test_create_sample_with_ground_truth(self):
        """Test creating sample with ground truth."""
        sample = EvaluationSample(
            question="What is 2+2?",
            answer="4",
            contexts=["Basic arithmetic."],
            ground_truth="4"
        )

        assert sample.ground_truth == "4"

    def test_create_sample_with_metadata(self):
        """Test creating sample with metadata."""
        sample = EvaluationSample(
            question="Test?",
            answer="Test.",
            contexts=[],
            metadata={"source": "unit_test", "id": 123}
        )

        assert sample.metadata["source"] == "unit_test"
        assert sample.metadata["id"] == 123

    def test_create_sample_multiple_contexts(self):
        """Test creating sample with multiple contexts."""
        contexts = [
            "Context 1: Python was created by Guido van Rossum.",
            "Context 2: Python is known for its simple syntax.",
            "Context 3: Python supports multiple programming paradigms."
        ]
        sample = EvaluationSample(
            question="Tell me about Python.",
            answer="Python is a programming language.",
            contexts=contexts
        )

        assert len(sample.contexts) == 3


class TestEvaluationResult:
    """Tests for EvaluationResult class."""

    def test_create_result(self):
        """Test creating a basic evaluation result."""
        result = EvaluationResult(
            score=0.85,
            metric_name="faithfulness"
        )

        assert result.score == 0.85
        assert result.metric_name == "faithfulness"
        assert result.reason is None
        assert result.details == {}

    def test_create_result_with_reason(self):
        """Test creating result with reason."""
        result = EvaluationResult(
            score=0.7,
            metric_name="answer_relevancy",
            reason="The answer addresses the question but misses some details."
        )

        assert result.reason == "The answer addresses the question but misses some details."

    def test_create_result_with_details(self):
        """Test creating result with details."""
        result = EvaluationResult(
            score=0.9,
            metric_name="context_relevancy",
            details={"context_count": 3, "relevant_count": 2}
        )

        assert result.details["context_count"] == 3
        assert result.details["relevant_count"] == 2

    def test_invalid_score_too_high(self):
        """Test that score > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="Score must be between"):
            EvaluationResult(score=1.5, metric_name="test")

    def test_invalid_score_too_low(self):
        """Test that score < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="Score must be between"):
            EvaluationResult(score=-0.1, metric_name="test")

    def test_boundary_scores(self):
        """Test boundary score values."""
        result_zero = EvaluationResult(score=0.0, metric_name="test")
        result_one = EvaluationResult(score=1.0, metric_name="test")

        assert result_zero.score == 0.0
        assert result_one.score == 1.0


class TestBatchEvaluationResult:
    """Tests for BatchEvaluationResult class."""

    def test_from_results(self):
        """Test creating batch result from individual results."""
        results = [
            EvaluationResult(score=0.8, metric_name="faithfulness"),
            EvaluationResult(score=0.9, metric_name="faithfulness"),
            EvaluationResult(score=0.7, metric_name="faithfulness"),
        ]

        batch = BatchEvaluationResult.from_results(
            results=results,
            metric_name="faithfulness"
        )

        assert batch.sample_count == 3
        assert batch.error_count == 0
        assert batch.mean_score == pytest.approx(0.8, abs=0.01)
        assert len(batch.results) == 3

    def test_from_results_empty(self):
        """Test creating batch result from empty list."""
        batch = BatchEvaluationResult.from_results(
            results=[],
            metric_name="test"
        )

        assert batch.sample_count == 0
        assert batch.mean_score == 0.0
        assert batch.error_count == 0

    def test_from_results_with_errors(self):
        """Test creating batch result with error count."""
        results = [
            EvaluationResult(score=0.8, metric_name="test"),
        ]

        batch = BatchEvaluationResult.from_results(
            results=results,
            metric_name="test",
            error_count=2
        )

        assert batch.sample_count == 1
        assert batch.error_count == 2

    def test_mean_score_calculation(self):
        """Test that mean score is calculated correctly."""
        results = [
            EvaluationResult(score=0.5, metric_name="test"),
            EvaluationResult(score=1.0, metric_name="test"),
        ]

        batch = BatchEvaluationResult.from_results(
            results=results,
            metric_name="test"
        )

        assert batch.mean_score == pytest.approx(0.75)
