"""
Tests for evaluation metrics.

These tests verify:
1. FaithfulnessEvaluator
2. AnswerRelevancyEvaluator
3. ContextRelevancyEvaluator
4. Score parsing from LLM responses
"""

from unittest.mock import MagicMock

import pytest

from langrag.evaluation import (
    EvaluationSample,
    FaithfulnessEvaluator,
    AnswerRelevancyEvaluator,
    ContextRelevancyEvaluator,
)
from langrag.evaluation.base import BaseEvaluator


class TestBaseEvaluator:
    """Tests for BaseEvaluator class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        return llm

    def test_parse_score_standard_format(self, mock_llm):
        """Test parsing score from standard format."""
        # Create a concrete evaluator for testing
        evaluator = FaithfulnessEvaluator(mock_llm)

        response = """Reason: The answer is well-grounded in the context.
Score: 0.85"""

        score, reason = evaluator._parse_score_from_response(response)

        assert score == pytest.approx(0.85)
        assert "well-grounded" in reason

    def test_parse_score_fraction_format(self, mock_llm):
        """Test parsing score from fraction format."""
        evaluator = FaithfulnessEvaluator(mock_llm)

        response = """Reason: Good answer.
Score: 8/10"""

        score, reason = evaluator._parse_score_from_response(response)

        assert score == pytest.approx(0.8)

    def test_parse_score_no_reason(self, mock_llm):
        """Test parsing score without reason."""
        evaluator = FaithfulnessEvaluator(mock_llm)

        response = """Score: 0.9"""

        score, reason = evaluator._parse_score_from_response(response)

        assert score == pytest.approx(0.9)
        assert reason is None

    def test_parse_score_clamp_high(self, mock_llm):
        """Test that scores > 1.0 are clamped."""
        evaluator = FaithfulnessEvaluator(mock_llm)

        response = """Score: 1.5"""

        score, _ = evaluator._parse_score_from_response(response)

        assert score == 1.0

    def test_parse_score_clamp_low(self, mock_llm):
        """Test that scores < 0.0 are clamped."""
        evaluator = FaithfulnessEvaluator(mock_llm)

        response = """Score: -0.5"""

        score, _ = evaluator._parse_score_from_response(response)

        assert score == 0.0

    def test_parse_score_invalid_raises(self, mock_llm):
        """Test that invalid response raises ValueError."""
        evaluator = FaithfulnessEvaluator(mock_llm)

        response = """This response has no score."""

        with pytest.raises(ValueError, match="Could not parse score"):
            evaluator._parse_score_from_response(response)


class TestFaithfulnessEvaluator:
    """Tests for FaithfulnessEvaluator class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.chat.return_value = """Reason: The answer correctly states that Python is a programming language, which is supported by the context.
Score: 0.95"""
        return llm

    @pytest.fixture
    def sample(self):
        """Create a sample for testing."""
        return EvaluationSample(
            question="What is Python?",
            answer="Python is a programming language.",
            contexts=["Python is a high-level programming language."]
        )

    def test_evaluator_name(self, mock_llm):
        """Test that evaluator returns correct name."""
        evaluator = FaithfulnessEvaluator(mock_llm)
        assert evaluator.name == "faithfulness"

    def test_evaluate_returns_result(self, mock_llm, sample):
        """Test that evaluate returns EvaluationResult."""
        evaluator = FaithfulnessEvaluator(mock_llm)

        result = evaluator.evaluate(sample)

        assert result.score == pytest.approx(0.95)
        assert result.metric_name == "faithfulness"
        assert result.reason is not None

    def test_evaluate_calls_llm(self, mock_llm, sample):
        """Test that evaluate calls LLM with correct messages."""
        evaluator = FaithfulnessEvaluator(mock_llm)

        evaluator.evaluate(sample)

        mock_llm.chat.assert_called_once()
        call_args = mock_llm.chat.call_args
        messages = call_args[0][0]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "Python" in messages[0]["content"]
        assert "faithfulness" in messages[0]["content"].lower()

    def test_evaluate_with_multiple_contexts(self, mock_llm):
        """Test evaluation with multiple contexts."""
        sample = EvaluationSample(
            question="What is Python?",
            answer="Python is a high-level programming language.",
            contexts=[
                "Python was created by Guido van Rossum.",
                "Python is a high-level programming language.",
                "Python supports multiple paradigms."
            ]
        )
        evaluator = FaithfulnessEvaluator(mock_llm)

        result = evaluator.evaluate(sample)

        assert result.details["context_count"] == 3

    def test_evaluate_with_empty_contexts(self, mock_llm):
        """Test evaluation with empty contexts."""
        sample = EvaluationSample(
            question="What is Python?",
            answer="I don't know.",
            contexts=[]
        )
        evaluator = FaithfulnessEvaluator(mock_llm)

        result = evaluator.evaluate(sample)

        assert result.details["context_count"] == 0


class TestAnswerRelevancyEvaluator:
    """Tests for AnswerRelevancyEvaluator class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.chat.return_value = """Reason: The answer directly addresses the question about what Python is.
Score: 0.9"""
        return llm

    @pytest.fixture
    def sample(self):
        """Create a sample for testing."""
        return EvaluationSample(
            question="What is Python?",
            answer="Python is a programming language.",
            contexts=["Python is a high-level programming language."]
        )

    def test_evaluator_name(self, mock_llm):
        """Test that evaluator returns correct name."""
        evaluator = AnswerRelevancyEvaluator(mock_llm)
        assert evaluator.name == "answer_relevancy"

    def test_evaluate_returns_result(self, mock_llm, sample):
        """Test that evaluate returns EvaluationResult."""
        evaluator = AnswerRelevancyEvaluator(mock_llm)

        result = evaluator.evaluate(sample)

        assert result.score == pytest.approx(0.9)
        assert result.metric_name == "answer_relevancy"

    def test_evaluate_includes_lengths(self, mock_llm, sample):
        """Test that result includes length details."""
        evaluator = AnswerRelevancyEvaluator(mock_llm)

        result = evaluator.evaluate(sample)

        assert "question_length" in result.details
        assert "answer_length" in result.details


class TestContextRelevancyEvaluator:
    """Tests for ContextRelevancyEvaluator class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.chat.return_value = """Reason: The context directly addresses the topic of Python as a programming language.
Score: 0.85"""
        return llm

    @pytest.fixture
    def sample(self):
        """Create a sample for testing."""
        return EvaluationSample(
            question="What is Python?",
            answer="Python is a programming language.",
            contexts=["Python is a high-level programming language."]
        )

    def test_evaluator_name(self, mock_llm):
        """Test that evaluator returns correct name."""
        evaluator = ContextRelevancyEvaluator(mock_llm)
        assert evaluator.name == "context_relevancy"

    def test_evaluate_returns_result(self, mock_llm, sample):
        """Test that evaluate returns EvaluationResult."""
        evaluator = ContextRelevancyEvaluator(mock_llm)

        result = evaluator.evaluate(sample)

        assert result.score == pytest.approx(0.85)
        assert result.metric_name == "context_relevancy"

    def test_evaluate_includes_context_stats(self, mock_llm, sample):
        """Test that result includes context statistics."""
        evaluator = ContextRelevancyEvaluator(mock_llm)

        result = evaluator.evaluate(sample)

        assert "context_count" in result.details
        assert "total_context_length" in result.details


class TestEvaluateBatch:
    """Tests for batch evaluation functionality."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM that returns different scores."""
        llm = MagicMock()
        llm.chat.side_effect = [
            "Reason: Good.\nScore: 0.9",
            "Reason: Okay.\nScore: 0.7",
            "Reason: Excellent.\nScore: 1.0",
        ]
        return llm

    @pytest.fixture
    def samples(self):
        """Create multiple samples."""
        return [
            EvaluationSample(
                question=f"Question {i}?",
                answer=f"Answer {i}.",
                contexts=[f"Context {i}."]
            )
            for i in range(3)
        ]

    def test_evaluate_batch(self, mock_llm, samples):
        """Test batch evaluation."""
        evaluator = FaithfulnessEvaluator(mock_llm)

        batch_result = evaluator.evaluate_batch(samples)

        assert batch_result.sample_count == 3
        assert batch_result.mean_score == pytest.approx((0.9 + 0.7 + 1.0) / 3, abs=0.01)
        assert batch_result.error_count == 0

    def test_evaluate_batch_with_errors(self, mock_llm, samples):
        """Test batch evaluation with errors."""
        mock_llm.chat.side_effect = [
            "Reason: Good.\nScore: 0.9",
            Exception("LLM Error"),
            "Reason: Excellent.\nScore: 1.0",
        ]
        evaluator = FaithfulnessEvaluator(mock_llm)

        batch_result = evaluator.evaluate_batch(samples, continue_on_error=True)

        assert batch_result.sample_count == 2
        assert batch_result.error_count == 1

    def test_evaluate_batch_fail_on_error(self, mock_llm, samples):
        """Test batch evaluation fails when continue_on_error=False."""
        mock_llm.chat.side_effect = [
            "Reason: Good.\nScore: 0.9",
            Exception("LLM Error"),
        ]
        evaluator = FaithfulnessEvaluator(mock_llm)

        with pytest.raises(Exception, match="LLM Error"):
            evaluator.evaluate_batch(samples, continue_on_error=False)
