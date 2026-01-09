"""
Tests for evaluation runner.

These tests verify:
1. EvaluationRunner initialization
2. Running multiple evaluators
3. Progress callbacks
4. Error handling
5. EvaluationReport generation
"""

from unittest.mock import MagicMock, call

import pytest

from langrag.evaluation import (
    EvaluationRunner,
    EvaluationSample,
    EvaluationReport,
    FaithfulnessEvaluator,
    AnswerRelevancyEvaluator,
    evaluate_rag,
)


class TestEvaluationRunner:
    """Tests for EvaluationRunner class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.chat.return_value = "Reason: Good answer.\nScore: 0.85"
        return llm

    @pytest.fixture
    def samples(self):
        """Create test samples."""
        return [
            EvaluationSample(
                question="What is Python?",
                answer="Python is a programming language.",
                contexts=["Python is a high-level programming language."]
            ),
            EvaluationSample(
                question="What is JavaScript?",
                answer="JavaScript is a scripting language.",
                contexts=["JavaScript is used for web development."]
            ),
        ]

    def test_runner_initialization(self, mock_llm):
        """Test runner initialization."""
        evaluators = [FaithfulnessEvaluator(mock_llm)]
        runner = EvaluationRunner(evaluators)

        assert len(runner.evaluators) == 1
        assert runner.continue_on_error is True

    def test_runner_with_single_metric(self, mock_llm, samples):
        """Test running with a single metric."""
        evaluators = [FaithfulnessEvaluator(mock_llm)]
        runner = EvaluationRunner(evaluators)

        report = runner.run(samples)

        assert isinstance(report, EvaluationReport)
        assert "faithfulness" in report.metric_results
        assert report.sample_count == 2

    def test_runner_with_multiple_metrics(self, mock_llm, samples):
        """Test running with multiple metrics."""
        evaluators = [
            FaithfulnessEvaluator(mock_llm),
            AnswerRelevancyEvaluator(mock_llm),
        ]
        runner = EvaluationRunner(evaluators)

        report = runner.run(samples)

        assert "faithfulness" in report.metric_results
        assert "answer_relevancy" in report.metric_results
        assert report.sample_count == 2

    def test_runner_empty_samples(self, mock_llm):
        """Test running with no samples."""
        evaluators = [FaithfulnessEvaluator(mock_llm)]
        runner = EvaluationRunner(evaluators)

        report = runner.run([])

        assert report.sample_count == 0
        assert len(report.metric_results) == 0

    def test_runner_progress_callback(self, mock_llm, samples):
        """Test that progress callback is called."""
        evaluators = [FaithfulnessEvaluator(mock_llm)]
        runner = EvaluationRunner(evaluators)

        progress_calls = []

        def on_progress(metric_name, current, total):
            progress_calls.append((metric_name, current, total))

        runner.run(samples, on_progress=on_progress)

        assert len(progress_calls) == 2
        assert progress_calls[0] == ("faithfulness", 1, 2)
        assert progress_calls[1] == ("faithfulness", 2, 2)

    def test_runner_error_handling(self, mock_llm, samples):
        """Test error handling during evaluation."""
        mock_llm.chat.side_effect = [
            "Reason: Good.\nScore: 0.9",
            Exception("API Error"),
        ]
        evaluators = [FaithfulnessEvaluator(mock_llm)]
        runner = EvaluationRunner(evaluators, continue_on_error=True)

        report = runner.run(samples)

        assert report.metric_results["faithfulness"].sample_count == 1
        assert report.metric_results["faithfulness"].error_count == 1

    def test_runner_fail_on_error(self, mock_llm, samples):
        """Test that errors raise when continue_on_error=False."""
        mock_llm.chat.side_effect = Exception("API Error")
        evaluators = [FaithfulnessEvaluator(mock_llm)]
        runner = EvaluationRunner(evaluators, continue_on_error=False)

        with pytest.raises(Exception, match="API Error"):
            runner.run(samples)

    def test_run_single_metric(self, mock_llm, samples):
        """Test run_single_metric method."""
        evaluator = FaithfulnessEvaluator(mock_llm)
        runner = EvaluationRunner([])

        result = runner.run_single_metric(evaluator, samples)

        assert result.sample_count == 2
        assert result.metric_name == "faithfulness"


class TestEvaluationReport:
    """Tests for EvaluationReport class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.chat.return_value = "Reason: Test.\nScore: 0.8"
        return llm

    @pytest.fixture
    def samples(self):
        """Create test samples."""
        return [
            EvaluationSample(
                question="Test?",
                answer="Test.",
                contexts=["Test context."]
            ),
        ]

    def test_report_summary(self, mock_llm, samples):
        """Test report summary generation."""
        evaluators = [
            FaithfulnessEvaluator(mock_llm),
            AnswerRelevancyEvaluator(mock_llm),
        ]
        runner = EvaluationRunner(evaluators)

        report = runner.run(samples)

        assert "faithfulness" in report.summary
        assert "answer_relevancy" in report.summary
        assert "mean" in report.summary["faithfulness"]
        assert report.summary["faithfulness"]["mean"] == pytest.approx(0.8)

    def test_report_to_dict(self, mock_llm, samples):
        """Test report to_dict method."""
        evaluators = [FaithfulnessEvaluator(mock_llm)]
        runner = EvaluationRunner(evaluators)

        report = runner.run(samples)
        report_dict = report.to_dict()

        assert "sample_count" in report_dict
        assert "metrics" in report_dict
        assert "summary" in report_dict
        assert "faithfulness" in report_dict["metrics"]
        assert "scores" in report_dict["metrics"]["faithfulness"]


class TestEvaluateRagFunction:
    """Tests for evaluate_rag convenience function."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = MagicMock()
        llm.chat.return_value = "Reason: Good.\nScore: 0.9"
        return llm

    def test_evaluate_rag(self, mock_llm):
        """Test evaluate_rag function."""
        samples = [
            EvaluationSample(
                question="What is Python?",
                answer="Python is a programming language.",
                contexts=["Python is a high-level programming language."]
            ),
        ]
        evaluators = [FaithfulnessEvaluator(mock_llm)]

        report = evaluate_rag(samples, evaluators)

        assert isinstance(report, EvaluationReport)
        assert report.sample_count == 1

    def test_evaluate_rag_with_progress(self, mock_llm):
        """Test evaluate_rag with progress callback."""
        samples = [
            EvaluationSample(
                question="Test?",
                answer="Test.",
                contexts=["Test."]
            ),
        ]
        evaluators = [FaithfulnessEvaluator(mock_llm)]

        progress_calls = []

        def on_progress(metric, current, total):
            progress_calls.append((metric, current, total))

        evaluate_rag(samples, evaluators, on_progress=on_progress)

        assert len(progress_calls) == 1
