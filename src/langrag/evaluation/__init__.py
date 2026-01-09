"""
RAG Evaluation Module.

This module provides LLM-as-a-Judge evaluation capabilities for RAG systems.
It implements common evaluation metrics following the RAGAS framework approach.

Metrics included:
- Faithfulness: Is the answer grounded in the retrieved context?
- Answer Relevancy: Does the answer address the question asked?
- Context Relevancy: Is the retrieved context relevant to the question?

Usage:
    >>> from langrag.evaluation import (
    ...     EvaluationRunner,
    ...     EvaluationSample,
    ...     FaithfulnessEvaluator,
    ...     AnswerRelevancyEvaluator,
    ...     ContextRelevancyEvaluator,
    ... )
    >>>
    >>> # Create evaluators with an LLM
    >>> evaluators = [
    ...     FaithfulnessEvaluator(llm),
    ...     AnswerRelevancyEvaluator(llm),
    ...     ContextRelevancyEvaluator(llm),
    ... ]
    >>>
    >>> # Create samples
    >>> samples = [
    ...     EvaluationSample(
    ...         question="What is Python?",
    ...         answer="Python is a programming language.",
    ...         contexts=["Python is a high-level programming language."]
    ...     )
    ... ]
    >>>
    >>> # Run evaluation
    >>> runner = EvaluationRunner(evaluators)
    >>> report = runner.run(samples)
    >>> print(report.summary)
"""

# Types
from langrag.evaluation.types import (
    EvaluationSample,
    EvaluationResult,
    BatchEvaluationResult,
)

# Base class
from langrag.evaluation.base import BaseEvaluator

# Metrics
from langrag.evaluation.metrics import (
    FaithfulnessEvaluator,
    AnswerRelevancyEvaluator,
    ContextRelevancyEvaluator,
)

# Runner
from langrag.evaluation.runner import (
    EvaluationRunner,
    EvaluationReport,
    EvaluationProgressCallback,
    evaluate_rag,
)

__all__ = [
    # Types
    "EvaluationSample",
    "EvaluationResult",
    "BatchEvaluationResult",
    # Base
    "BaseEvaluator",
    # Metrics
    "FaithfulnessEvaluator",
    "AnswerRelevancyEvaluator",
    "ContextRelevancyEvaluator",
    # Runner
    "EvaluationRunner",
    "EvaluationReport",
    "EvaluationProgressCallback",
    "evaluate_rag",
]
