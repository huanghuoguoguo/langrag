"""
Context Relevancy evaluator for RAG systems.

This metric measures how relevant the retrieved context passages are to the
user's question. High context relevancy indicates the retrieval system found
appropriate information, while low relevancy suggests retrieval issues.
"""

from langrag.evaluation.base import BaseEvaluator
from langrag.evaluation.types import EvaluationSample, EvaluationResult

CONTEXT_RELEVANCY_PROMPT = """You are an expert evaluator assessing the relevancy of retrieved context.

Context Relevancy measures how relevant and useful the retrieved context passages are
for answering the given question. Good context should contain information directly
related to answering the question.

## Question:
{question}

## Retrieved Context:
{context}

## Evaluation Criteria:
1. Relevance: Does the context contain information related to the question?
2. Usefulness: Could this context help answer the question?
3. Noise: How much irrelevant information is included?
4. Coverage: Does the context cover the key aspects of the question?

## Response Format:
Provide your evaluation in this exact format:
Reason: <Brief explanation of context relevancy>
Score: <0.0 to 1.0>

Scoring guide:
- 1.0: Perfect - context directly addresses the question with no noise
- 0.7-0.9: Good - context is relevant with minimal noise
- 0.4-0.6: Moderate - context is partially relevant or has significant noise
- 0.1-0.3: Poor - context is mostly irrelevant with few useful parts
- 0.0: None - context is completely irrelevant to the question
"""


class ContextRelevancyEvaluator(BaseEvaluator):
    """
    Evaluator for measuring context relevancy to the question.

    Context Relevancy assesses the quality of the retrieval step by measuring
    how relevant the retrieved passages are to answering the user's question.
    This metric helps identify retrieval issues separate from generation issues.

    This metric is useful for:
    - Diagnosing retrieval vs generation problems
    - Tuning retrieval parameters (k, similarity threshold)
    - Comparing different retrieval strategies

    Score interpretation:
    - 1.0: Retrieved context perfectly matches the question's information need
    - 0.5: Context is partially relevant, contains some noise
    - 0.0: Context is completely irrelevant

    Example:
        >>> evaluator = ContextRelevancyEvaluator(llm)
        >>> sample = EvaluationSample(
        ...     question="What is the boiling point of water?",
        ...     answer="100°C at standard pressure",
        ...     contexts=["Water boils at 100°C (212°F) at standard pressure."]
        ... )
        >>> result = evaluator.evaluate(sample)
        >>> print(f"Context Relevancy: {result.score:.2f}")
    """

    @property
    def name(self) -> str:
        """Return metric name."""
        return "context_relevancy"

    def evaluate(self, sample: EvaluationSample) -> EvaluationResult:
        """
        Evaluate the relevancy of retrieved context to the question.

        Args:
            sample: Evaluation sample with question and contexts

        Returns:
            EvaluationResult with context relevancy score
        """
        # Combine contexts into a single string
        context_text = "\n\n".join(
            f"[Context {i+1}]: {ctx}"
            for i, ctx in enumerate(sample.contexts)
        ) if sample.contexts else "[No context provided]"

        prompt = CONTEXT_RELEVANCY_PROMPT.format(
            question=sample.question,
            context=context_text
        )

        response = self._call_llm(prompt)
        score, reason = self._parse_score_from_response(response)

        return EvaluationResult(
            score=score,
            metric_name=self.name,
            reason=reason,
            details={
                "context_count": len(sample.contexts),
                "total_context_length": sum(len(c) for c in sample.contexts)
            }
        )
