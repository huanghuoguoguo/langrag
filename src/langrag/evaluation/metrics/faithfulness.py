"""
Faithfulness evaluator for RAG systems.

This metric measures whether the generated answer is grounded in (faithful to)
the retrieved context. A faithful answer only contains information that can be
supported by the provided context passages.

Faithfulness is critical for RAG systems to prevent hallucination and ensure
that answers are backed by retrieved evidence.
"""

from langrag.evaluation.base import BaseEvaluator
from langrag.evaluation.types import EvaluationSample, EvaluationResult

FAITHFULNESS_PROMPT = """You are an expert evaluator assessing the faithfulness of a generated answer.

Faithfulness measures whether all claims in the answer can be supported by the given context.
A faithful answer only contains information that is directly stated or can be logically inferred from the context.

## Context:
{context}

## Question:
{question}

## Answer to Evaluate:
{answer}

## Evaluation Instructions:
1. Identify all factual claims made in the answer
2. For each claim, check if it is supported by the context
3. Calculate the faithfulness score as: (supported claims) / (total claims)

## Response Format:
Provide your evaluation in this exact format:
Reason: <Brief explanation of which claims are supported/unsupported>
Score: <0.0 to 1.0>

If the answer makes no factual claims (e.g., "I don't know"), give a score of 1.0.
If the answer is completely faithful to the context, give a score of 1.0.
If the answer contains hallucinated information not in the context, reduce the score proportionally.
"""


class FaithfulnessEvaluator(BaseEvaluator):
    """
    Evaluator for measuring answer faithfulness to retrieved context.

    Faithfulness (also called "groundedness") measures whether the generated
    answer is supported by the retrieved context. This is one of the most
    important metrics for RAG systems as it directly measures hallucination.

    Score interpretation:
    - 1.0: All claims in the answer are supported by the context
    - 0.5: Half of the claims are supported
    - 0.0: No claims are supported (complete hallucination)

    Example:
        >>> evaluator = FaithfulnessEvaluator(llm)
        >>> sample = EvaluationSample(
        ...     question="What color is the sky?",
        ...     answer="The sky is blue during the day.",
        ...     contexts=["The sky appears blue due to Rayleigh scattering."]
        ... )
        >>> result = evaluator.evaluate(sample)
        >>> print(f"Faithfulness: {result.score:.2f}")
    """

    @property
    def name(self) -> str:
        """Return metric name."""
        return "faithfulness"

    def evaluate(self, sample: EvaluationSample) -> EvaluationResult:
        """
        Evaluate the faithfulness of an answer.

        Args:
            sample: Evaluation sample with question, answer, and contexts

        Returns:
            EvaluationResult with faithfulness score
        """
        # Combine contexts into a single string
        context_text = "\n\n".join(
            f"[Context {i+1}]: {ctx}"
            for i, ctx in enumerate(sample.contexts)
        ) if sample.contexts else "[No context provided]"

        prompt = FAITHFULNESS_PROMPT.format(
            context=context_text,
            question=sample.question,
            answer=sample.answer
        )

        response = self._call_llm(prompt)
        score, reason = self._parse_score_from_response(response)

        return EvaluationResult(
            score=score,
            metric_name=self.name,
            reason=reason,
            details={
                "context_count": len(sample.contexts),
                "answer_length": len(sample.answer)
            }
        )
