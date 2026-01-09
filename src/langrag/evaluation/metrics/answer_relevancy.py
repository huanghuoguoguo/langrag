"""
Answer Relevancy evaluator for RAG systems.

This metric measures how well the generated answer addresses the user's question.
An answer with high relevancy directly and completely answers what was asked,
while low relevancy indicates the answer is off-topic or incomplete.
"""

from langrag.evaluation.base import BaseEvaluator
from langrag.evaluation.types import EvaluationSample, EvaluationResult

ANSWER_RELEVANCY_PROMPT = """You are an expert evaluator assessing the relevancy of a generated answer.

Answer Relevancy measures how well the answer addresses the user's question.
A relevant answer directly responds to what was asked, is complete, and stays on topic.

## Question:
{question}

## Answer to Evaluate:
{answer}

## Evaluation Criteria:
1. Directness: Does the answer directly address the question?
2. Completeness: Does the answer fully address all parts of the question?
3. Focus: Does the answer stay on topic without unnecessary tangents?
4. Clarity: Is the answer clear and understandable?

## Response Format:
Provide your evaluation in this exact format:
Reason: <Brief explanation of relevancy assessment>
Score: <0.0 to 1.0>

Scoring guide:
- 1.0: Perfect relevancy - directly and completely answers the question
- 0.7-0.9: Good relevancy - answers the question but may miss minor aspects
- 0.4-0.6: Partial relevancy - partially addresses the question
- 0.1-0.3: Low relevancy - touches on the topic but doesn't answer the question
- 0.0: No relevancy - completely off-topic or refuses to answer
"""


class AnswerRelevancyEvaluator(BaseEvaluator):
    """
    Evaluator for measuring answer relevancy to the question.

    Answer Relevancy measures how well the generated answer addresses the
    user's question, independent of factual correctness. This metric focuses
    on whether the answer is on-topic and responsive.

    This metric complements Faithfulness:
    - Faithfulness: Is the answer factually grounded in context?
    - Answer Relevancy: Does the answer address the question asked?

    A good RAG answer should score high on both metrics.

    Score interpretation:
    - 1.0: Answer perfectly addresses the question
    - 0.5: Answer partially addresses the question
    - 0.0: Answer is completely off-topic

    Example:
        >>> evaluator = AnswerRelevancyEvaluator(llm)
        >>> sample = EvaluationSample(
        ...     question="What is the capital of France?",
        ...     answer="Paris is the capital and largest city of France.",
        ...     contexts=["Paris is the capital of France."]
        ... )
        >>> result = evaluator.evaluate(sample)
        >>> print(f"Answer Relevancy: {result.score:.2f}")
    """

    @property
    def name(self) -> str:
        """Return metric name."""
        return "answer_relevancy"

    def evaluate(self, sample: EvaluationSample) -> EvaluationResult:
        """
        Evaluate the relevancy of an answer to the question.

        Args:
            sample: Evaluation sample with question and answer

        Returns:
            EvaluationResult with relevancy score
        """
        prompt = ANSWER_RELEVANCY_PROMPT.format(
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
                "question_length": len(sample.question),
                "answer_length": len(sample.answer)
            }
        )
