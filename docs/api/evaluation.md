# Evaluation Metrics

LLM-as-a-Judge evaluation metrics for RAG quality assessment.

## FaithfulnessEvaluator

Measures whether the answer is grounded in the retrieved context.

```python
from langrag import FaithfulnessEvaluator

evaluator = FaithfulnessEvaluator(llm)
result = evaluator.evaluate(sample)

print(f"Faithfulness: {result.score:.2f}")
print(f"Reason: {result.reason}")
```

### Score Interpretation

| Score | Meaning |
|-------|---------|
| 1.0 | All claims supported by context |
| 0.7-0.9 | Most claims supported |
| 0.4-0.6 | Some claims unsupported |
| 0.0-0.3 | Significant hallucination |

## AnswerRelevancyEvaluator

Measures how well the answer addresses the question.

```python
from langrag import AnswerRelevancyEvaluator

evaluator = AnswerRelevancyEvaluator(llm)
result = evaluator.evaluate(sample)
```

### Score Interpretation

| Score | Meaning |
|-------|---------|
| 1.0 | Directly and completely answers the question |
| 0.7-0.9 | Good answer with minor gaps |
| 0.4-0.6 | Partially addresses the question |
| 0.0-0.3 | Off-topic or refuses to answer |

## ContextRelevancyEvaluator

Measures how relevant the retrieved context is to the question.

```python
from langrag import ContextRelevancyEvaluator

evaluator = ContextRelevancyEvaluator(llm)
result = evaluator.evaluate(sample)
```

### Score Interpretation

| Score | Meaning |
|-------|---------|
| 1.0 | Context perfectly matches information need |
| 0.7-0.9 | Good context with minimal noise |
| 0.4-0.6 | Partially relevant context |
| 0.0-0.3 | Mostly irrelevant context |

## BaseEvaluator

Abstract base class for custom evaluators.

```python
from langrag import BaseEvaluator, EvaluationSample, EvaluationResult

class CustomEvaluator(BaseEvaluator):
    @property
    def name(self) -> str:
        return "custom_metric"

    def evaluate(self, sample: EvaluationSample) -> EvaluationResult:
        # Your evaluation logic
        prompt = f"Evaluate: {sample.answer}"
        response = self._call_llm(prompt)
        score, reason = self._parse_score_from_response(response)

        return EvaluationResult(
            score=score,
            metric_name=self.name,
            reason=reason
        )
```

## EvaluationSample

Input data for evaluation.

```python
from langrag import EvaluationSample

sample = EvaluationSample(
    question="What is Python?",
    answer="Python is a programming language.",
    contexts=["Python is a high-level programming language."],
    ground_truth="Python is a programming language.",  # Optional
    metadata={"source": "test"}  # Optional
)
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `question` | str | The user's question |
| `answer` | str | Generated answer |
| `contexts` | list[str] | Retrieved context passages |
| `ground_truth` | str \| None | Reference answer (optional) |
| `metadata` | dict | Additional metadata |

## EvaluationResult

Output from evaluation.

```python
result = evaluator.evaluate(sample)

print(result.score)        # 0.85
print(result.metric_name)  # "faithfulness"
print(result.reason)       # "The answer is grounded in..."
print(result.details)      # {"context_count": 3}
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `score` | float | Score from 0.0 to 1.0 |
| `metric_name` | str | Name of the metric |
| `reason` | str \| None | Explanation for score |
| `details` | dict | Additional details |
