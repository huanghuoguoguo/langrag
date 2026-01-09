# Evaluation Guide

This guide covers how to evaluate RAG system quality using LangRAG's evaluation framework.

## Overview

LangRAG provides an LLM-as-a-Judge evaluation framework based on the RAGAS methodology. It measures:

- **Faithfulness**: Is the answer grounded in the retrieved context?
- **Answer Relevancy**: Does the answer address the question?
- **Context Relevancy**: Is the retrieved context relevant to the question?

## Quick Start

```python
from langrag import (
    EvaluationRunner,
    EvaluationSample,
    FaithfulnessEvaluator,
    AnswerRelevancyEvaluator,
    ContextRelevancyEvaluator,
)

# Create evaluation samples
samples = [
    EvaluationSample(
        question="What is Python?",
        answer="Python is a high-level programming language.",
        contexts=["Python is a high-level, general-purpose programming language."]
    )
]

# Create evaluators with an LLM
evaluators = [
    FaithfulnessEvaluator(llm),
    AnswerRelevancyEvaluator(llm),
    ContextRelevancyEvaluator(llm),
]

# Run evaluation
runner = EvaluationRunner(evaluators)
report = runner.run(samples)

# View results
print(report.summary)
```

## Evaluation Metrics

### Faithfulness

Measures whether all claims in the answer are supported by the context.

```python
from langrag import FaithfulnessEvaluator

evaluator = FaithfulnessEvaluator(llm)
result = evaluator.evaluate(sample)

print(f"Score: {result.score}")   # 0.0 - 1.0
print(f"Reason: {result.reason}") # Explanation
```

**Score Interpretation:**

| Score | Meaning |
|-------|---------|
| 1.0 | All claims are supported by context |
| 0.7-0.9 | Most claims supported, minor issues |
| 0.4-0.6 | Some claims unsupported |
| 0.0-0.3 | Significant hallucination |

**Use Cases:**

- Detecting hallucination
- Ensuring factual accuracy
- Validating RAG responses

### Answer Relevancy

Measures how well the answer addresses the question.

```python
from langrag import AnswerRelevancyEvaluator

evaluator = AnswerRelevancyEvaluator(llm)
result = evaluator.evaluate(sample)
```

**Score Interpretation:**

| Score | Meaning |
|-------|---------|
| 1.0 | Directly and completely answers the question |
| 0.7-0.9 | Good answer with minor gaps |
| 0.4-0.6 | Partially addresses the question |
| 0.0-0.3 | Off-topic or refuses to answer |

**Use Cases:**

- Measuring response quality
- Identifying off-topic responses
- Evaluating LLM understanding

### Context Relevancy

Measures how relevant the retrieved context is to the question.

```python
from langrag import ContextRelevancyEvaluator

evaluator = ContextRelevancyEvaluator(llm)
result = evaluator.evaluate(sample)
```

**Score Interpretation:**

| Score | Meaning |
|-------|---------|
| 1.0 | Context perfectly matches information need |
| 0.7-0.9 | Good context with minimal noise |
| 0.4-0.6 | Partially relevant context |
| 0.0-0.3 | Mostly irrelevant context |

**Use Cases:**

- Evaluating retrieval quality
- Tuning retrieval parameters
- Diagnosing retrieval vs generation issues

## Evaluation Samples

### Creating Samples

```python
from langrag import EvaluationSample

sample = EvaluationSample(
    question="What is the capital of France?",
    answer="The capital of France is Paris.",
    contexts=[
        "Paris is the capital and largest city of France.",
        "France is a country in Western Europe."
    ],
    ground_truth="Paris",  # Optional reference answer
    metadata={"source": "test_set_1"}  # Optional metadata
)
```

### Creating from RAG Results

```python
# After running RAG search
results = workflow.search(query, top_k=5)

sample = EvaluationSample(
    question=query,
    answer=generated_answer,
    contexts=[r.content for r in results]
)
```

### Batch Sample Creation

```python
samples = []
for item in test_dataset:
    # Run your RAG pipeline
    results = workflow.search(item["question"])
    answer = generate_answer(results)

    samples.append(EvaluationSample(
        question=item["question"],
        answer=answer,
        contexts=[r.content for r in results],
        ground_truth=item.get("answer")
    ))
```

## Running Evaluations

### Single Metric

```python
evaluator = FaithfulnessEvaluator(llm)
result = evaluator.evaluate(sample)
```

### Multiple Metrics

```python
runner = EvaluationRunner([
    FaithfulnessEvaluator(llm),
    AnswerRelevancyEvaluator(llm),
    ContextRelevancyEvaluator(llm),
])

report = runner.run(samples)
```

### With Progress Tracking

```python
def on_progress(metric_name, current, total):
    print(f"[{metric_name}] {current}/{total}")

report = runner.run(samples, on_progress=on_progress)
```

### Error Handling

```python
# Continue evaluation even if some samples fail
runner = EvaluationRunner(
    evaluators,
    continue_on_error=True
)

report = runner.run(samples)
print(f"Errors: {report.metric_results['faithfulness'].error_count}")
```

## Evaluation Report

### Accessing Results

```python
report = runner.run(samples)

# Summary statistics
print(report.summary)
# {
#     'faithfulness': {'mean': 0.85, 'sample_count': 100, 'error_count': 0},
#     'answer_relevancy': {'mean': 0.92, ...},
#     'context_relevancy': {'mean': 0.78, ...}
# }

# Individual results
for result in report.metric_results['faithfulness'].results:
    print(f"Score: {result.score}, Reason: {result.reason}")
```

### Exporting Results

```python
# To dictionary
data = report.to_dict()

# To JSON
import json
json.dump(data, open("evaluation_results.json", "w"))

# To pandas DataFrame
import pandas as pd
df = pd.DataFrame([
    {
        "metric": metric,
        "score": r.score,
        "reason": r.reason
    }
    for metric, batch in report.metric_results.items()
    for r in batch.results
])
```

## Best Practices

### 1. Use Representative Samples

```python
# Include diverse question types
samples = [
    # Factual questions
    EvaluationSample(question="What year was Python created?", ...),
    # Conceptual questions
    EvaluationSample(question="How does Python handle memory?", ...),
    # Comparative questions
    EvaluationSample(question="How is Python different from Java?", ...),
]
```

### 2. Include Ground Truth When Available

```python
sample = EvaluationSample(
    question="What is 2+2?",
    answer=rag_answer,
    contexts=contexts,
    ground_truth="4"  # Enables additional validation
)
```

### 3. Run Multiple Metrics

```python
# Single metric can be misleading
# Always evaluate multiple aspects
runner = EvaluationRunner([
    FaithfulnessEvaluator(llm),      # Factual accuracy
    AnswerRelevancyEvaluator(llm),   # Response quality
    ContextRelevancyEvaluator(llm),  # Retrieval quality
])
```

### 4. Use Consistent LLM Settings

```python
# Use low temperature for consistent evaluation
evaluator = FaithfulnessEvaluator(
    llm,
    temperature=0.0  # Deterministic
)
```

### 5. Evaluate Regularly

```python
# Set up periodic evaluation
def weekly_evaluation():
    samples = generate_test_samples()
    report = runner.run(samples)

    # Alert on quality drops
    if report.summary['faithfulness']['mean'] < 0.8:
        alert("Faithfulness dropped below threshold!")
```

## Custom Evaluators

Create custom metrics by extending `BaseEvaluator`:

```python
from langrag import BaseEvaluator, EvaluationSample, EvaluationResult

class CompletenessEvaluator(BaseEvaluator):
    @property
    def name(self) -> str:
        return "completeness"

    def evaluate(self, sample: EvaluationSample) -> EvaluationResult:
        prompt = f"""
        Question: {sample.question}
        Answer: {sample.answer}

        Rate how complete the answer is (0.0 to 1.0).
        Score:
        """

        response = self._call_llm(prompt)
        score, reason = self._parse_score_from_response(response)

        return EvaluationResult(
            score=score,
            metric_name=self.name,
            reason=reason
        )
```

## Troubleshooting

### Low Faithfulness Scores

- Check if context contains relevant information
- Verify answer generation prompt encourages using context
- Consider adding "cite your sources" instruction

### Low Answer Relevancy

- Check if question is clear
- Verify LLM understands the domain
- Consider query rewriting

### Low Context Relevancy

- Adjust retrieval `top_k`
- Try hybrid search
- Improve chunking strategy
- Fine-tune embeddings
