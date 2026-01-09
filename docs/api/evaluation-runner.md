# Evaluation Runner

Orchestrates running multiple evaluation metrics across sample datasets.

## EvaluationRunner

```python
from langrag import (
    EvaluationRunner,
    FaithfulnessEvaluator,
    AnswerRelevancyEvaluator,
    ContextRelevancyEvaluator,
)

runner = EvaluationRunner([
    FaithfulnessEvaluator(llm),
    AnswerRelevancyEvaluator(llm),
    ContextRelevancyEvaluator(llm),
])

report = runner.run(samples)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `evaluators` | list[BaseEvaluator] | required | List of evaluators |
| `continue_on_error` | bool | True | Continue on failures |

### Methods

#### run()

```python
report = runner.run(
    samples: list[EvaluationSample],
    on_progress: Callable[[str, int, int], None] = None
)
```

#### run_single_metric()

```python
result = runner.run_single_metric(
    evaluator: BaseEvaluator,
    samples: list[EvaluationSample]
)
```

## EvaluationReport

Complete evaluation report with results from all metrics.

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
    print(f"Score: {result.score}")

# Export to dict
data = report.to_dict()
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `metric_results` | dict | Results per metric |
| `sample_count` | int | Number of samples |
| `summary` | dict | Summary statistics |

## evaluate_rag()

Convenience function for quick evaluation.

```python
from langrag import evaluate_rag, FaithfulnessEvaluator

report = evaluate_rag(
    samples=samples,
    evaluators=[FaithfulnessEvaluator(llm)]
)
```

## Example

```python
from langrag import (
    EvaluationRunner,
    EvaluationSample,
    FaithfulnessEvaluator,
    AnswerRelevancyEvaluator,
    ContextRelevancyEvaluator,
)

# Create samples
samples = [
    EvaluationSample(
        question="What is Python?",
        answer="Python is a programming language.",
        contexts=["Python is a high-level programming language."]
    ),
    # ... more samples
]

# Create runner
runner = EvaluationRunner([
    FaithfulnessEvaluator(llm),
    AnswerRelevancyEvaluator(llm),
    ContextRelevancyEvaluator(llm),
])

# Run with progress tracking
def on_progress(metric_name, current, total):
    print(f"[{metric_name}] {current}/{total}")

report = runner.run(samples, on_progress=on_progress)

# Print results
for metric, stats in report.summary.items():
    print(f"{metric}: {stats['mean']:.2f} (n={stats['sample_count']})")
```
