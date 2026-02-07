# Evaluation Layer Design

## 1. 职责 (Responsibilities)

本模块负责 **RAG 系统评估**，使用 LLM-as-a-Judge 方法评估检索和生成质量。

**核心目标：**
- **质量度量**: 评估 RAG 系统的检索和生成质量
- **多维评估**: 支持 Faithfulness、Relevancy 等多种评估维度
- **自动化**: 使用 LLM 作为评判者，减少人工评估成本

## 2. 模块结构

```text
evaluation/
├── base.py                     # BaseEvaluator 基类
├── types.py                    # EvaluationSample, EvaluationResult 类型定义
├── runner.py                   # EvaluationRunner 批量评估执行器
├── metrics/                    # 具体评估指标
│   ├── faithfulness.py         # 忠实度评估
│   ├── answer_relevancy.py     # 答案相关性评估
│   ├── context_relevancy.py    # 上下文相关性评估
│   └── __init__.py
└── __init__.py
```

## 3. 核心概念

### 3.1 LLM-as-a-Judge

本模块采用 LLM-as-a-Judge 评估范式：

```mermaid
graph LR
    Sample[评估样本] -->|1. 构造Prompt| Prompt[评估提示]
    Prompt -->|2. 调用LLM| LLM[(评判LLM)]
    LLM -->|3. 解析响应| Result[评估结果]
```

### 3.2 数据类型

#### EvaluationSample

评估样本，包含评估所需的所有信息：

```python
@dataclass
class EvaluationSample:
    question: str               # 用户问题
    answer: str                 # RAG 系统生成的答案
    contexts: list[str]         # 检索到的上下文
    ground_truth: str | None    # 可选的标准答案
    metadata: dict[str, Any]    # 可选元数据
```

#### EvaluationResult

单个样本的评估结果：

```python
@dataclass
class EvaluationResult:
    score: float                # 评估分数 (0.0-1.0)
    metric_name: str            # 指标名称
    reason: str | None          # 评估理由
    details: dict[str, Any]     # 额外细节
```

#### BatchEvaluationResult

批量评估结果，包含统计信息：

```python
@dataclass
class BatchEvaluationResult:
    results: list[EvaluationResult]
    metric_name: str
    mean_score: float           # 平均分
    sample_count: int           # 样本数
    error_count: int            # 失败数
```

## 4. 评估指标

### 4.1 Faithfulness (忠实度) ✅ 已实现

评估答案是否忠实于检索到的上下文，检测幻觉。

| 分数 | 含义 |
|------|------|
| 1.0 | 所有声明都有上下文支持 |
| 0.5 | 一半的声明有支持 |
| 0.0 | 完全幻觉 |

```python
evaluator = FaithfulnessEvaluator(llm)
result = evaluator.evaluate(sample)
```

### 4.2 Answer Relevancy (答案相关性) ✅ 已实现

评估答案与问题的相关程度。

| 分数 | 含义 |
|------|------|
| 1.0 | 完全回答了问题 |
| 0.5 | 部分相关 |
| 0.0 | 完全无关 |

### 4.3 Context Relevancy (上下文相关性) ✅ 已实现

评估检索到的上下文与问题的相关程度。

| 分数 | 含义 |
|------|------|
| 1.0 | 所有上下文都相关 |
| 0.5 | 一半相关 |
| 0.0 | 所有上下文都不相关 |

## 5. 使用方式

### 5.1 单指标评估

```python
from langrag.evaluation import FaithfulnessEvaluator, EvaluationSample

# 初始化评估器
evaluator = FaithfulnessEvaluator(llm=your_llm, temperature=0.0)

# 准备样本
sample = EvaluationSample(
    question="什么是机器学习？",
    answer="机器学习是一种人工智能技术，让计算机能够从数据中学习。",
    contexts=["机器学习是人工智能的一个分支，使计算机系统能够从数据中学习和改进。"]
)

# 评估
result = evaluator.evaluate(sample)
print(f"Faithfulness: {result.score:.2f}")
print(f"Reason: {result.reason}")
```

### 5.2 批量评估

```python
samples = [sample1, sample2, sample3, ...]
batch_result = evaluator.evaluate_batch(samples, continue_on_error=True)

print(f"Mean Score: {batch_result.mean_score:.2f}")
print(f"Samples: {batch_result.sample_count}")
print(f"Errors: {batch_result.error_count}")
```

### 5.3 多指标评估

```python
from langrag.evaluation import (
    FaithfulnessEvaluator,
    AnswerRelevancyEvaluator,
    ContextRelevancyEvaluator,
)

evaluators = [
    FaithfulnessEvaluator(llm),
    AnswerRelevancyEvaluator(llm),
    ContextRelevancyEvaluator(llm),
]

for evaluator in evaluators:
    result = evaluator.evaluate(sample)
    print(f"{evaluator.name}: {result.score:.2f}")
```

## 6. BaseEvaluator 接口

```python
class BaseEvaluator(ABC):
    """Abstract base class for RAG evaluation metrics."""

    def __init__(self, llm: BaseLLM, temperature: float = 0.0):
        self.llm = llm
        self.temperature = temperature

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this evaluation metric."""
        pass

    @abstractmethod
    def evaluate(self, sample: EvaluationSample) -> EvaluationResult:
        """Evaluate a single sample."""
        pass

    def evaluate_batch(
        self,
        samples: list[EvaluationSample],
        continue_on_error: bool = True
    ) -> BatchEvaluationResult:
        """Evaluate multiple samples."""
        pass

    # 辅助方法
    def _call_llm(self, prompt: str) -> str: ...
    def _parse_score_from_response(self, response: str) -> tuple[float, str | None]: ...
```

## 7. 扩展性

### 7.1 添加新指标

```python
class HallucinationEvaluator(BaseEvaluator):
    """Evaluator specifically for detecting hallucination."""

    @property
    def name(self) -> str:
        return "hallucination"

    def evaluate(self, sample: EvaluationSample) -> EvaluationResult:
        prompt = self._build_prompt(sample)
        response = self._call_llm(prompt)
        score, reason = self._parse_score_from_response(response)
        return EvaluationResult(
            score=score,
            metric_name=self.name,
            reason=reason
        )
```

### 7.2 未来可扩展的指标

| 指标 | 描述 | 状态 |
|------|------|------|
| Answer Correctness | 答案正确性 (需要 ground_truth) | 待实现 |
| Semantic Similarity | 语义相似度 | 待实现 |
| Coherence | 连贯性 | 待实现 |
| Fluency | 流畅度 | 待实现 |

### 7.3 Async 支持 (待实现)

当前评估方法是同步的，未来可添加异步版本：

```python
# 建议添加
async def evaluate_async(self, sample: EvaluationSample) -> EvaluationResult:
    """Async version of evaluate."""
    pass

async def evaluate_batch_async(
    self,
    samples: list[EvaluationSample]
) -> BatchEvaluationResult:
    """Parallel async batch evaluation."""
    pass
```

## 8. 最佳实践

### 8.1 评估 LLM 选择

| 场景 | 建议模型 | 说明 |
|------|----------|------|
| 开发测试 | gpt-4o-mini | 成本低，速度快 |
| 生产评估 | gpt-4o / claude-3 | 准确性更高 |
| 大规模评估 | gpt-4o-mini + 采样 | 平衡成本和准确性 |

### 8.2 评估数据准备

```python
# 推荐：从实际用户查询中采样
samples = [
    EvaluationSample(
        question=query.question,
        answer=rag_response.answer,
        contexts=rag_response.retrieved_contexts,
        ground_truth=annotated_answer  # 如有
    )
    for query, rag_response, annotated_answer in test_dataset
]
```

### 8.3 解读评估结果

| 指标组合 | 诊断 | 建议 |
|----------|------|------|
| 高 Faithfulness + 低 Answer Relevancy | 答案虽然准确但跑题 | 优化 Prompt |
| 低 Faithfulness + 高 Answer Relevancy | 存在幻觉 | 加强上下文引用 |
| 低 Context Relevancy | 检索质量差 | 优化索引/检索 |

---

*Last updated: 2026-02-07*
