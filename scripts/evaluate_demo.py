
import os
import sys
from pathlib import Path
from typing import Any, List

# Add src to python path
sys.path.append(str(Path(__file__).parents[1] / "src"))

from openai import OpenAI
from langrag.llm.base import BaseLLM
from langrag.evaluation.types import EvaluationSample
from langrag.evaluation.metrics.faithfulness import FaithfulnessEvaluator
from langrag.evaluation.metrics.answer_relevancy import AnswerRelevancyEvaluator
from langrag.evaluation.metrics.context_relevancy import ContextRelevancyEvaluator
from langrag.evaluation.runner import EvaluationRunner

class KimiLLM(BaseLLM):
    """
    Adapter for Kimi (Moonshot AI) using OpenAI SDK.
    """
    def __init__(self, api_key: str, model: str = "moonshot-v1-8k"):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1",
        )
        self.model = model

    def chat(self, messages: list[dict], **kwargs) -> str:
        # Map common params if needed, but Kimi is OpenAI compatible
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return completion.choices[0].message.content

    # Implement abstract methods with dummies
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return []

    def embed_query(self, text: str) -> list[float]:
        return []

    def stream_chat(self, messages: list[dict], **kwargs):
        yield "Not implemented"

def main():
    # 1. Configuration
    KIMI_API_KEY = "sk-81hrUrV0tMYJWa6KioTYay7kHyMYt0LIcdS76QE0bqE9hgRb"

    if not KIMI_API_KEY:
        print("Please provide a valid API Key")
        return

    print(f"Initializing Kimi LLM Judge...")
    llm = KimiLLM(api_key=KIMI_API_KEY)

    # 2. Prepare Evaluators
    evaluators = [
        FaithfulnessEvaluator(llm),
        AnswerRelevancyEvaluator(llm),
        ContextRelevancyEvaluator(llm)
    ]
    
    runner = EvaluationRunner(evaluators=evaluators)

    # 3. Create Sample Data
    # Simulate a RAG q&a pair
    sample = EvaluationSample(
        question="LangRAG 的主要特点是什么？",
        answer="LangRAG 采用了模块化设计，支持 Agentic RAG 工作流，并且具有高性能的语义缓存功能。",
        contexts=[
            "LangRAG 是一个高性能的 RAG 框架。",
            "它采用了完全的模块化设计，允许用户自定义 pipeline。",
            "最新版本引入了语义缓存 (Semantic Cache) 以减少重复查询开销。",
            "支持 Agentic 工作流，包含 Router 和 Rewriter 组件。"
        ]
    )

    print("\nProcessing sample evaluation...")
    print("-" * 50)
    print(f"Question: {sample.question}")
    print(f"Answer: {sample.answer}")
    print("-" * 50)

    # 4. Run Evaluation
    report = runner.run([sample])

    # 5. Display Results
    for metric_name, batch_result in report.metric_results.items():
        if batch_result.results:
            result = batch_result.results[0]
            print(f"\nMetric: {metric_name}")
            print(f"Score:  {result.score:.2f}")
            print(f"Reason: {result.reason}")
        else:
             print(f"\nMetric: {metric_name} - No result")

if __name__ == "__main__":
    main()
