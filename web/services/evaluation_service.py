
from typing import Dict, Any, List
import logging

from langrag.evaluation.runner import EvaluationRunner, EvaluationReport
from langrag.evaluation.types import EvaluationSample, BatchEvaluationResult
from langrag.evaluation.metrics.faithfulness import FaithfulnessEvaluator
from langrag.evaluation.metrics.answer_relevancy import AnswerRelevancyEvaluator
from langrag.evaluation.metrics.context_relevancy import ContextRelevancyEvaluator
from web.core.rag_kernel import RAGKernel

logger = logging.getLogger(__name__)

class EvaluationService:
    """
    Service for running RAG evaluations.
    """

    def __init__(self, rag_kernel: RAGKernel):
        self.rag_kernel = rag_kernel

    def evaluate_sample(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str | None = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single RAG interaction.
        
        Args:
            question: User question
            answer: RAG system answer
            contexts: List of retrieved context elements (strings)
            ground_truth: Optional correct answer
            
        Returns:
            Dictionary containing scores and reasons for each metric.
            {
                "faithfulness": {"score": 0.9, "reason": "..."},
                "answer_relevancy": {"score": 0.8, "reason": "..."},
                "context_relevancy": {"score": 0.7, "reason": "..."}
            }
        """
        # 1. Check if LLM is available
        if not self.rag_kernel.llm_adapter:
            raise ValueError("LLM is not configured. Cannot perform evaluation.")

        # 2. Initialize Evaluators using the kernel's LLM adapter
        llm = self.rag_kernel.llm_adapter
        evaluators = [
            FaithfulnessEvaluator(llm),
            AnswerRelevancyEvaluator(llm),
            ContextRelevancyEvaluator(llm)
        ]

        # 3. Prepare Sample
        sample = EvaluationSample(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth
        )

        # 4. Run Evaluation
        logger.info(f"Starting evaluation for question: '{question[:30]}...'")
        runner = EvaluationRunner(evaluators=evaluators)
        report: EvaluationReport = runner.run([sample])

        # 5. Format Result
        output = {}
        for metric_name, batch_result in report.metric_results.items():
            if batch_result.results:
                res = batch_result.results[0]
                output[metric_name] = {
                    "score": res.score,
                    "reason": res.reason
                }
            else:
                 output[metric_name] = {
                    "score": 0.0,
                    "reason": "Evaluation failed or returned no result"
                }
        
        return output
