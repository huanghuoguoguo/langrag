import json

from loguru import logger

from langrag.entities.dataset import Dataset
from langrag.llm.base import BaseLLM

from .base import BaseRouter

ROUTER_PROMPT = """
You are a routing agent. You have access to the following datasets:
{datasets_info}

The user's query is: "{query}"

Which datasets should be queried to answer this question?
Respond with a JSON object: {{"dataset_names": ["name1", "name2"]}}
If no specific dataset is clear, return all of them.
"""

class LLMRouter(BaseRouter):
    """
    Router that uses an LLM to decide which datasets to query.
    """

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def route(self, query: str, datasets: list[Dataset]) -> list[Dataset]:
        if not datasets:
            return []

        if len(datasets) == 1:
            return datasets

        # Format info
        info_lines = []
        for d in datasets:
            desc = d.description or "No description"
            info_lines.append(f"- Name: {d.name}, Description: {desc}")

        datasets_info = "\n".join(info_lines)

        prompt = ROUTER_PROMPT.format(datasets_info=datasets_info, query=query)

        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            logger.info(f"[LLMRouter] Raw Response: {response}")

            # Basic parsing of JSON from text
            # Often LLMs wrap in ```json ... ```
            content = response.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            logger.info(f"[LLMRouter] Parsed JSON Content: {content}")
            data = json.loads(content)
            names = data.get("dataset_names", [])

            selected = [d for d in datasets if d.name in names]

            if not selected:
                logger.warning("Router selected no datasets, falling back to all.")
                return datasets

            return selected

        except Exception as e:
            logger.error(f"Router failed: {e}. Falling back to all datasets.")
            return datasets
