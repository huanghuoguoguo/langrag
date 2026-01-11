import json
import re

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
If no dataset is relevant to the question, return an empty list: [].
Do NOT select datasets unless they are clearly relevant.
"""

# Regex to extract JSON object from LLM response
JSON_PATTERN = re.compile(r'\{[^{}]*"dataset_names"\s*:\s*\[[^\]]*\][^{}]*\}', re.DOTALL)


class LLMRouter(BaseRouter):
    """
    Router that uses an LLM to decide which datasets to query.
    """

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def _extract_json(self, response: str) -> dict | None:
        """
        Extract JSON object from LLM response.

        Handles various formats:
        - Plain JSON: {"dataset_names": [...]}
        - Markdown code block: ```json {"dataset_names": [...]} ```
        - Text with embedded JSON

        Args:
            response: Raw LLM response text

        Returns:
            Parsed JSON dict or None if parsing fails
        """
        content = response.strip()

        # Try to extract from markdown code block first
        if "```" in content:
            # Extract content between code fences
            parts = content.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:  # Odd indices are inside code blocks
                    # Remove language identifier if present (e.g., "json\n")
                    code_content = part.strip()
                    if code_content.startswith("json"):
                        code_content = code_content[4:].strip()
                    try:
                        return json.loads(code_content)
                    except json.JSONDecodeError:
                        continue

        # Try direct JSON parsing
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try regex extraction for embedded JSON
        match = JSON_PATTERN.search(content)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return None

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
            logger.debug(f"[LLMRouter] Raw Response: {response}")

            # Parse JSON from response
            data = self._extract_json(response)

            if data is None:
                logger.warning(
                    f"[LLMRouter] Failed to parse JSON from response: {response[:200]}..."
                )
                return datasets

            names = data.get("dataset_names", [])

            if not isinstance(names, list):
                logger.warning(
                    f"[LLMRouter] 'dataset_names' is not a list: {type(names)}"
                )
                return datasets

            # Note: We respect empty list if LLM explicitly decides no KB is relevant.
            selected = [d for d in datasets if d.name in names]

            if not selected and names:
                # If names were provided but none matched (e.g. hallucinated names), warning.
                # But if names was [], selected is [], which is valid.
                logger.warning(f"Router returned names {names} but none matched available datasets.")
                # In this specific case, maybe we should return [] or all? 
                # Let's return [] because the intent was likely specific.
                return []
            
            logger.info(f"[LLMRouter] Selected datasets: {[d.name for d in selected]}")
            return selected

        except json.JSONDecodeError as e:
            logger.error(f"[LLMRouter] JSON parsing error: {e}. Falling back to all datasets.")
            return datasets
        except Exception as e:
            logger.error(
                f"[LLMRouter] Unexpected error: {type(e).__name__}: {e}. "
                "Falling back to all datasets."
            )
            return datasets
