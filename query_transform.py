from typing import List
from loguru import logger

class QueryTransformer:
    """
    Handles prompt generation for query expansion.
    Centralized here so prompts are easy to manage.
    """
    
    @staticmethod
    def get_hyde_prompt(query: str) -> str:
        return f"""Write a short paragraph that answers the following question. 
It doesn't have to be perfectly accurate, but it should look like a passage from a textbook or documentation.

Question: {query}

Hypothetical Answer:"""

    @staticmethod
    def get_decomposition_prompt(query: str) -> str:
        return f"""Break down the following complex question into 2-3 simpler sub-questions 
that would help in answering the original question step-by-step. 
Provide only the sub-questions, one per line.

Original Question: {query}

Sub-questions:"""

    @staticmethod
    def parse_decomposition(response_text: str) -> List[str]:
        return [line.strip("- ").strip() for line in response_text.strip().split("\n") if line.strip()]
