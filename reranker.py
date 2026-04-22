import json
from typing import List
from google import genai
from langchain_core.documents import Document
from loguru import logger
from config import settings

class GeminiReranker:
    """
    Cloud-based Reranker using Gemini 1.5 Flash.
    Provides high-accuracy document ranking without local RAM overhead.
    """
    def __init__(self):
        self.client = genai.Client(api_key=settings.google_api_key)
        self.model = "gemini-1.5-flash"

    def rerank(self, query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
        if not documents:
            return []
        
        logger.info(f"Cloud-Reranking {len(documents)} documents using Gemini...")
        
        # Build prompt for LLM-based Reranking
        doc_list = "\n".join([f"ID {i}: {doc.page_content[:400]}" for i, doc in enumerate(documents)])
        prompt = f"""
        Rate the following document snippets based on their relevance to the specific query: "{query}"
        Answer with ONLY a JSON list of document IDs sorted by relevance (most relevant first).
        Limit the output to a maximum of {top_n} IDs.
        
        Example Output: [3, 0, 2]
        
        Documents:
        {doc_list}
        """
        
        try:
            response = self.client.models.generate_content(model=self.model, contents=prompt)
            # Clean response text and parse JSON
            cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
            ordered_ids = json.loads(cleaned_text)
            
            reranked_docs = []
            for idx in ordered_ids:
                if isinstance(idx, int) and idx < len(documents):
                    reranked_docs.append(documents[idx])
            
            logger.info(f"Reranking complete. Selected {len(reranked_docs)} results.")
            return reranked_docs[:top_n]
        except Exception as e:
            logger.error(f"Cloud Rerank failed: {e}. Falling back to original retrieval order.")
            return documents[:top_n]
