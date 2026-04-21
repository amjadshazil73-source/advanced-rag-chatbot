from typing import List
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from loguru import logger

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Loading local reranker model: {model_name}...")
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
        """
        Refines the initial retrieval results by re-scoring them with a Cross-Encoder.
        """
        if not documents:
            return []
        
        # Prepare pairs: (query, document_text)
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Get relevance scores
        logger.info(f"Reranking {len(documents)} documents...")
        scores = self.model.predict(pairs)
        
        # Pair documents with their scores and sort
        doc_scores = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        
        # Return top N re-ranked documents
        reranked_docs = [doc for doc, score in doc_scores[:top_n]]
        logger.info(f"Reranking complete. Selected top {len(reranked_docs)} results.")
        return reranked_docs
