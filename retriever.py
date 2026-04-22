from typing import List, Optional
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from google import genai
from langchain_core.embeddings import Embeddings
import numpy as np
from loguru import logger

from config import settings

_qdrant_client = None

def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        if settings.qdrant_api_key:
            _qdrant_client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key
            )
        else:
            _qdrant_client = QdrantClient(path="./qdrant_storage")
    return _qdrant_client

class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "models/gemini-embedding-001"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = self.client.models.embed_content(
            model=self.model,
            contents=texts,
        )
        return [e.values for e in result.embeddings]

    def embed_query(self, text: str) -> List[float]:
        result = self.client.models.embed_content(
            model=self.model,
            contents=[text],
        )
        return result.embeddings[0].values


def get_vector_store() -> QdrantVectorStore:
    embeddings = GeminiEmbeddings(api_key=settings.google_api_key)
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
    )

class BM25Manager:
    """Industrial grade BM25 manager that caches the index for speed."""
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.bm25 = None
        if documents:
            self.update_index(documents)

    def update_index(self, documents: List[Document]):
        logger.info(f"Building BM25 index for {len(documents)} docs...")
        self.documents = documents
        tokenized_corpus = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.success("BM25 index built.")

    def search(self, query: str, k: int = 5) -> List[Document]:
        if not self.bm25 or not self.documents:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:k]
        return [self.documents[i] for i in top_indices]

def dense_search(
    vector_store: QdrantVectorStore,
    query: str,
    k: int = None,
) -> List[Document]:
    k = k or settings.top_k
    try:
        return vector_store.similarity_search(query, k=k)
    except Exception as e:
        logger.warning(f"Dense search skipped: {e}")
        return []

def reciprocal_rank_fusion(
    dense_results: List[Document],
    bm25_results: List[Document],
    k: int = 60,
) -> List[Document]:
    scores = {}
    doc_map = {}

    for rank, doc in enumerate(dense_results):
        key = doc.page_content[:150] # Increased precision
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_results):
        key = doc.page_content[:150]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    sorted_keys = sorted(scores, key=scores.get, reverse=True)
    return [doc_map[key] for key in sorted_keys]

def hybrid_search(
    query: str,
    vector_store: QdrantVectorStore,
    bm25_manager: BM25Manager,
) -> List[Document]:
    """Optimized hybrid search using pre-built BM25 index."""
    dense_results = dense_search(vector_store, query)
    bm25_results = bm25_manager.search(query, k=settings.top_k)
    fused = reciprocal_rank_fusion(dense_results, bm25_results)
    return fused[:settings.top_n]