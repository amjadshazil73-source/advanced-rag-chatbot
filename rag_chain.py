import os
import time
import asyncio
from typing import Dict, Any, List, AsyncGenerator
from langchain_core.documents import Document
from google import genai
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from google.genai import errors

from config import settings
from retriever import get_vector_store, hybrid_search, BM25Manager
from prompt import format_context
from query_transform import QueryTransformer
from reranker import GeminiReranker

# --- Industry Standard: Observability Setup ---
if settings.langchain_tracing_v2 and settings.langchain_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

def load_all_chunks() -> List[Document]:
    from retriever import get_qdrant_client
    client = get_qdrant_client()
    try:
        points = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )[0]
    except Exception as e:
        logger.warning(f"Could not load chunks from Qdrant: {e}")
        return []

    docs = []
    for point in points:
        payload = point.payload or {}
        content = payload.get("page_content", "")
        metadata = payload.get("metadata", {})
        if content:
            docs.append(Document(
                page_content=content,
                metadata=metadata,
            ))
    logger.info(f"Retrieved {len(docs)} chunks for indexing.")
    return docs


class RAGChain:
    def __init__(self):
        logger.info("Initializing Industry-Grade RAG chain...")
        self.vector_store = get_vector_store()
        all_docs = load_all_chunks()
        
        # Performance: Pre-build BM25 index on startup
        self.bm25_manager = BM25Manager(all_docs)
        
        self.gemini = genai.Client(api_key=settings.google_api_key)
        self.llm_model = settings.llm_model
        
        # Advanced RAG Components
        self.query_transformer = QueryTransformer()
        self.reranker = GeminiReranker()
        
        # Fallback list for production resilience
        self.fallback_models = [
            "gemini-flash-latest",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite-001"
        ]
        logger.info(f"RAG chain ready (Primary: {self.llm_model})")

    def refresh(self):
        """Reloads document chunks and REBUILDS the BM25 index."""
        logger.info("Refreshing document index...")
        all_docs = load_all_chunks()
        self.bm25_manager.update_index(all_docs)
        logger.success(f"Index refreshed. {len(all_docs)} total chunks available.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((errors.ClientError, errors.ServerError)),
        reraise=True
    )
    async def _agenerate_with_retry(self, prompt: str, model_name: str) -> str:
        """Call Gemini asynchronously with retry logic."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: self.gemini.models.generate_content(model=model_name, contents=prompt)
        )
        return response.text

    async def _get_retrieved_context(self, question: str, skip_expansion: bool = False) -> List[Document]:
        """Core retrieval logic using cached BM25 index."""
        unique_contents = set()
        all_retrieved_docs = []

        # 1. Expansion (HyDE + Decomposition)
        search_queries = [question]
        if not skip_expansion:
            if settings.use_decomposition:
                try:
                    decomp_prompt = self.query_transformer.get_decomposition_prompt(question)
                    resp = await self._agenerate_with_retry(decomp_prompt, self.llm_model)
                    search_queries.extend(self.query_transformer.parse_decomposition(resp))
                except Exception as e:
                    logger.warning(f"Decomposition failed: {e}")

            if settings.use_hyde:
                try:
                    hyde_prompt = self.query_transformer.get_hyde_prompt(question)
                    hyde_doc = await self._agenerate_with_retry(hyde_prompt, self.llm_model)
                    search_queries.append(hyde_doc)
                except Exception as e:
                    logger.warning(f"HyDE failed: {e}")

        # 2. Hybrid Search (Optimized with BM25Manager)
        for q in search_queries:
            docs = hybrid_search(q, self.vector_store, self.bm25_manager)
            for doc in docs:
                if doc.page_content not in unique_contents:
                    all_retrieved_docs.append(doc)
                    unique_contents.add(doc.page_content)
        
        if not all_retrieved_docs:
            all_retrieved_docs = hybrid_search(question, self.vector_store, self.bm25_manager)

        # 3. Cross-Encoder Rerank (Final polish)
        return self.reranker.rerank(
            query=question,
            documents=all_retrieved_docs,
            top_n=settings.top_n
        )

    def _build_prompt(self, context: str, question: str) -> str:
        return f"""You are an expert assistant. Answer the question using ONLY the context provided below.

Rules:
- If the answer is not in the context, say: "I don't have enough information..."
- Always cite source file names.
- Be concise.

Context:
{context}

Question:
{question}

Answer:"""

    async def ask(self, question: str, skip_expansion: bool = False) -> Dict[str, Any]:
        """Standard async ask method."""
        reranked_docs = await self._get_retrieved_context(question, skip_expansion)
        context = format_context(reranked_docs)
        prompt = self._build_prompt(context, question)
        
        answer = None
        used_model = self.llm_model
        
        for model_name in [self.llm_model] + self.fallback_models:
            try:
                answer = await self._agenerate_with_retry(prompt, model_name)
                used_model = model_name
                break
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
                continue

        if not answer:
            raise RuntimeError("All models failed.")

        return {
            "answer": answer,
            "source_chunks": reranked_docs,
            "model_used": used_model
        }

    async def ask_stream(self, question: str, skip_expansion: bool = False) -> AsyncGenerator[Dict[str, Any], None]:
        """Streaming Async Generator."""
        reranked_docs = await self._get_retrieved_context(question, skip_expansion)
        context = format_context(reranked_docs)
        prompt = self._build_prompt(context, question)

        yield {"type": "metadata", "source_chunks": [
            {"content": d.page_content, "metadata": d.metadata} for d in reranked_docs
        ]}

        loop = asyncio.get_event_loop()
        
        for model_name in [self.llm_model] + self.fallback_models:
            try:
                logger.info(f"Streaming with {model_name}...")
                def get_stream():
                    return self.gemini.models.generate_content_stream(model=model_name, contents=prompt)
                stream = await loop.run_in_executor(None, get_stream)
                for chunk in stream:
                    if chunk.text:
                        yield {"type": "content", "delta": chunk.text}
                yield {"type": "end", "model": model_name}
                return
            except Exception as e:
                logger.warning(f"Stream failed for {model_name}: {e}")
                continue
        yield {"type": "error", "message": "All models exhausted."}