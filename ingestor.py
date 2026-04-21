from typing import List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from google import genai
from loguru import logger

from config import settings
from loader import load_directory
from chunker import chunk_documents


EMBEDDING_DIM = 3072
BATCH_SIZE = 100


class GeminiEmbeddings(Embeddings):
    """Custom embeddings class using the new google.genai package."""

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


def ingest(
    data_dir: str,
    chunk_strategy: str = "recursive",
) -> QdrantVectorStore:

    # Step 1 — Load
    docs = load_directory(data_dir)
    if not docs:
        raise ValueError(f"No documents found in: {data_dir}")

    # Step 2 — Chunk
    chunks = chunk_documents(docs, strategy=chunk_strategy)

    # Step 3 — Single client for everything
    client = QdrantClient(path="./qdrant_storage")

    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Created collection: {settings.qdrant_collection}")
    else:
        logger.info(f"Collection exists: {settings.qdrant_collection}")

    # Step 4 — Embeddings
    embeddings = GeminiEmbeddings(api_key=settings.google_api_key)

    # Step 5 — Test embedding first before processing all chunks
    logger.info("Testing embedding connection...")
    test = embeddings.embed_query("test")
    logger.info(f"Embedding works. Vector size: {len(test)}")

    # Step 6 — Embed + upsert in batches
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
    )

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]
        logger.info(
            f"Embedding batch {i // BATCH_SIZE + 1}/"
            f"{(len(chunks) - 1) // BATCH_SIZE + 1} "
            f"({len(batch)} chunks)"
        )
        vector_store.add_documents(batch)

    logger.success(
        f"Ingestion complete. {len(chunks)} chunks in "
        f"collection '{settings.qdrant_collection}'"
    )
    return vector_store


if __name__ == "__main__":
    store = ingest("./data", chunk_strategy="recursive")
    print(f"Ready. Collection: {settings.qdrant_collection}")