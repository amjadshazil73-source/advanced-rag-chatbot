from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Google Gemini
    google_api_key: str = ""
    embedding_model: str = "models/text-embedding-001" 
    llm_model: str = "gemini-flash-latest"

    # Qdrant vector DB
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rag_documents"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50
    tokenizer_model: str = "cl100k_base"

    # Advanced RAG
    use_hyde: bool = True
    use_decomposition: bool = True

    # Retrieval
    top_k: int = 20
    top_n: int = 8
    
    # Qdrant Cloud (Optional)
    qdrant_api_key: str = ""
    
    # Observability (LangSmith)
    langchain_tracing_v2: bool = False
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: str = ""
    langchain_project: str = "rag-project-portfolio"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()