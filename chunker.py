from typing import List, Literal
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import tiktoken
from loguru import logger

from config import settings


def get_tokenizer():
    """tiktoken tokenizer — matches GPT-4's exact token counts."""
    return tiktoken.get_encoding(settings.tokenizer_model)


def token_length(text: str) -> int:
    return len(get_tokenizer().encode(text))


def recursive_chunker() -> RecursiveCharacterTextSplitter:
    """
    Splits on [paragraph, sentence, word, char] boundaries — tries
    the largest separator first, falls back if chunk is still too big.
    Best default: fast and respects natural text structure.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=token_length,   # count TOKENS not chars
        separators=["\n\n", "\n", ". ", " ", ""],
    )
def semantic_chunker() -> SemanticChunker:
    """
    Uses embedding similarity to detect topic-shift boundaries.
    Higher quality, but ~10x slower. Use for high-value documents.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )
    return SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85,
    )

def chunk_documents(
    docs: List[Document],
    strategy: Literal["recursive", "semantic", "parent_child"] = "recursive",
) -> List[Document]:
    """
    Main chunking function. Returns chunks with metadata preserved.
    Each chunk gets a chunk_id and chunk_strategy tag in its metadata.
    """
    logger.info(f"Chunking {len(docs)} docs with strategy='{strategy}'")

    if strategy == "recursive":
        splitter = recursive_chunker()
        chunks = splitter.split_documents(docs)

    elif strategy == "semantic":
        splitter = semantic_chunker()
        chunks = splitter.split_documents(docs)

    elif strategy == "parent_child":
        # Index small child chunks; store large parent for context
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=token_length,
        )
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size * 3,   # 3× bigger parents
            chunk_overlap=settings.chunk_overlap,
            length_function=token_length,
        )
        parents = parent_splitter.split_documents(docs)
        chunks = []
        for i, parent in enumerate(parents):
            children = child_splitter.split_documents([parent])
            for child in children:
                child.metadata["parent_id"] = i
                child.metadata["parent_content"] = parent.page_content
            chunks.extend(children)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")       
    # Tag every chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_strategy"] = strategy
        chunk.metadata["token_count"] = token_length(chunk.page_content)

    logger.info(f"Created {len(chunks)} chunks")
    return chunks                                                                                                           