from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
)
from loguru import logger
import re


def clean_text(text: str) -> str:
    """Remove PDF artifacts and normalise whitespace."""
    text = re.sub(r'\x00', '', text)          # null bytes
    text = re.sub(r'[ \t]+', ' ', text)        # multiple spaces
    text = re.sub(r'\n{3,}', '\n\n', text)    # excess newlines
    return text.strip()


def load_document(file_path: str) -> List[Document]:
    """
    Load a single document from disk.
    Returns a list of Document objects (PDFs return one per page).
    Each Document carries: page_content + metadata.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    logger.info(f"Loading {suffix} file: {path.name}")

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix in (".txt", ".md"):
        loader = TextLoader(str(path), encoding="utf-8")
    elif suffix in (".html", ".htm"):
        loader = UnstructuredHTMLLoader(str(path))
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    docs = loader.load()

    # Enrich metadata + clean text
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
        doc.metadata["source_file"] = path.name
        doc.metadata["file_type"] = suffix

    logger.info(f"Loaded {len(docs)} document(s) from {path.name}")
    return docs


def load_directory(dir_path: str) -> List[Document]:
    """Load all supported documents from a directory."""
    supported = {".pdf", ".txt", ".md", ".html", ".htm"}
    all_docs = []

    for file in Path(dir_path).rglob("*"):
        if file.suffix.lower() in supported:
            try:
                all_docs.extend(load_document(str(file)))
            except Exception as e:
                logger.warning(f"Skipping {file.name}: {e}")

    logger.info(f"Total documents loaded: {len(all_docs)}")
    return all_docs