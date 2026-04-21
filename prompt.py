from langchain_core.prompts import ChatPromptTemplate


RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an expert assistant. Answer the question using ONLY the
context provided below. Do not use any prior knowledge.

Rules:
- If the answer is not in the context, say exactly:
  "I don't have enough information in the provided documents
  to answer this question."
- Always cite which source file your answer comes from.
- Be concise and precise.
- Do not make up facts.

Context:
{context}

Question:
{question}

Answer:
""")


def format_context(docs) -> str:
    """
    Format retrieved chunks into a single context string.
    Each chunk is separated and labelled with its source file
    and page number for citation.
    """
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(
            f"[Source {i+1}: {source}, page {page}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)