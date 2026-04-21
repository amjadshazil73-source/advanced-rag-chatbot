from rag_chain import RAGChain
from loguru import logger
from google.genai import errors


def print_result(result: dict) -> None:
    """Pretty print a RAG result with sources."""
    print("\n" + "="*60)
    print(f"Question: {result['question']}")
    print("="*60)
    print(f"\nAnswer ({result.get('model_used', 'unknown model')}):\n{result['answer']}")
    print(f"\n--- Sources used ({result['num_chunks_used']} chunks) ---")
    for i, doc in enumerate(result['source_chunks']):
        source = doc.metadata.get('source_file', 'unknown')
        page = doc.metadata.get('page', '?')
        preview = doc.page_content[:120].replace('\n', ' ')
        print(f"[{i+1}] {source} p.{page} — {preview}...")
    print("="*60 + "\n")


def main():
    print("\nInitialising RAG pipeline...")
    try:
        rag = RAGChain()
    except Exception as e:
        print(f"\n[!] Failed to initialize RAG pipeline: {e}")
        return

    print("Ready! Ask questions about your documents.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            question = input("Your question: ").strip()
            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            result = rag.ask(question)
            print_result(result)
        except errors.ClientError as e:
            print(f"\n[!] API Error: {e.message}")
            print("This usually means the rate limit or quota was hit. Please wait a few seconds and try again.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[!] An unexpected error occurred: {e}")
            logger.exception("Unexpected error in main loop")


if __name__ == "__main__":
    main()