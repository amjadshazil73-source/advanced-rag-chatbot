import os
from dotenv import load_dotenv
from ingestor import ingest
from loguru import logger

# Load local .env
load_dotenv()

def bake_knowledge_base():
    logger.info("🥖 STARTING KNOWLEDGE BASE BAKE...")
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        logger.error(f"Directory {data_dir} not found!")
        return

    try:
        # This will populate qdrant_storage/ using Gemini embeddings
        ingest(data_dir)
        logger.success("✅ BAKE COMPLETE! Knowledge is now stored in qdrant_storage/")
    except Exception as e:
        logger.error(f"Bake failed: {e}")

if __name__ == "__main__":
    bake_knowledge_base()
