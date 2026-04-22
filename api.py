import os
print("🚀 API MODULE LOADING...")
import shutil
import tempfile
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import asyncio

from rag_chain import RAGChain
from ingestor import ingest
from loader import load_directory
from chunker import chunk_documents
from config import settings

app = FastAPI(
    title="Industry-Grade RAG API",
    description="Production API for Advanced RAG with Hybrid Search and Reranking",
    version="1.0.0"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG Chain instance
rag_chain = None

def get_rag():
    global rag_chain
    if rag_chain is None:
        rag_chain = RAGChain()
    return rag_chain



class QueryRequest(BaseModel):
    question: str
    skip_expansion: bool = False

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "rag-api", "model": settings.llm_model}

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        rag = get_rag()
        result = await rag.ask(request.question, request.skip_expansion)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask_stream")
async def ask_question_stream(request: QueryRequest):
    rag = get_rag()
    
    async def event_generator():
        async for chunk in rag.ask_stream(request.question, request.skip_expansion):
            yield json.dumps(chunk) + "\n"
            await asyncio.sleep(0) # Yield control

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    """
    Ingests a new PDF into the vector store.
    Industry practice: Save to temp, process, then clean up.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run ingestion logic
        ingest(temp_dir)
        
        # Optimized refresh: Only reload chunks, don't rebuild the whole AI
        rag = get_rag()
        rag.refresh()
        
        return {"message": f"Successfully ingested {file.filename}", "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir)

@app.get("/metrics")
def get_eval_metrics():
    """Returns local evaluation results for the UI dashboard."""
    results_path = "eval_results.json"
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            return json.load(f)
    return []

if __name__ == "__main__":
    import uvicorn
    # Use 127.0.0.1 for internal container communication
    uvicorn.run(app, host="127.0.0.1", port=8000)
