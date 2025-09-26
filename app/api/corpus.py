"""
Corpus API routes for managing corpus documents
"""
from fastapi import APIRouter
import logging
from app.core.instances import rag_engine
from pathlib import Path

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/status")
async def get_corpus_status():
    """Get corpus loading status"""
    return {"message": "Corpus status endpoint - implementation pending"}


@router.get("/")
async def list_corpus():
    """List corpus documents"""
    return {"message": "Corpus list endpoint - implementation pending"}

@router.post("/build")
async def build_vectorstore():
    corpus_path = Path("data/corpus")
    if corpus_path.exists() and any(corpus_path.iterdir()):
        success = await rag_engine.build_vectorstore_from_directory(str(corpus_path))
        if success:
            return {"status": "success"}
        else:
            return {"status": "failed"}
    return {"status": "no corpus found"}