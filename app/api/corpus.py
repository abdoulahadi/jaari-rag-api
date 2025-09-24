"""
Corpus API routes for managing corpus documents
"""
from fastapi import APIRouter
import logging

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