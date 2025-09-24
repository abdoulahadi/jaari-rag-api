"""
Additional Pydantic schemas for document content
"""
from pydantic import BaseModel
from typing import Optional, Dict, Any


class DocumentContent(BaseModel):
    """Document content response schema"""
    content: Optional[str] = None
    status: str
    error: Optional[str] = None
    word_count: Optional[int] = None
    char_count: Optional[int] = None
    page_count: Optional[int] = None
    
    class Config:
        from_attributes = True
