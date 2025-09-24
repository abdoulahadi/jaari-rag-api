"""
Document schemas for API
"""
from pydantic import BaseModel, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentType(str, Enum):
    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"


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


class DocumentBase(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    is_public: bool = False


class DocumentCreate(DocumentBase):
    pass


class DocumentUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    is_public: Optional[bool] = None


class DocumentResponse(DocumentBase):
    id: int
    filename: str
    original_filename: str
    file_size: int
    file_type: DocumentType
    mime_type: Optional[str] = None
    
    # Content metadata
    language: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    char_count: Optional[int] = None
    
    # Processing status
    status: DocumentStatus
    processing_error: Optional[str] = None
    chunks_count: int
    
    # User info
    uploaded_by_id: int
    
    # Access control
    is_active: bool
    
    # Usage statistics
    download_count: int
    query_count: int
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    indexed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class DocumentWithUploader(DocumentResponse):
    """Document with uploader information"""
    uploader_username: str
    uploader_full_name: Optional[str] = None


class DocumentStats(BaseModel):
    """Document usage statistics"""
    total: int  # Compatible avec le frontend
    total_documents: int  # Alias pour compatibilité
    recent_uploads: int  # Documents uploadés dans les 7 derniers jours
    documents_by_status: Dict[str, int]
    documents_by_type: Dict[str, int]
    by_type: Dict[str, int]  # Alias pour compatibilité frontend
    total_size: int  # Taille totale en bytes pour le frontend
    total_size_mb: float  # Taille en MB pour l'affichage
    avg_processing_time: Optional[float] = None
    most_popular_documents: List[Dict[str, Any]]
    
    class Config:
        from_attributes = True


class DocumentSearch(BaseModel):
    """Document search parameters"""
    query: str
    category: Optional[str] = None
    file_type: Optional[DocumentType] = None
    tags: Optional[List[str]] = None
    is_public: Optional[bool] = None
    limit: int = 10
    offset: int = 0
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if len(v.strip()) < 2:
            raise ValueError('Search query must be at least 2 characters long')
        return v.strip()


class DocumentSearchResult(BaseModel):
    """Single document search result"""
    document_id: int
    filename: str
    title: Optional[str] = None
    content_snippet: str
    similarity_score: float
    metadata: Dict[str, Any]
    
    
class DocumentSearchResponse(BaseModel):
    """Complete search response"""
    query: str
    results: List[DocumentSearchResult]
    total_results: int
    search_time: float


class DocumentChunkResponse(BaseModel):
    """Document chunk information"""
    id: int
    document_id: int
    chunk_index: int
    content: str
    content_hash: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    char_count: int
    word_count: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class DocumentAnalysis(BaseModel):
    """Document content analysis"""
    document_id: int
    filename: str
    
    # Content statistics
    total_pages: Optional[int] = None
    total_words: int
    total_characters: int
    total_chunks: int
    
    # Language and content analysis
    detected_language: Optional[str] = None
    key_topics: List[str] = []
    keywords: List[str] = []
    
    # Structure analysis
    sections: List[Dict[str, Any]] = []
    
    # Readability metrics
    readability_score: Optional[float] = None
    complexity_level: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    """Response after successful document upload"""
    document: DocumentResponse
    message: str
    processing_started: bool
    estimated_processing_time: Optional[int] = None  # in seconds


class BulkDocumentOperation(BaseModel):
    """For bulk operations on documents"""
    document_ids: List[int]
    operation: str  # "delete", "reindex", "make_public", "make_private"
    
    @field_validator('document_ids')
    @classmethod
    def validate_document_ids(cls, v):
        if len(v) == 0:
            raise ValueError('At least one document ID is required')
        if len(v) > 50:
            raise ValueError('Maximum 50 documents can be processed at once')
        return v


class BulkOperationResponse(BaseModel):
    """Response for bulk operations"""
    operation: str
    total_requested: int
    successful: int
    failed: int
    errors: List[Dict[str, str]] = []
    message: str
