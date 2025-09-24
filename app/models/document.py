"""
Document model for file management
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Boolean, JSON, BigInteger
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import enum
from sqlalchemy import Enum

from app.config.database import Base


class DocumentStatus(enum.Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentType(enum.Enum):
    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    MD = "md"


class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    
    # File information
    file_path = Column(String(500), nullable=False)
    file_size = Column(BigInteger, nullable=False)  # Size in bytes
    file_type = Column(Enum(DocumentType), nullable=False)
    mime_type = Column(String(100), nullable=True)
    checksum = Column(String(64), nullable=False, index=True)  # SHA-256 hash
    
    # Content metadata
    title = Column(String(500), nullable=True)
    description = Column(Text, nullable=True)
    language = Column(String(10), nullable=True)
    page_count = Column(Integer, nullable=True)
    word_count = Column(Integer, nullable=True)
    char_count = Column(Integer, nullable=True)
    
    # Processing status
    status = Column(Enum(DocumentStatus), default=DocumentStatus.UPLOADED, nullable=False)
    processing_error = Column(Text, nullable=True)
    chunks_count = Column(Integer, default=0, nullable=False)
    
    # Vectorization metadata
    embedding_model = Column(String(200), nullable=True)
    vector_ids = Column(JSON, nullable=True)  # Store FAISS vector IDs
    
    # User relationship
    uploaded_by_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    uploaded_by = relationship("User", back_populates="documents")
    
    # Access control
    is_public = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Content categories (for agriculture domain)
    category = Column(String(100), nullable=True)  # e.g., "cereales", "legumineuses", "techniques"
    tags = Column(JSON, nullable=True)  # Array of tags
    
    # Usage statistics
    download_count = Column(Integer, default=0, nullable=False)
    query_count = Column(Integer, default=0, nullable=False)  # How many times referenced in queries
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    indexed_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}', status='{self.status}')>"
    
    @property
    def file_size_mb(self) -> float:
        """Get file size in MB"""
        return round(self.file_size / (1024 * 1024), 2)
    
    @property
    def is_processed(self) -> bool:
        """Check if document is successfully processed"""
        return self.status == DocumentStatus.INDEXED
    
    @property
    def is_processing(self) -> bool:
        """Check if document is currently being processed"""
        return self.status == DocumentStatus.PROCESSING
    
    @property
    def has_failed(self) -> bool:
        """Check if document processing failed"""
        return self.status == DocumentStatus.FAILED


class DocumentChunk(Base):
    """Individual chunks of documents for better tracking"""
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Document relationship
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    document = relationship("Document")
    
    # Chunk information
    chunk_index = Column(Integer, nullable=False)  # Order in document
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False, index=True)  # SHA-256 of content
    
    # Metadata from original document
    page_number = Column(Integer, nullable=True)
    section_title = Column(String(500), nullable=True)
    
    # Vector information
    vector_id = Column(String(100), nullable=True, index=True)  # FAISS vector ID
    embedding_model = Column(String(200), nullable=True)
    
    # Statistics
    char_count = Column(Integer, nullable=False)
    word_count = Column(Integer, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, document_id={self.document_id}, chunk_index={self.chunk_index})>"
