"""
Document management service
"""
import os
import hashlib
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc
import logging
import asyncio

from app.models.document import Document, DocumentStatus, DocumentType
from app.models.user import User
from app.schemas.document import DocumentCreate, DocumentResponse, DocumentStats
from app.config.settings import settings
from sqlalchemy.sql import func

# Import independent document loader functionality
from app.core.document_loader import DocumentLoader
from app.core.vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)


class DocumentService:
    """Document management and processing service"""
    
    def __init__(self):
        # Initialize with original document loader
        self.document_loader = DocumentLoader()
        self.vector_store_manager = VectorStoreManager()
        # Get corpus manager for integration
        try:
            from app.services.corpus_manager import corpus_manager
            self.corpus_manager = corpus_manager
        except ImportError:
            self.corpus_manager = None
    
    async def upload_document(
        self,
        db: Session,
        user_id: int,
        file_content: bytes,
        filename: str,
        original_filename: str,
        file_type: 'DocumentType',
        mime_type: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        is_public: bool = False
    ) -> Document:
        """Upload and process a new document"""
        try:
            # Calculate file hash for uniqueness check
            file_hash = hashlib.sha256(file_content).hexdigest()
            
            # Check if document already exists
            existing_doc = db.query(Document).filter(
                Document.uploaded_by_id == user_id,
                Document.checksum == file_hash
            ).first()
            
            if existing_doc:
                logger.info(f"Document already exists: {filename}")
                return existing_doc
            
            # Save file to uploads directory
            uploads_dir = Path("uploads")
            uploads_dir.mkdir(exist_ok=True)
            
            # Generate unique filename
            file_path = uploads_dir / f"{file_hash}_{filename}"
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            # Create document record
            db_document = Document(
                filename=f"{file_hash}_{filename}",
                original_filename=original_filename,
                file_path=str(file_path),
                file_size=len(file_content),
                file_type=file_type,
                mime_type=mime_type,
                checksum=file_hash,
                title=title,
                description=description,
                category=category,
                tags=tags,
                is_public=is_public,
                uploaded_by_id=user_id
            )
            
            db.add(db_document)
            db.commit()
            db.refresh(db_document)
            
            # Process document asynchronously
            asyncio.create_task(
                self._process_document_async(db_document.id, str(file_path))
            )
            
            logger.info(f"Document uploaded: {filename} (ID: {db_document.id})")
            return db_document
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to upload document: {str(e)}")
            raise
    
    async def _process_document_async(self, document_id: int, file_path: str):
        """Process document in background"""
        from app.config.database import SessionLocal
        
        db = SessionLocal()
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                return
            
            # Update status to processing
            document.status = DocumentStatus.PROCESSING
            db.commit()
            
            try:
                # Load document using original loader
                documents = self.document_loader.load_document(file_path)
                
                if not documents:
                    raise Exception("No content extracted from document")
                
                # Update document with extracted content
                total_content = "\n\n".join([doc.page_content for doc in documents])
                word_count = len(total_content.split())
                char_count = len(total_content)
                
                document.word_count = word_count
                document.char_count = char_count
                document.page_count = len(documents)
                
                # Add to vector store
                vectorstore_info = await self._add_to_vectorstore(
                    documents, document_id, document.original_filename
                )
                
                # Ensure chunks_count is never None
                chunks_count = vectorstore_info.get("vector_count", 0)
                document.chunks_count = chunks_count if chunks_count is not None else 0
                document.status = DocumentStatus.INDEXED
                document.processing_error = None
                document.indexed_at = func.now()
                
                logger.info(f"Document processed successfully: {document.original_filename}")
                
            except Exception as e:
                document.status = DocumentStatus.FAILED
                document.processing_error = str(e)
                logger.error(f"Document processing failed: {document.original_filename} - {str(e)}")
            
            finally:
                db.commit()
                
        except Exception as e:
            logger.error(f"Background processing error: {str(e)}")
        finally:
            db.close()
    
    async def _add_to_vectorstore(
        self, 
        documents: List[Any], 
        document_id: int, 
        filename: str
    ) -> Dict[str, Any]:
        """Add document to vector store"""
        try:
            # Add metadata to documents
            for doc in documents:
                doc.metadata["document_id"] = document_id
                doc.metadata["filename"] = filename
                doc.metadata["source"] = f"api_upload_{document_id}"
            
            # Use the vector store manager to add documents
            # Share the same vector store as corpus for unified search
            if self.corpus_manager:
                vector_count = await asyncio.to_thread(
                    self.corpus_manager.vector_store_manager.add_documents, documents
                )
            else:
                vector_count = await asyncio.to_thread(
                    self.vector_store_manager.add_documents, documents
                )
            
            # Ensure vector_count is always an integer
            if vector_count is None:
                vector_count = len(documents)
            
            return {"vector_count": vector_count}
            
        except Exception as e:
            logger.error(f"Failed to add to vectorstore: {str(e)}")
            return {"vector_count": 0, "error": str(e)}
    
    async def get_document(
        self, 
        db: Session, 
        document_id: int, 
        user_id: int
    ) -> Optional[Document]:
        """Get document by ID for specific user"""
        return db.query(Document).filter(
            Document.id == document_id,
            Document.uploaded_by_id == user_id
        ).first()
    
    async def get_user_documents(
        self, 
        db: Session, 
        user_id: int, 
        skip: int = 0, 
        limit: int = 20,
        status_filter: Optional[str] = None
    ) -> List[Document]:
        """Get user documents with optional status filter"""
        query = db.query(Document).filter(Document.uploaded_by_id == user_id)
        
        if status_filter:
            query = query.filter(Document.status == status_filter)
        
        return query.order_by(desc(Document.created_at)).offset(skip).limit(limit).all()
    
    async def delete_document(self, db: Session, document_id: int, user_id: int) -> bool:
        """Delete document and remove from vector store"""
        try:
            document = await self.get_document(db, document_id, user_id)
            if not document:
                return False
            
            # Remove from vector store
            await self._remove_from_vectorstore(document_id)
            
            # Delete from database
            db.delete(document)
            db.commit()
            
            logger.info(f"Document deleted: {document_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete document: {str(e)}")
            raise
    
    async def _remove_from_vectorstore(self, document_id: int):
        """Remove document from vector store"""
        try:
            # This would need to be implemented in the vector store manager
            # For now, we'll log the action
            logger.info(f"Document {document_id} should be removed from vector store")
            
        except Exception as e:
            logger.error(f"Failed to remove from vectorstore: {str(e)}")
    
    async def search_documents(
        self, 
        db: Session, 
        user_id: int, 
        query: str, 
        skip: int = 0, 
        limit: int = 20
    ) -> List[Document]:
        """Search user documents"""
        return db.query(Document).filter(
            Document.uploaded_by_id == user_id,
            Document.filename.ilike(f"%{query}%")
        ).order_by(desc(Document.created_at)).offset(skip).limit(limit).all()
    
    async def get_user_stats(self, db: Session, user_id: int) -> DocumentStats:
        """Get user document statistics"""
        try:
            from datetime import datetime, timedelta
            
            documents = db.query(Document).filter(Document.uploaded_by_id == user_id).all()
            
            total_documents = len(documents)
            total_size = sum(doc.file_size for doc in documents if doc.file_size)
            
            # Calculer les documents récents (7 derniers jours)
            seven_days_ago = datetime.utcnow() - timedelta(days=7)
            recent_uploads = len([
                doc for doc in documents 
                if doc.created_at and doc.created_at >= seven_days_ago
            ])
            
            # Count by status
            documents_by_status = {}
            for doc in documents:
                status = doc.status.value if doc.status else 'unknown'
                documents_by_status[status] = documents_by_status.get(status, 0) + 1
            
            # Count by type
            documents_by_type = {}
            for doc in documents:
                file_type = doc.file_type.value if doc.file_type else 'unknown'
                documents_by_type[file_type] = documents_by_type.get(file_type, 0) + 1
            
            # Get most popular documents (by query_count)
            popular_docs = sorted(documents, key=lambda x: x.query_count, reverse=True)[:5]
            most_popular_documents = [
                {
                    "id": doc.id,
                    "title": doc.title or doc.original_filename,
                    "query_count": doc.query_count,
                    "download_count": doc.download_count
                }
                for doc in popular_docs
            ]
            
            return DocumentStats(
                total=total_documents,
                total_documents=total_documents,
                recent_uploads=recent_uploads,
                documents_by_status=documents_by_status,
                documents_by_type=documents_by_type,
                by_type=documents_by_type,  # Alias pour frontend
                total_size=total_size,
                total_size_mb=round(total_size / (1024 * 1024), 2),
                most_popular_documents=most_popular_documents
            )
            
        except Exception as e:
            logger.error(f"Failed to get document stats: {str(e)}")
            raise
    
    async def get_global_stats(self, db: Session) -> DocumentStats:
        """Get global document statistics for admins"""
        try:
            from datetime import datetime, timedelta
            
            # Obtenir tous les documents (pas de filtre par utilisateur)
            documents = db.query(Document).all()
            
            total_documents = len(documents)
            total_size = sum(doc.file_size for doc in documents if doc.file_size)
            
            # Calculer les documents récents (7 derniers jours)
            seven_days_ago = datetime.utcnow() - timedelta(days=7)
            recent_uploads = len([
                doc for doc in documents 
                if doc.created_at and doc.created_at >= seven_days_ago
            ])
            
            # Count by status
            documents_by_status = {}
            for doc in documents:
                status = doc.status.value if doc.status else 'unknown'
                documents_by_status[status] = documents_by_status.get(status, 0) + 1
            
            # Count by type
            documents_by_type = {}
            for doc in documents:
                file_type = doc.file_type.value if doc.file_type else 'unknown'
                documents_by_type[file_type] = documents_by_type.get(file_type, 0) + 1
            
            # Get most popular documents (by query_count)
            popular_docs = sorted(documents, key=lambda x: x.query_count, reverse=True)[:5]
            most_popular_documents = [
                {
                    "id": doc.id,
                    "title": doc.title or doc.original_filename,
                    "query_count": doc.query_count,
                    "download_count": doc.download_count
                }
                for doc in popular_docs
            ]
            
            return DocumentStats(
                total=total_documents,
                total_documents=total_documents,
                recent_uploads=recent_uploads,
                documents_by_status=documents_by_status,
                documents_by_type=documents_by_type,
                by_type=documents_by_type,  # Alias pour frontend
                total_size=total_size,
                total_size_mb=round(total_size / (1024 * 1024), 2),
                most_popular_documents=most_popular_documents
            )
            
        except Exception as e:
            logger.error(f"Failed to get global document stats: {str(e)}")
            raise
    
    async def reprocess_document(
        self, 
        db: Session, 
        document_id: int, 
        user_id: int
    ) -> Optional[Document]:
        """Reprocess a failed document"""
        try:
            document = await self.get_document(db, document_id, user_id)
            if not document:
                return None
            
            if document.status not in [DocumentStatus.FAILED, DocumentStatus.INDEXED]:
                return document
            
            # Reset status
            document.status = DocumentStatus.UPLOADED
            document.processing_error = None
            db.commit()
            
            # Note: In a real implementation, you'd need to store the original file
            # or have a way to retrieve it for reprocessing
            logger.info(f"Document marked for reprocessing: {document_id}")
            
            return document
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to reprocess document: {str(e)}")
            raise
    
    async def get_document_content(
        self, 
        db: Session, 
        document_id: int, 
        user_id: int,
        page: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Get document content (preview or specific page)"""
        document = await self.get_document(db, document_id, user_id)
        if not document:
            return None
        
        if document.status != DocumentStatus.INDEXED:
            return {
                "content": None,
                "status": document.status.value,
                "error": document.processing_error
            }
        
        # For now, return basic info since we don't store content in document table
        # In a full implementation, you'd retrieve content from storage
        return {
            "content": f"Document '{document.original_filename}' processed successfully. Content stored in vector database.",
            "status": "completed",
            "word_count": document.word_count,
            "char_count": document.char_count,
            "page_count": document.page_count
        }
