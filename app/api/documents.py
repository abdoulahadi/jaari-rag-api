"""
Document management API routes
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import List, Optional
import logging
import mimetypes
import os

from app.config.database import get_db
from app.core.auth import auth_manager
from app.models.user import User
from app.models.document import Document, DocumentStatus, DocumentType
from app.schemas.document import (
    DocumentResponse, DocumentWithUploader, DocumentStats, DocumentSearch,
    DocumentSearchResponse, DocumentUploadResponse, DocumentUpdate,
    BulkDocumentOperation, BulkOperationResponse, DocumentAnalysis
)
from app.services.document_service import DocumentService

router = APIRouter()
security = HTTPBearer()
logger = logging.getLogger(__name__)

# Initialize document service
document_service = DocumentService()

# Allowed file types and their MIME types
ALLOWED_FILE_TYPES = {
    'pdf': ['application/pdf'],
    'txt': ['text/plain'],
    'docx': ['application/vnd.openxmlformats-officedocument.wordprocessingml.document'],
    'md': ['text/markdown', 'text/x-markdown']
}

# Maximum file size (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    try:
        payload = auth_manager.verify_token(credentials.credentials)
        user_id = int(payload.get("sub"))
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return user
    except Exception as e:
        logger.error(f"Get current user error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )


def validate_file(file: UploadFile) -> str:
    """Validate uploaded file"""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided"
        )
    
    # Get file extension
    file_ext = os.path.splitext(file.filename)[1].lower().lstrip('.')
    if file_ext not in ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not supported. Allowed types: {', '.join(ALLOWED_FILE_TYPES.keys())}"
        )
    
    # Validate MIME type
    if file.content_type and file.content_type not in ALLOWED_FILE_TYPES[file_ext]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid MIME type for {file_ext} file"
        )
    
    return file_ext


@router.get("/", response_model=List[DocumentResponse])
async def list_documents(
    skip: int = 0,
    limit: int = 20,
    status_filter: Optional[str] = None,
    category: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List user documents with optional filters"""
    try:
        # Validate status filter
        if status_filter and status_filter not in [s.value for s in DocumentStatus]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status. Allowed: {[s.value for s in DocumentStatus]}"
            )
        
        documents = await document_service.get_user_documents(
            db, current_user.id, skip=skip, limit=limit, 
            status_filter=status_filter
        )
        
        return documents
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List documents error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents"
        )


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    is_public: bool = Form(False),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload a new document"""
    try:
        # Validate file
        file_ext = validate_file(file)
        
        # Read file content
        file_content = await file.read()
        
        # Check file size
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.1f}MB"
            )
        
        # Parse tags if provided
        tag_list = []
        if tags:
            tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
        
        # Create document record
        document = await document_service.upload_document(
            db=db,
            user_id=current_user.id,
            file_content=file_content,
            filename=file.filename,
            original_filename=file.filename,
            file_type=DocumentType(file_ext),
            mime_type=file.content_type,
            title=title,
            description=description,
            category=category,
            tags=tag_list,
            is_public=is_public
        )
        
        logger.info(f"Document uploaded by user {current_user.id}: {file.filename}")
        
        return DocumentUploadResponse(
            document=document,
            message="Document uploaded successfully and processing started",
            processing_started=True,
            estimated_processing_time=60  # Estimate 1 minute
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload document error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload document"
        )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get document details"""
    try:
        document = await document_service.get_document(db, document_id, current_user.id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document"
        )


@router.patch("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: int,
    document_update: DocumentUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update document metadata"""
    try:
        document = await document_service.get_document(db, document_id, current_user.id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Update fields
        update_data = document_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(document, field, value)
        
        db.commit()
        db.refresh(document)
        
        logger.info(f"Document updated by user {current_user.id}: {document_id}")
        
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Update document error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update document"
        )


@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a document"""
    try:
        success = await document_service.delete_document(db, document_id, current_user.id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        logger.info(f"Document deleted by user {current_user.id}: {document_id}")
        
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete document error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document"
        )


@router.get("/{document_id}/content")
async def get_document_content(
    document_id: int,
    page: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get document content"""
    try:
        content = await document_service.get_document_content(
            db, document_id, current_user.id, page=page
        )
        
        if not content:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return content
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document content error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document content"
        )


@router.post("/{document_id}/reprocess", response_model=DocumentResponse)
async def reprocess_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Reprocess a failed document"""
    try:
        document = await document_service.reprocess_document(db, document_id, current_user.id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        logger.info(f"Document reprocessing initiated by user {current_user.id}: {document_id}")
        
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reprocess document error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reprocess document"
        )


@router.post("/search", response_model=DocumentSearchResponse)
async def search_documents(
    search_params: DocumentSearch,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Search user documents"""
    try:
        results = await document_service.search_documents(
            db, current_user.id, search_params.query, 
            skip=search_params.offset, limit=search_params.limit
        )
        
        # For now, return simple search results
        # In a full implementation, this would use semantic search
        search_results = []
        for doc in results:
            search_results.append({
                "document_id": doc.id,
                "filename": doc.filename,
                "title": doc.title,
                "content_snippet": doc.description or "No description available",
                "similarity_score": 1.0,
                "metadata": {
                    "file_type": doc.file_type.value,
                    "file_size": doc.file_size,
                    "status": doc.status.value,
                    "created_at": doc.created_at.isoformat()
                }
            })
        
        return DocumentSearchResponse(
            query=search_params.query,
            results=search_results,
            total_results=len(search_results),
            search_time=0.1
        )
        
    except Exception as e:
        logger.error(f"Search documents error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search documents"
        )


@router.get("/stats/overview", response_model=DocumentStats)
async def get_document_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get document statistics - global for admins, user-specific for others"""
    try:
        # Si l'utilisateur est admin, retourner les statistiques globales
        if current_user.is_admin:
            stats = await document_service.get_global_stats(db)
        else:
            stats = await document_service.get_user_stats(db, current_user.id)
        return stats
        
    except Exception as e:
        logger.error(f"Get document stats error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document statistics"
        )


@router.post("/bulk", response_model=BulkOperationResponse)
async def bulk_document_operation(
    operation: BulkDocumentOperation,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Perform bulk operations on documents"""
    try:
        successful = 0
        failed = 0
        errors = []
        
        for doc_id in operation.document_ids:
            try:
                if operation.operation == "delete":
                    success = await document_service.delete_document(db, doc_id, current_user.id)
                    if success:
                        successful += 1
                    else:
                        failed += 1
                        errors.append({"document_id": str(doc_id), "error": "Document not found"})
                
                elif operation.operation == "reindex":
                    doc = await document_service.reprocess_document(db, doc_id, current_user.id)
                    if doc:
                        successful += 1
                    else:
                        failed += 1
                        errors.append({"document_id": str(doc_id), "error": "Document not found"})
                
                elif operation.operation in ["make_public", "make_private"]:
                    doc = await document_service.get_document(db, doc_id, current_user.id)
                    if doc:
                        doc.is_public = (operation.operation == "make_public")
                        db.commit()
                        successful += 1
                    else:
                        failed += 1
                        errors.append({"document_id": str(doc_id), "error": "Document not found"})
                
                else:
                    failed += 1
                    errors.append({"document_id": str(doc_id), "error": "Unknown operation"})
                    
            except Exception as e:
                failed += 1
                errors.append({"document_id": str(doc_id), "error": str(e)})
        
        logger.info(f"Bulk operation {operation.operation} by user {current_user.id}: {successful} successful, {failed} failed")
        
        return BulkOperationResponse(
            operation=operation.operation,
            total_requested=len(operation.document_ids),
            successful=successful,
            failed=failed,
            errors=errors,
            message=f"Bulk {operation.operation} completed: {successful} successful, {failed} failed"
        )
        
    except Exception as e:
        logger.error(f"Bulk operation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform bulk operation"
        )