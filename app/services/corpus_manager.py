"""
Corpus manager for automatic document loading at startup
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

from app.config.settings import get_settings

# Import independent document loader functionality
from app.core.document_loader import DocumentLoader
from app.core.vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)
settings = get_settings()


class CorpusManager:
    """Manager for loading and managing corpus documents"""
    
    def __init__(self):
        self.document_loader = DocumentLoader()
        self.vector_store_manager = VectorStoreManager()
        self.corpus_path = Path("data/corpus")
        self.supported_extensions = {'.pdf', '.txt', '.docx', '.doc', '.md', '.rtf'}
        self.loaded_documents = {}
        self.last_scan_time = None
    
    async def initialize_corpus(self, force_reload: bool = False) -> Dict[str, Any]:
        """Initialize corpus by loading all documents from data/corpus folder"""
        logger.info("🌾 Initializing corpus documents...")
        
        if not self.corpus_path.exists():
            logger.warning(f"Corpus directory not found: {self.corpus_path}")
            self.corpus_path.mkdir(parents=True, exist_ok=True)
            return {"status": "created_directory", "documents_loaded": 0}
        
        # Scan for documents
        corpus_files = await self._scan_corpus_directory()
        
        if not corpus_files:
            logger.info("No corpus documents found")
            return {"status": "no_documents", "documents_loaded": 0}
        
        # Load documents if needed
        loaded_count = 0
        errors = []
        
        for file_path in corpus_files:
            try:
                if force_reload or not self._is_document_loaded(file_path):
                    await self._load_corpus_document(file_path)
                    loaded_count += 1
                    logger.info(f"✅ Loaded corpus document: {file_path.name}")
                else:
                    logger.debug(f"Document already loaded: {file_path.name}")
                    
            except Exception as e:
                error_msg = f"Failed to load {file_path.name}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        self.last_scan_time = datetime.utcnow()
        
        result = {
            "status": "completed",
            "documents_found": len(corpus_files),
            "documents_loaded": loaded_count,
            "documents_cached": len(corpus_files) - loaded_count,
            "errors": errors,
            "scan_time": self.last_scan_time.isoformat()
        }
        
        logger.info(f"🎉 Corpus initialization completed: {loaded_count} documents loaded")
        return result
    
    async def _scan_corpus_directory(self) -> List[Path]:
        """Scan corpus directory for supported document files"""
        corpus_files = []
        
        for file_path in self.corpus_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                # Skip hidden files and system files
                if not file_path.name.startswith('.') and not file_path.name.startswith('~'):
                    corpus_files.append(file_path)
        
        logger.info(f"Found {len(corpus_files)} corpus documents")
        return sorted(corpus_files)
    
    def _is_document_loaded(self, file_path: Path) -> bool:
        """Check if document is already loaded in vector store"""
        file_key = str(file_path.relative_to(self.corpus_path))
        file_stat = file_path.stat()
        
        if file_key in self.loaded_documents:
            cached_info = self.loaded_documents[file_key]
            # Check if file was modified since last load
            if cached_info.get("mtime") == file_stat.st_mtime:
                return True
        
        return False
    
    async def _load_corpus_document(self, file_path: Path) -> Dict[str, Any]:
        """Load a single corpus document"""
        try:
            # Load document using original document loader
            documents = await asyncio.to_thread(
                self.document_loader.load_document, str(file_path)
            )
            
            if not documents:
                raise Exception("No content extracted from document")
            
            # Add corpus metadata to documents
            file_key = str(file_path.relative_to(self.corpus_path))
            for doc in documents:
                doc.metadata.update({
                    "source": "corpus",
                    "corpus_file": file_key,
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "file_size": file_path.stat().st_size,
                    "loaded_at": datetime.utcnow().isoformat(),
                    "document_type": "corpus"
                })
            
            # Add to vector store
            vector_count = await asyncio.to_thread(
                self.vector_store_manager.add_documents, documents
            )
            
            # Cache document info
            file_stat = file_path.stat()
            self.loaded_documents[file_key] = {
                "mtime": file_stat.st_mtime,
                "size": file_stat.st_size,
                "loaded_at": datetime.utcnow().isoformat(),
                "vector_count": vector_count,
                "chunks": len(documents)
            }
            
            return {
                "file_path": str(file_path),
                "chunks": len(documents),
                "vector_count": vector_count,
                "status": "loaded"
            }
            
        except Exception as e:
            logger.error(f"Failed to load corpus document {file_path}: {str(e)}")
            raise
    
    async def refresh_corpus(self) -> Dict[str, Any]:
        """Refresh corpus by scanning for new or modified documents"""
        logger.info("🔄 Refreshing corpus documents...")
        
        corpus_files = await self._scan_corpus_directory()
        new_files = []
        updated_files = []
        
        for file_path in corpus_files:
            file_key = str(file_path.relative_to(self.corpus_path))
            file_stat = file_path.stat()
            
            if file_key not in self.loaded_documents:
                new_files.append(file_path)
            else:
                cached_info = self.loaded_documents[file_key]
                if cached_info.get("mtime") != file_stat.st_mtime:
                    updated_files.append(file_path)
        
        # Load new and updated files
        loaded_count = 0
        errors = []
        
        for file_path in new_files + updated_files:
            try:
                if file_path in updated_files:
                    logger.info(f"Updating modified document: {file_path.name}")
                else:
                    logger.info(f"Loading new document: {file_path.name}")
                
                await self._load_corpus_document(file_path)
                loaded_count += 1
                
            except Exception as e:
                error_msg = f"Failed to load {file_path.name}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        # Clean up removed files
        removed_count = await self._cleanup_removed_documents(corpus_files)
        
        result = {
            "status": "refreshed",
            "new_documents": len(new_files),
            "updated_documents": len(updated_files),
            "removed_documents": removed_count,
            "total_loaded": loaded_count,
            "errors": errors,
            "scan_time": datetime.utcnow().isoformat()
        }
        
        logger.info(f"✅ Corpus refresh completed: {loaded_count} documents processed")
        return result
    
    async def _cleanup_removed_documents(self, current_files: List[Path]) -> int:
        """Remove documents that no longer exist in corpus directory"""
        current_file_keys = {str(f.relative_to(self.corpus_path)) for f in current_files}
        cached_file_keys = set(self.loaded_documents.keys())
        
        removed_keys = cached_file_keys - current_file_keys
        
        for file_key in removed_keys:
            logger.info(f"Removing deleted document from cache: {file_key}")
            del self.loaded_documents[file_key]
            # Note: In a full implementation, you'd also remove from vector store
        
        return len(removed_keys)
    
    def get_corpus_status(self) -> Dict[str, Any]:
        """Get current corpus status"""
        return {
            "corpus_path": str(self.corpus_path),
            "documents_loaded": len(self.loaded_documents),
            "last_scan_time": self.last_scan_time.isoformat() if self.last_scan_time else None,
            "supported_extensions": list(self.supported_extensions),
            "documents": {
                file_key: {
                    "loaded_at": info["loaded_at"],
                    "chunks": info["chunks"],
                    "vector_count": info["vector_count"],
                    "size_bytes": info["size"]
                }
                for file_key, info in self.loaded_documents.items()
            }
        }
    
    def get_corpus_stats(self) -> Dict[str, Any]:
        """Get corpus statistics"""
        if not self.loaded_documents:
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "total_vectors": 0,
                "total_size_bytes": 0,
                "file_types": {}
            }
        
        total_chunks = sum(info["chunks"] for info in self.loaded_documents.values())
        total_vectors = sum(info["vector_count"] for info in self.loaded_documents.values())
        total_size = sum(info["size"] for info in self.loaded_documents.values())
        
        # File type distribution
        file_types = {}
        for file_key in self.loaded_documents.keys():
            ext = Path(file_key).suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1
        
        return {
            "total_documents": len(self.loaded_documents),
            "total_chunks": total_chunks,
            "total_vectors": total_vectors,
            "total_size_bytes": total_size,
            "file_types": file_types,
            "avg_chunks_per_doc": round(total_chunks / len(self.loaded_documents), 2),
            "avg_size_per_doc": round(total_size / len(self.loaded_documents), 2)
        }
    
    async def search_corpus(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search in corpus documents"""
        try:
            # Use vector store to search
            results = await asyncio.to_thread(
                self.vector_store_manager.similarity_search,
                query,
                k=limit
            )
            
            # Filter only corpus documents
            corpus_results = []
            for result in results:
                if result.metadata.get("source") == "corpus":
                    corpus_results.append({
                        "content": result.page_content,
                        "metadata": result.metadata,
                        "score": getattr(result, 'score', 0.0)
                    })
            
            return corpus_results
            
        except Exception as e:
            logger.error(f"Corpus search failed: {str(e)}")
            return []


# Global corpus manager instance
corpus_manager = CorpusManager()
