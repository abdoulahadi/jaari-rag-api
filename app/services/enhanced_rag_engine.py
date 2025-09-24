"""
Enhanced RAG engine with corpus integration
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import independent RAG functionality  
from app.core.rag_engine import RAGEngine as CoreRAGEngine
from app.config.settings import settings

logger = logging.getLogger(__name__)


class RAGEngine:
    """Enhanced RAG engine with corpus integration for FastAPI"""
    
    def __init__(self):
        self.original_engine = None
        self.is_initialized = False
        self.corpus_manager = None
        
    async def initialize(self) -> bool:
        """Initialize the RAG engine"""
        try:
            # Initialize independent engine
            self.original_engine = CoreRAGEngine()
            
            # Initialize in thread to avoid blocking
            await asyncio.to_thread(self._init_sync)
            
            # Get corpus manager reference
            try:
                from app.services.corpus_manager import corpus_manager
                self.corpus_manager = corpus_manager
                logger.info("✅ RAG engine integrated with corpus manager")
            except ImportError:
                logger.warning("⚠️ Corpus manager not available")
            
            # Load corpus documents into core engine vectorstore
            if self.corpus_manager and settings.CORPUS_DIR:
                loaded = await asyncio.to_thread(
                    self.original_engine.load_documents_from_directory,
                    settings.CORPUS_DIR
                )
                if loaded:
                    logger.info(f"✅ Loaded corpus documents from '{settings.CORPUS_DIR}' into RAG engine")
                else:
                    logger.warning(f"⚠️ No corpus documents found in '{settings.CORPUS_DIR}' to load into RAG engine")
            self.is_initialized = True
            logger.info("✅ RAG engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ RAG engine initialization failed: {str(e)}")
            self.is_initialized = False
            return False
    
    def _init_sync(self):
        """Synchronous initialization of original engine"""
        try:
            if hasattr(self.original_engine, 'initialize'):
                # Check if initialize is a coroutine
                if asyncio.iscoroutinefunction(self.original_engine.initialize):
                    # Run async method in sync context
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self.original_engine.initialize())
                    finally:
                        loop.close()
                else:
                    self.original_engine.initialize()
            elif hasattr(self.original_engine, 'setup'):
                if asyncio.iscoroutinefunction(self.original_engine.setup):
                    # Run async method in sync context
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self.original_engine.setup())
                    finally:
                        loop.close()
                else:
                    self.original_engine.setup()
        except Exception as e:
            logger.error(f"Error in sync initialization: {str(e)}")
            raise e
    
    async def get_response(
        self,
        question: str,
        model_name: str = "llama3.2",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """Get RAG response for a question"""
        if not self.is_initialized or not self.original_engine:
            raise Exception("RAG engine not initialized")
        
        start_time = datetime.utcnow()
        
        try:
            # Use original engine in thread
            response = await asyncio.to_thread(
                self._get_response_sync,
                question,
                model_name,
                temperature,
                max_tokens,
                include_sources
            )
            
            # Calculate response time
            response_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Enhance response with metadata
            enhanced_response = {
                "answer": response.get("response", response.get("answer", "")),
                "sources": response.get("sources", []) if include_sources else [],
                "model": model_name,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_time": response_time,
                "tokens_used": self._estimate_tokens(response.get("response", "")),
                "timestamp": start_time.isoformat()
            }
            
            # Add corpus context if available
            if self.corpus_manager and include_sources:
                corpus_context = await self._get_corpus_context(question)
                if corpus_context:
                    enhanced_response["corpus_sources"] = corpus_context
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"RAG response generation failed: {str(e)}")
            return {
                "answer": f"Désolé, une erreur est survenue lors de la génération de la réponse: {str(e)}",
                "sources": [],
                "model": model_name,
                "temperature": temperature,
                "response_time": (datetime.utcnow() - start_time).total_seconds(),
                "error": str(e)
            }
    
    def _get_response_sync(
        self,
        question: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        include_sources: bool
    ) -> Dict[str, Any]:
        """Synchronous response generation using original engine"""
        try:
            # Check if original engine has the expected method
            if hasattr(self.original_engine, 'get_response'):
                return self.original_engine.get_response(
                    question=question,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            elif hasattr(self.original_engine, 'query'):
                return self.original_engine.query(question)
            elif hasattr(self.original_engine, 'ask'):
                return self.original_engine.ask(question)
            else:
                # Fallback - try to use any callable method
                for method_name in dir(self.original_engine):
                    if not method_name.startswith('_') and callable(getattr(self.original_engine, method_name)):
                        method = getattr(self.original_engine, method_name)
                        try:
                            return method(question)
                        except:
                            continue
                
                raise Exception("No suitable method found in original RAG engine")
                
        except Exception as e:
            logger.error(f"Original engine error: {str(e)}")
            raise
    
    async def _get_corpus_context(self, question: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get relevant context from corpus"""
        try:
            if not self.corpus_manager:
                return []
            
            corpus_results = await self.corpus_manager.search_corpus(question, limit=limit)
            return corpus_results
            
        except Exception as e:
            logger.error(f"Corpus context retrieval failed: {str(e)}")
            return []
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Rough estimation: ~4 characters per token
        return len(text) // 4 if text else 0
    
    async def get_available_models(self) -> List[str]:
        """Get list of available LLM models"""
        try:
            if hasattr(self.original_engine, 'get_available_models'):
                models = await asyncio.to_thread(self.original_engine.get_available_models)
                return models
            else:
                # Default models
                return ["llama3.2", "llama3.1", "codellama", "mistral"]
        except Exception as e:
            logger.error(f"Failed to get available models: {str(e)}")
            return ["llama3.2"]  # Fallback
    
    def get_status(self) -> Dict[str, Any]:
        """Get RAG engine status"""
        status = {
            "initialized": self.is_initialized,
            "engine_available": self.original_engine is not None,
            "corpus_integrated": self.corpus_manager is not None
        }
        
        # Try to get more detailed status from original engine
        try:
            if self.original_engine and hasattr(self.original_engine, 'get_status'):
                original_status = self.original_engine.get_status()
                status.update(original_status)
        except Exception as e:
            logger.debug(f"Could not get original engine status: {str(e)}")
        
        # Add default status info
        if "llm_available" not in status:
            status["llm_available"] = self.is_initialized
        if "vectorstore_available" not in status:
            status["vectorstore_available"] = self.is_initialized
        if "model_name" not in status:
            status["model_name"] = "llama3.2"
        if "embeddings_model" not in status:
            status["embeddings_model"] = "sentence-transformers/all-MiniLM-L6-v2"
        
        return status
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.original_engine and hasattr(self.original_engine, 'cleanup'):
                await asyncio.to_thread(self.original_engine.cleanup)
            
            self.is_initialized = False
            logger.info("✅ RAG engine cleanup completed")
            
        except Exception as e:
            logger.error(f"RAG engine cleanup error: {str(e)}")
    
    async def reload_vectorstore(self):
        """Reload vector store (useful after corpus updates)"""
        try:
            if self.original_engine and hasattr(self.original_engine, 'reload_vectorstore'):
                await asyncio.to_thread(self.original_engine.reload_vectorstore)
                logger.info("✅ Vector store reloaded")
            elif self.corpus_manager:
                # Refresh corpus to reload vectorstore
                await self.corpus_manager.refresh_corpus()
                logger.info("✅ Corpus refreshed")
        except Exception as e:
            logger.error(f"Vector store reload failed: {str(e)}")
            raise
