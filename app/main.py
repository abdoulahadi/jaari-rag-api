"""
FastAPI main application
"""
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import time
import json
import logging
from contextlib import asynccontextmanager

from app.config.settings import settings
from app.config.database import db_manager, redis_manager
from app.core.instances import rag_engine
from app.services.corpus_manager import corpus_manager

# Initialize RAG engine
from app.api.v1 import auth, chat, users
from app.api import websocket, corpus, documents, recommendations, analytics
from app.api import audio  # Nouveau module audio
from app.api import websocket as websockets
from app.utils.middleware import (
    RateLimitMiddleware,
    LoggingMiddleware,
    SecurityHeadersMiddleware
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if settings.LOG_FORMAT != "json"
    else None
)
logger = logging.getLogger("jaari-rag-api")


# Lifespan events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🚀 Starting Jaari RAG API...")
    
    # Initialize Google Cloud credentials
    try:
        from app.utils.google_credentials import setup_google_credentials
        credentials_path = setup_google_credentials()
        if credentials_path:
            logger.info("✅ Google Cloud credentials configured")
        else:
            logger.warning("⚠️ Google Cloud credentials not found - translation features may be limited")
    except Exception as e:
        logger.warning(f"⚠️ Google Cloud credentials setup failed: {str(e)}")
    
    # Test database connection
    try:
        # Import models to ensure they're registered with Base
        from app.models import user, document, conversation
        
        # Create tables if they don't exist
        db_manager.create_tables()
        
        if db_manager.test_connection():
            logger.info("✅ Database connected")
        else:
            logger.warning("⚠️ Database connection failed - some features may be limited")
    except Exception as e:
        logger.warning(f"⚠️ Database connection failed: {str(e)} - some features may be limited")
    
    # Test Redis connection
    try:
        if redis_manager.test_connection():
            logger.info("✅ Redis connected")
        else:
            logger.warning("⚠️ Redis connection failed - some features may be limited")
    except Exception as e:
        logger.warning(f"⚠️ Redis connection failed: {str(e)} - some features may be limited")
    
    # Initialize RAG engine (non-blocking)
    try:
        logger.info("Initializing RAG Engine...")
        if await rag_engine.initialize():
            logger.info("✅ RAG Engine initialized")
            
            # Check auto-build setting
            if settings.AUTO_BUILD_VECTORSTORE:
                # Check if vectorstore needs to be built
                rag_status = rag_engine.get_status()
                if not rag_status.get("vectorstore_available", False):
                    logger.info("🔄 No vectorstore found, auto-building from corpus...")
                    try:
                        from pathlib import Path
                        corpus_path = Path("data/corpus")
                        if corpus_path.exists() and any(corpus_path.iterdir()):
                            # Auto-build vectorstore
                            success = await rag_engine.build_vectorstore_from_directory(str(corpus_path))
                            if success:
                                logger.info("✅ Auto-vectorization completed successfully")
                            else:
                                logger.warning("⚠️ Auto-vectorization failed")
                        else:
                            logger.warning("⚠️ No corpus directory found for auto-vectorization")
                    except Exception as e:
                        logger.warning(f"⚠️ Auto-vectorization failed: {str(e)}")
            else:
                logger.info("⚡ Auto-build disabled for faster startup - use /api/v1/corpus/build to build vectorstore")
        else:
            logger.warning("⚠️ RAG Engine initialization partial - some features may be limited")
    except Exception as e:
        logger.warning(f"⚠️ RAG Engine initialization failed: {str(e)} - continuing with limited functionality")
    
    # Initialize default admin user
    try:
        from app.services.user_service import UserService
        from app.config.database import SessionLocal
        
        user_service = UserService()
        
        # Use a separate database session for admin creation
        admin_db = SessionLocal()
        try:
            admin_data = {
                "email": settings.DEFAULT_ADMIN_EMAIL,
                "username": settings.DEFAULT_ADMIN_USERNAME,
                "password": settings.DEFAULT_ADMIN_PASSWORD,
                "full_name": settings.DEFAULT_ADMIN_FULL_NAME
            }
            
            admin_user = await user_service.create_default_admin(admin_db, admin_data)
            if admin_user:
                logger.info(f"👤 Default admin initialized: {admin_user.email}")
            
        finally:
            admin_db.close()
            
    except Exception as e:
        logger.error(f"⚠️ Default admin initialization failed: {str(e)}")
        # Don't fail startup for admin creation issues
    
    logger.info("🎉 Jaari RAG API started successfully!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Jaari RAG API...")


# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# Add middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(LoggingMiddleware)
# Temporarily disabled rate limiting middleware
# app.add_middleware(
#     RateLimitMiddleware,
#     calls=settings.RATE_LIMIT_REQUESTS,
#     period=settings.RATE_LIMIT_WINDOW
# )


# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    # Convert error details to JSON serializable format
    error_details = []
    for error in exc.errors():
        if isinstance(error, dict):
            # Convert any non-serializable values to strings
            serializable_error = {}
            for key, value in error.items():
                try:
                    json.dumps(value)  # Test if value is JSON serializable
                    serializable_error[key] = value
                except (TypeError, ValueError):
                    serializable_error[key] = str(value)
            error_details.append(serializable_error)
        else:
            error_details.append(str(error))
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "message": "Invalid request data",
            "details": error_details,
            "timestamp": time.time()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred" if settings.is_production else str(exc),
            "timestamp": time.time()
        }
    )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🌾 Welcome to Jaari RAG API - Agriculture Intelligence Platform",
        "version": settings.VERSION,
        "status": "active",
        "docs": "/docs",
        "api": settings.API_V1_STR
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check database
    db_status = db_manager.test_connection()
    
    # Check Redis
    redis_status = redis_manager.test_connection()
    
    # Check RAG engine
    rag_status = rag_engine.get_status()
    
    health_data = {
        "status": "healthy" if all([db_status, rag_status["initialized"]]) else "degraded",
        "timestamp": time.time(),
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "services": {
            "database": "healthy" if db_status else "unhealthy",
            "redis": "healthy" if redis_status else "unhealthy",
            "rag_engine": "healthy" if rag_status["initialized"] else "unhealthy",
            "llm": "healthy" if rag_status["llm_available"] else "unhealthy",
            "vectorstore": "healthy" if rag_status["vectorstore_available"] else "unhealthy"
        },
        "rag_info": {
            "model": rag_status.get("model_name"),
            "embeddings_model": rag_status.get("embeddings_model"),
            "documents_count": rag_status.get("documents_count", 0)
        }
    }
    
    status_code = 200 if health_data["status"] == "healthy" else 503
    return JSONResponse(content=health_data, status_code=status_code)


# Include routers
app.include_router(
    auth.router,
    prefix=f"{settings.API_V1_STR}/auth",
    tags=["Authentication"]
)

app.include_router(
    chat.router,
    prefix=f"{settings.API_V1_STR}/chat",
    tags=["Chat & RAG"]
)

app.include_router(
    documents.router,
    prefix=f"{settings.API_V1_STR}/documents",
    tags=["Documents"]
)

app.include_router(
    users.router,
    prefix=f"{settings.API_V1_STR}/users",
    tags=["Users"]
)

app.include_router(
    analytics.router,
    prefix=f"{settings.API_V1_STR}/analytics",
    tags=["Analytics"]
)

# WebSocket routes
app.include_router(
    websockets.router,
    prefix="/ws",
    tags=["WebSocket"]
)

# Corpus routes
app.include_router(
    corpus.router,
    prefix=f"{settings.API_V1_STR}/corpus",
    tags=["Corpus"]
)

# Recommendations routes
app.include_router(
    recommendations.router,
    prefix=f"{settings.API_V1_STR}/recommendations",
    tags=["Recommendations"]
)

# Audio TTS routes
app.include_router(
    audio.router,
    tags=["Audio TTS"]
)


# Metrics endpoint (for Prometheus)
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    # This would typically return Prometheus-formatted metrics
    # For now, return basic JSON metrics
    rag_status = rag_engine.get_status()
    
    return {
        "api_requests_total": "Counter metric would go here",
        "api_request_duration_seconds": "Histogram metric would go here",
        "rag_documents_total": rag_status.get("documents_count", 0),
        "rag_engine_status": 1 if rag_status["initialized"] else 0,
        "database_connections_active": "Gauge metric would go here"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
