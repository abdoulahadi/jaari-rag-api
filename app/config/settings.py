"""
Configuration settings for Jaari RAG API
"""
from functools import lru_cache
from typing import List, Optional
from pydantic import field_validator, Field, ConfigDict
from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    """Application settings"""
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Jaari RAG API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Agriculture Intelligence Platform API"
    
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # Startup behavior
    AUTO_BUILD_VECTORSTORE: bool = False  # Disable auto-build by default
    
    # Cache directories
    TRANSFORMERS_CACHE: str = "/app/cache/transformers"
    HF_HOME: str = "/app/cache/huggingface"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # Database
    FORCE_SQLITE: bool = False  # Force l'utilisation de SQLite
    DATABASE_URL: str = "sqlite:///./data/jaari_rag.db"
    DATABASE_POOL_SIZE: int = 5  # Réduit pour SQLite
    DATABASE_MAX_OVERFLOW: int = 10  # Réduit pour SQLite
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CACHE_TTL: int = 3600
    
    # JWT Authentication
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Security
    ALLOWED_HOSTS: str = "*"
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8080"
    
    # File Upload
    UPLOAD_PATH: str = "./uploads"
    MAX_FILE_SIZE: int = 100_000_000  # 100MB
    ALLOWED_EXTENSIONS: str = ".pdf,.txt,.docx,.md"
    
    # LLM & RAG (adapted from original project)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:8b"
    OLLAMA_TIMEOUT: int = 180
    OLLAMA_KEEP_ALIVE: str = "10m"
    
    # Embeddings
    EMBEDDINGS_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDINGS_DEVICE: str = "cpu"
    EMBEDDINGS_CACHE_DIR: str = "./cache/embeddings"
    
    # Vector Store
    VECTORSTORE_TYPE: str = "faiss"
    VECTORSTORE_PATH: str = "./data/vectorstore"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    SEARCH_K: int = 15  # Augmenté pour plus de contexte avec chunks intelligents
    
    # Corpus Management
    CORPUS_DIR: str = "./data/corpus"
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60
    
    # Monitoring
    PROMETHEUS_ENDPOINT: str = "/metrics"
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    
    # Email (optional)
    SMTP_TLS: bool = True
    SMTP_PORT: int = 587
    SMTP_HOST: Optional[str] = None
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    
    # Default Admin User (created at startup if no admin exists)
    DEFAULT_ADMIN_EMAIL: str = "admin@jaari.com"
    DEFAULT_ADMIN_USERNAME: str = "admin"
    DEFAULT_ADMIN_PASSWORD: str = "admin123"
    DEFAULT_ADMIN_FULL_NAME: str = "Administrateur Jaari"
    
    # Storage (S3 compatible)
    USE_S3: bool = False
    S3_BUCKET: Optional[str] = None
    S3_REGION: str = "us-east-1"
    S3_ACCESS_KEY: Optional[str] = None
    S3_SECRET_KEY: Optional[str] = None
    
    @property
    def allowed_extensions_list(self) -> List[str]:
        """Convert ALLOWED_EXTENSIONS string to list"""
        return [ext.strip() for ext in self.ALLOWED_EXTENSIONS.split(",")]
    
    @property
    def allowed_hosts_list(self) -> List[str]:
        """Convert ALLOWED_HOSTS string to list"""
        return [host.strip() for host in self.ALLOWED_HOSTS.split(",")]
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Convert CORS_ORIGINS string to list"""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
    
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT.lower() == "development"
    


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Create global settings instance
settings = get_settings()
