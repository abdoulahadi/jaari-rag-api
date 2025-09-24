"""
Configuration management for the API
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings"""
    
    # Project info
    PROJECT_NAME: str = "Jaari RAG API"
    DESCRIPTION: str = "Agriculture Intelligence Platform with RAG capabilities"
    VERSION: str = "1.0.0"
    
    # API configuration
    API_V1_STR: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    # Security
    SECRET_KEY: str = "your-super-secret-key-change-this-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"
    
    # Database
    DATABASE_URL: str = "sqlite:///./data/jaari_rag.db"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Ollama/LLM
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    DEFAULT_LLM_MODEL: str = "llama3.2"
    EMBEDDINGS_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # File upload
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    UPLOAD_DIR: str = "data/uploads"
    CORPUS_DIR: str = "data/corpus"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "text"  # or "json"
    
    # CORS
    ALLOWED_ORIGINS: list = ["*"]
    
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields from .env file


def get_settings() -> Settings:
    """Get settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()
