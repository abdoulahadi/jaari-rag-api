"""
Database configuration and setup
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import redis
from typing import Generator

from app.config.settings import settings

# SQLAlchemy setup
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    pool_pre_ping=True,
    pool_recycle=300,  # 5 minutes
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Redis setup
redis_client = redis.from_url(
    settings.REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5,
    retry_on_timeout=True
)


def get_db() -> Generator:
    """Database dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_redis() -> redis.Redis:
    """Redis dependency"""
    return redis_client


class DatabaseManager:
    """Database operations manager"""
    
    @staticmethod
    def create_tables():
        """Create all database tables"""
        Base.metadata.create_all(bind=engine)
    
    @staticmethod
    def drop_tables():
        """Drop all database tables"""
        Base.metadata.drop_all(bind=engine)
    
    @staticmethod
    def test_connection() -> bool:
        """Test database connection"""
        try:
            from sqlalchemy import text
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False


class RedisManager:
    """Redis operations manager"""
    
    def __init__(self, client: redis.Redis = None):
        self.client = client or redis_client
    
    async def set_cache(
        self, 
        key: str, 
        value: str, 
        ttl: int = settings.REDIS_CACHE_TTL
    ) -> bool:
        """Set cache value with TTL"""
        try:
            return self.client.setex(key, ttl, value)
        except Exception:
            return False
    
    async def get_cache(self, key: str) -> str:
        """Get cache value"""
        try:
            return self.client.get(key)
        except Exception:
            return None
    
    async def delete_cache(self, key: str) -> bool:
        """Delete cache key"""
        try:
            return bool(self.client.delete(key))
        except Exception:
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear cache keys matching pattern"""
        try:
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception:
            return 0
    
    def test_connection(self) -> bool:
        """Test Redis connection"""
        try:
            self.client.ping()
            return True
        except Exception:
            return False


# Global instances
db_manager = DatabaseManager()
redis_manager = RedisManager()
