"""
Authentication and JWT utilities
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
import secrets

from app.config.settings import settings

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthManager:
    """Authentication manager with JWT support"""
    
    def __init__(self):
        self.secret_key = settings.SECRET_KEY
        self.algorithm = settings.ALGORITHM
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
        self.refresh_token_expire_days = settings.REFRESH_TOKEN_EXPIRE_DAYS
    
    def create_access_token(
        self, 
        data: Dict[str, Any], 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({
            "exp": expire,
            "type": "access",
            "iat": datetime.utcnow()
        })
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(
        self, 
        data: Dict[str, Any]
    ) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        
        to_encode.update({
            "exp": expire,
            "type": "refresh",
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(32)  # Unique token ID
        })
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            if payload.get("type") != token_type:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid token type. Expected {token_type}"
                )
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired"
                )
            
            return payload
            
        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Could not validate credentials: {str(e)}"
            )
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        # Debug logging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Hashing password - length: {len(password)}, bytes: {len(password.encode('utf-8'))}")
        
        # Ensure password is not too long for bcrypt (72 bytes max)
        if len(password.encode('utf-8')) > 72:
            logger.warning(f"Password too long for bcrypt ({len(password.encode('utf-8'))} bytes), truncating to 72 bytes")
            # Truncate to 72 bytes while preserving UTF-8 encoding
            password_bytes = password.encode('utf-8')[:72]
            # Decode back, handling potential incomplete UTF-8 sequences
            password = password_bytes.decode('utf-8', errors='ignore')
        
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        # Debug logging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Verifying password - length: {len(plain_password)}, bytes: {len(plain_password.encode('utf-8'))}")
        
        # Ensure password is not too long for bcrypt (72 bytes max)
        if len(plain_password.encode('utf-8')) > 72:
            logger.warning(f"Password too long for bcrypt verification ({len(plain_password.encode('utf-8'))} bytes), truncating to 72 bytes")
            # Truncate to 72 bytes while preserving UTF-8 encoding
            password_bytes = plain_password.encode('utf-8')[:72]
            # Decode back, handling potential incomplete UTF-8 sequences
            plain_password = password_bytes.decode('utf-8', errors='ignore')
        
        return pwd_context.verify(plain_password, hashed_password)
    
    def generate_api_key(self) -> str:
        """Generate secure API key"""
        return secrets.token_urlsafe(32)


# Global auth manager instance
auth_manager = AuthManager()


def create_access_token(data: Dict[str, Any]) -> str:
    """Create access token - convenience function"""
    return auth_manager.create_access_token(data)


def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create refresh token - convenience function"""
    return auth_manager.create_refresh_token(data)


def verify_token(token: str, token_type: str = "access") -> Dict[str, Any]:
    """Verify token - convenience function"""
    return auth_manager.verify_token(token, token_type)


def hash_password(password: str) -> str:
    """Hash password - convenience function"""
    return auth_manager.hash_password(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password - convenience function"""
    return auth_manager.verify_password(plain_password, hashed_password)
