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

# Password hashing - Configuration robuste
pwd_context = CryptContext(
    schemes=["bcrypt"], 
    deprecated="auto",
    bcrypt__rounds=12,  # Rounds par défaut
    bcrypt__truncate_error=True  # Activer la gestion automatique des erreurs de troncature
)


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
        
        try:
            # Ensure password is not too long for bcrypt (72 bytes max)
            password_bytes = password.encode('utf-8')
            if len(password_bytes) > 72:
                logger.warning(f"Password too long for bcrypt ({len(password_bytes)} bytes), truncating to 72 bytes")
                # Truncate to 72 bytes while preserving UTF-8 encoding
                password_bytes = password_bytes[:72]
                # Decode back, handling potential incomplete UTF-8 sequences
                password = password_bytes.decode('utf-8', errors='ignore')
                logger.info(f"Password truncated to: {len(password)} characters, {len(password.encode('utf-8'))} bytes")
            
            return pwd_context.hash(password)
            
        except Exception as e:
            logger.error(f"Error hashing password: {str(e)}")
            # Fallback: manually truncate and retry
            try:
                password_truncated = password[:72] if len(password) > 72 else password
                logger.warning(f"Retrying with simple truncation: {len(password_truncated)} characters")
                return pwd_context.hash(password_truncated)
            except Exception as fallback_error:
                logger.error(f"Fallback hashing also failed: {str(fallback_error)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Password hashing failed"
                )
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        # Debug logging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Verifying password - length: {len(plain_password)}, bytes: {len(plain_password.encode('utf-8'))}")
        
        try:
            # Ensure password is not too long for bcrypt (72 bytes max)
            password_bytes = plain_password.encode('utf-8')
            if len(password_bytes) > 72:
                logger.warning(f"Password too long for bcrypt verification ({len(password_bytes)} bytes), truncating to 72 bytes")
                # Truncate to 72 bytes while preserving UTF-8 encoding
                password_bytes = password_bytes[:72]
                # Decode back, handling potential incomplete UTF-8 sequences
                plain_password = password_bytes.decode('utf-8', errors='ignore')
                logger.info(f"Password truncated to: {len(plain_password)} characters, {len(plain_password.encode('utf-8'))} bytes")
            
            return pwd_context.verify(plain_password, hashed_password)
            
        except Exception as e:
            logger.error(f"Error verifying password: {str(e)}")
            # Fallback: manually truncate and retry
            try:
                password_truncated = plain_password[:72] if len(plain_password) > 72 else plain_password
                logger.warning(f"Retrying verification with simple truncation: {len(password_truncated)} characters")
                return pwd_context.verify(password_truncated, hashed_password)
            except Exception as fallback_error:
                logger.error(f"Fallback verification also failed: {str(fallback_error)}")
                return False
    
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
