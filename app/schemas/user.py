"""
User schemas for API requests and responses
"""
from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional, List
from datetime import datetime
from enum import Enum


class UserRole(str, Enum):
    ADMIN = "admin"
    EXPERT = "expert"
    USER = "user"


class UserBase(BaseModel):
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    organization: Optional[str] = None
    bio: Optional[str] = None
    country: Optional[str] = None
    language_preference: str = "fr"


class UserCreate(UserBase):
    password: str
    role: Optional[UserRole] = UserRole.USER
    is_active: Optional[bool] = True
    is_verified: Optional[bool] = False
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters long')
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username can only contain letters, numbers, hyphens and underscores')
        return v


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    organization: Optional[str] = None
    bio: Optional[str] = None
    country: Optional[str] = None
    language_preference: Optional[str] = None


class UserUpdateAdmin(UserUpdate):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None


class PasswordChange(BaseModel):
    current_password: str
    new_password: str
    
    @field_validator('new_password')
    @classmethod
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v


class UserResponse(UserBase):
    id: int
    role: UserRole
    is_active: bool
    is_verified: bool
    api_key: Optional[str] = None
    last_login: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class UserPublic(BaseModel):
    """Public user information (limited fields)"""
    id: int
    username: str
    full_name: Optional[str] = None
    organization: Optional[str] = None
    country: Optional[str] = None
    role: UserRole
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserStats(BaseModel):
    """User usage statistics"""
    total_conversations: int
    total_messages: int
    total_documents: int
    total_queries: int
    avg_response_time: Optional[float] = None
    last_activity: Optional[datetime] = None


class UserWithStats(UserResponse):
    """User with usage statistics"""
    stats: UserStats


# Authentication schemas
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenRefresh(BaseModel):
    refresh_token: str


class LoginRequest(BaseModel):
    username: str  # Can be username or email
    password: str


class LoginResponse(BaseModel):
    user: UserResponse
    token: Token
    message: str = "Login successful"


# API Key schemas
class APIKeyCreate(BaseModel):
    name: str
    description: Optional[str] = None


class APIKeyResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    key: str
    created_at: datetime
    last_used: Optional[datetime] = None
    is_active: bool
    
    class Config:
        from_attributes = True
