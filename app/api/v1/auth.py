"""
Authentication API routes
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import logging

from app.config.database import get_db
from app.core.auth import auth_manager
from app.models.user import User, UserRole
from app.schemas.user import (
    UserCreate, UserResponse, LoginRequest, LoginResponse, 
    Token, TokenRefresh, PasswordChange
)
from app.services.user_service import UserService

router = APIRouter()
security = HTTPBearer()
logger = logging.getLogger(__name__)

# Initialize user service
user_service = UserService()


@router.post("/register", response_model=LoginResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """Register a new user"""
    try:
        # Check if user already exists
        existing_user = await user_service.get_by_email(db, user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        existing_username = await user_service.get_by_username(db, user_data.username)
        if existing_username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
        
        # Create new user
        user = await user_service.create(db, user_data)
        logger.info(f"New user registered: {user.email}")
        
        # Create tokens for the new user (auto-login after registration)
        token_data = {"sub": str(user.id), "username": user.username, "role": user.role.value}
        access_token = auth_manager.create_access_token(token_data)
        refresh_token = auth_manager.create_refresh_token(token_data)
        
        # Store refresh token JTI
        refresh_payload = auth_manager.verify_token(refresh_token, "refresh")
        user.refresh_token_jti = refresh_payload.get("jti")
        db.commit()
        
        token = Token(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=auth_manager.access_token_expire_minutes * 60
        )
        
        return LoginResponse(
            user=user,
            token=token,
            message="Registration successful"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=LoginResponse)
async def login(
    login_data: LoginRequest,
    db: Session = Depends(get_db)
):
    """User authentication"""
    try:
        # Find user by username or email
        user = await user_service.get_by_username(db, login_data.username)
        if not user:
            user = await user_service.get_by_email(db, login_data.username)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # Check if user is active
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is deactivated"
            )
        
        # Check if user is locked
        if user.is_locked:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is temporarily locked"
            )
        
        # Verify password
        if not auth_manager.verify_password(login_data.password, user.hashed_password):
            # Increment failed attempts
            await user_service.increment_failed_attempts(db, user)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # Reset failed attempts on successful login
        await user_service.reset_failed_attempts(db, user)
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()
        
        # Create tokens
        token_data = {"sub": str(user.id), "username": user.username, "role": user.role.value}
        access_token = auth_manager.create_access_token(token_data)
        refresh_token = auth_manager.create_refresh_token(token_data)
        
        # Store refresh token JTI
        refresh_payload = auth_manager.verify_token(refresh_token, "refresh")
        user.refresh_token_jti = refresh_payload.get("jti")
        db.commit()
        
        token = Token(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=auth_manager.access_token_expire_minutes * 60
        )
        
        logger.info(f"User logged in: {user.email}")
        
        return LoginResponse(user=user, token=token)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    token_data: TokenRefresh,
    db: Session = Depends(get_db)
):
    """Refresh access token"""
    try:
        # Verify refresh token
        payload = auth_manager.verify_token(token_data.refresh_token, "refresh")
        user_id = int(payload.get("sub"))
        jti = payload.get("jti")
        
        # Get user and verify refresh token
        user = await user_service.get_by_id(db, user_id)
        if not user or user.refresh_token_jti != jti:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is deactivated"
            )
        
        # Create new tokens
        token_data = {"sub": str(user.id), "username": user.username, "role": user.role.value}
        access_token = auth_manager.create_access_token(token_data)
        new_refresh_token = auth_manager.create_refresh_token(token_data)
        
        # Update refresh token JTI
        refresh_payload = auth_manager.verify_token(new_refresh_token, "refresh")
        user.refresh_token_jti = refresh_payload.get("jti")
        db.commit()
        
        return Token(
            access_token=access_token,
            refresh_token=new_refresh_token,
            expires_in=auth_manager.access_token_expire_minutes * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token refresh failed"
        )


@router.post("/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Logout user"""
    try:
        # Verify token and get user
        payload = auth_manager.verify_token(credentials.credentials)
        user_id = int(payload.get("sub"))
        
        user = await user_service.get_by_id(db, user_id)
        if user:
            # Clear refresh token
            user.refresh_token_jti = None
            db.commit()
            logger.info(f"User logged out: {user.email}")
        
        return {"message": "Logged out successfully"}
        
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        # Don't raise error for logout, just return success
        return {"message": "Logged out successfully"}


@router.get("/me", response_model=UserResponse)
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Get current user profile"""
    try:
        payload = auth_manager.verify_token(credentials.credentials)
        user_id = int(payload.get("sub"))
        
        user = await user_service.get_by_id(db, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get current user error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user profile"
        )


@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Change user password"""
    try:
        payload = auth_manager.verify_token(credentials.credentials)
        user_id = int(payload.get("sub"))
        
        user = await user_service.get_by_id(db, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Verify current password
        if not auth_manager.verify_password(password_data.current_password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Update password
        user.hashed_password = auth_manager.hash_password(password_data.new_password)
        
        # Clear refresh token to force re-login
        user.refresh_token_jti = None
        
        db.commit()
        
        logger.info(f"Password changed for user: {user.email}")
        
        return {"message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Change password error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )


@router.get("/admin-status")
async def check_admin_status(db: Session = Depends(get_db)):
    """Check if admin users exist in the system (public endpoint)"""
    try:
        has_admin = await user_service.ensure_admin_exists(db)
        
        return {
            "has_admin": has_admin,
            "message": "Admin user exists" if has_admin else "No admin user found",
            "setup_required": not has_admin
        }
        
    except Exception as e:
        logger.error(f"Admin status check error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check admin status"
        )
