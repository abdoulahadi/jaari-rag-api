"""
User management API routes
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import Optional, List
import logging

from app.config.database import get_db
from app.core.auth import auth_manager
from app.models.user import User, UserRole
from app.schemas.user import UserResponse, UserCreate, UserUpdate

router = APIRouter()
security = HTTPBearer()
logger = logging.getLogger(__name__)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    try:
        payload = auth_manager.verify_token(credentials.credentials)
        user_id = int(payload.get("sub"))
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return user
    except Exception as e:
        logger.error(f"Get current user error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require admin role"""
    user_role = current_user.role
    if hasattr(user_role, 'value'):
        role_value = user_role.value.lower()
    else:
        role_value = str(user_role).lower()
    
    if role_value != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


@router.get("/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return {
        "id": current_user.id,
        "email": current_user.email,
        "username": current_user.username,
        "full_name": current_user.full_name,
        "role": current_user.role.value if hasattr(current_user.role, 'value') else str(current_user.role),
        "is_active": current_user.is_active,
        "is_verified": current_user.is_verified,
        "organization": current_user.organization,
        "bio": current_user.bio,
        "country": current_user.country,
        "language_preference": current_user.language_preference,
        "last_login": current_user.last_login,
        "created_at": current_user.created_at,
        "updated_at": current_user.updated_at
    }


@router.post("/", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Create a new user (admin only)"""
    try:
        # Check if user with email already exists
        existing_user = db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        # Check if user with username already exists
        if user_data.username:
            existing_username = db.query(User).filter(User.username == user_data.username).first()
            if existing_username:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="User with this username already exists"
                )
        
        # Hash password
        hashed_password = auth_manager.get_password_hash(user_data.password)
        
        # Create user
        db_user = User(
            email=user_data.email,
            username=user_data.username,
            full_name=user_data.full_name,
            hashed_password=hashed_password,
            role=UserRole(user_data.role) if isinstance(user_data.role, str) else user_data.role,
            is_active=user_data.is_active if hasattr(user_data, 'is_active') else True,
            is_verified=user_data.is_verified if hasattr(user_data, 'is_verified') else False,
            organization=user_data.organization if hasattr(user_data, 'organization') else None,
            bio=user_data.bio if hasattr(user_data, 'bio') else None,
            country=user_data.country if hasattr(user_data, 'country') else None,
            language_preference=user_data.language_preference if hasattr(user_data, 'language_preference') else 'fr'
        )
        
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        logger.info(f"User created successfully: {db_user.email}")
        
        return UserResponse(
            id=db_user.id,
            email=db_user.email,
            username=db_user.username,
            full_name=db_user.full_name,
            role=db_user.role.value if hasattr(db_user.role, 'value') else str(db_user.role),
            is_active=db_user.is_active,
            is_verified=db_user.is_verified,
            organization=db_user.organization,
            bio=db_user.bio,
            country=db_user.country,
            language_preference=db_user.language_preference,
            last_login=db_user.last_login,
            created_at=db_user.created_at,
            updated_at=db_user.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create user error: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )


@router.get("/")
async def get_users(
    skip: int = Query(0, description="Number of users to skip"),
    limit: int = Query(10, description="Maximum number of users to return"),
    role: Optional[str] = Query(None, description="Filter by role"),
    is_active: Optional[str] = Query(None, description="Filter by active status"),
    search: Optional[str] = Query(None, description="Search in email, username, or full_name"),
    sort_by: str = Query("created_at", description="Field to sort by"),
    sort_order: str = Query("desc", description="Sort order: asc or desc"),
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get all users (admin only)"""
    try:
        # Build query
        query = db.query(User)
        
        # Apply filters - handle "null" string values from frontend
        if role and role.lower() not in ["null", "none", ""]:
            try:
                role_enum = UserRole(role.lower())
                query = query.filter(User.role == role_enum)
            except ValueError:
                pass  # Invalid role, ignore filter
        
        # Handle is_active parameter - convert string to boolean or ignore null values
        if is_active and is_active.lower() not in ["null", "none", ""]:
            if is_active.lower() in ["true", "1", "yes"]:
                query = query.filter(User.is_active == True)
            elif is_active.lower() in ["false", "0", "no"]:
                query = query.filter(User.is_active == False)
            
        if search and search.strip() and search.lower() not in ["null", "none", ""]:
            search_filter = f"%{search.strip()}%"
            query = query.filter(
                (User.email.ilike(search_filter)) |
                (User.username.ilike(search_filter)) |
                (User.full_name.ilike(search_filter))
            )
        
        # Apply sorting
        valid_sort_fields = ["id", "email", "username", "full_name", "role", "is_active", "created_at", "updated_at"]
        if sort_by in valid_sort_fields and hasattr(User, sort_by):
            sort_column = getattr(User, sort_by)
            if sort_order.lower() == "desc":
                query = query.order_by(desc(sort_column))
            else:
                query = query.order_by(sort_column)
        else:
            # Default sorting
            query = query.order_by(desc(User.created_at))
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        users = query.offset(skip).limit(limit).all()
        
        # Format response
        users_data = []
        for user in users:
            users_data.append({
                "id": user.id,
                "email": user.email,
                "username": user.username,
                "full_name": user.full_name,
                "role": user.role.value if hasattr(user.role, 'value') else str(user.role),
                "is_active": user.is_active,
                "is_verified": user.is_verified,
                "organization": user.organization,
                "bio": user.bio,
                "country": user.country,
                "language_preference": user.language_preference,
                "last_login": user.last_login,
                "created_at": user.created_at,
                "updated_at": user.updated_at
            })
        
        return {
            "items": users_data,
            "total": total,
            "skip": skip,
            "limit": limit
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get users error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve users"
        )


@router.get("/{user_id}")
async def get_user_by_id(
    user_id: int,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get user by ID (admin only)"""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "full_name": user.full_name,
            "role": user.role.value if hasattr(user.role, 'value') else str(user.role),
            "is_active": user.is_active,
            "is_verified": user.is_verified,
            "organization": user.organization,
            "bio": user.bio,
            "country": user.country,
            "language_preference": user.language_preference,
            "last_login": user.last_login,
            "created_at": user.created_at,
            "updated_at": user.updated_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user by ID error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user"
        )


@router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Delete user (admin only)"""
    try:
        # Prevent admin from deleting themselves
        if user_id == current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete your own account"
            )
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        db.delete(user)
        db.commit()
        
        return {"message": "User deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete user error: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        )