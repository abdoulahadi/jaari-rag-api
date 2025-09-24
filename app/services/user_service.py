"""
User service for business logic
"""
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from typing import Optional, List
from datetime import datetime, timedelta
import logging

from app.models.user import User, UserRole
from app.schemas.user import UserCreate, UserUpdate, UserUpdateAdmin
from app.core.auth import auth_manager

logger = logging.getLogger(__name__)


class UserService:
    """User business logic service"""
    
    async def create_default_admin(self, db: Session, admin_data: dict) -> Optional[User]:
        """Create default admin user if no admin exists"""
        try:
            # Check if any admin already exists
            existing_admin = db.query(User).filter(User.role == UserRole.ADMIN).first()
            if existing_admin:
                logger.info("Admin user already exists, skipping creation")
                return existing_admin
            
            # Check if email or username already exists (with different role)
            existing_email = await self.get_by_email(db, admin_data["email"])
            if existing_email:
                logger.warning(f"User with email {admin_data['email']} already exists but is not admin. Promoting to admin.")
                existing_email.role = UserRole.ADMIN
                existing_email.is_active = True
                existing_email.is_verified = True
                db.commit()
                db.refresh(existing_email)
                return existing_email
            
            existing_username = await self.get_by_username(db, admin_data["username"])
            if existing_username:
                logger.warning(f"User with username {admin_data['username']} already exists but is not admin. Promoting to admin.")
                existing_username.role = UserRole.ADMIN
                existing_username.is_active = True
                existing_username.is_verified = True
                db.commit()
                db.refresh(existing_username)
                return existing_username
            
            # Hash password
            hashed_password = auth_manager.hash_password(admin_data["password"])
            
            # Create admin user
            admin_user = User(
                email=admin_data["email"],
                username=admin_data["username"],
                hashed_password=hashed_password,
                full_name=admin_data.get("full_name", "Administrator"),
                role=UserRole.ADMIN,
                is_active=True,
                is_verified=True,
                language_preference="fr",
                api_key=auth_manager.generate_api_key()
            )
            
            db.add(admin_user)
            db.commit()
            db.refresh(admin_user)
            
            logger.info(f"Default admin user created: {admin_user.email}")
            return admin_user
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create default admin: {str(e)}")
            raise
    
    async def ensure_admin_exists(self, db: Session) -> bool:
        """Ensure at least one admin user exists in the system"""
        try:
            admin_count = db.query(User).filter(User.role == UserRole.ADMIN).count()
            return admin_count > 0
        except Exception:
            return False
    
    async def create(self, db: Session, user_data: UserCreate) -> User:
        """Create a new user"""
        try:
            # Hash password
            hashed_password = auth_manager.hash_password(user_data.password)
            
            # Create user object
            db_user = User(
                email=user_data.email,
                username=user_data.username,
                hashed_password=hashed_password,
                full_name=user_data.full_name,
                organization=user_data.organization,
                bio=user_data.bio,
                country=user_data.country,
                language_preference=user_data.language_preference,
                role=UserRole.USER,  # Default role
                api_key=auth_manager.generate_api_key()
            )
            
            db.add(db_user)
            db.commit()
            db.refresh(db_user)
            
            logger.info(f"User created: {db_user.email}")
            return db_user
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create user: {str(e)}")
            raise
    
    async def get_by_id(self, db: Session, user_id: int) -> Optional[User]:
        """Get user by ID"""
        return db.query(User).filter(User.id == user_id).first()
    
    async def get_by_email(self, db: Session, email: str) -> Optional[User]:
        """Get user by email"""
        return db.query(User).filter(User.email == email).first()
    
    async def get_by_username(self, db: Session, username: str) -> Optional[User]:
        """Get user by username"""
        return db.query(User).filter(User.username == username).first()
    
    async def get_by_api_key(self, db: Session, api_key: str) -> Optional[User]:
        """Get user by API key"""
        return db.query(User).filter(User.api_key == api_key).first()
    
    async def get_all(
        self, 
        db: Session, 
        skip: int = 0, 
        limit: int = 100,
        role: Optional[UserRole] = None,
        is_active: Optional[bool] = None
    ) -> List[User]:
        """Get all users with filters"""
        query = db.query(User)
        
        if role:
            query = query.filter(User.role == role)
        
        if is_active is not None:
            query = query.filter(User.is_active == is_active)
        
        return query.offset(skip).limit(limit).all()
    
    async def update(self, db: Session, user_id: int, user_data: UserUpdate) -> Optional[User]:
        """Update user profile"""
        try:
            db_user = await self.get_by_id(db, user_id)
            if not db_user:
                return None
            
            # Update fields
            update_data = user_data.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(db_user, field, value)
            
            db.commit()
            db.refresh(db_user)
            
            logger.info(f"User updated: {db_user.email}")
            return db_user
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update user: {str(e)}")
            raise
    
    async def update_admin(self, db: Session, user_id: int, user_data: UserUpdateAdmin) -> Optional[User]:
        """Update user (admin only - can change sensitive fields)"""
        try:
            db_user = await self.get_by_id(db, user_id)
            if not db_user:
                return None
            
            # Update fields
            update_data = user_data.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(db_user, field, value)
            
            db.commit()
            db.refresh(db_user)
            
            logger.info(f"User updated by admin: {db_user.email}")
            return db_user
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update user (admin): {str(e)}")
            raise
    
    async def delete(self, db: Session, user_id: int) -> bool:
        """Delete user"""
        try:
            db_user = await self.get_by_id(db, user_id)
            if not db_user:
                return False
            
            db.delete(db_user)
            db.commit()
            
            logger.info(f"User deleted: {db_user.email}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete user: {str(e)}")
            raise
    
    async def deactivate(self, db: Session, user_id: int) -> Optional[User]:
        """Deactivate user account"""
        try:
            db_user = await self.get_by_id(db, user_id)
            if not db_user:
                return None
            
            db_user.is_active = False
            db_user.refresh_token_jti = None  # Invalidate refresh token
            
            db.commit()
            db.refresh(db_user)
            
            logger.info(f"User deactivated: {db_user.email}")
            return db_user
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to deactivate user: {str(e)}")
            raise
    
    async def activate(self, db: Session, user_id: int) -> Optional[User]:
        """Activate user account"""
        try:
            db_user = await self.get_by_id(db, user_id)
            if not db_user:
                return None
            
            db_user.is_active = True
            db_user.failed_login_attempts = 0
            db_user.locked_until = None
            
            db.commit()
            db.refresh(db_user)
            
            logger.info(f"User activated: {db_user.email}")
            return db_user
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to activate user: {str(e)}")
            raise
    
    async def increment_failed_attempts(self, db: Session, user: User) -> None:
        """Increment failed login attempts"""
        try:
            user.failed_login_attempts += 1
            
            # Lock user after 5 failed attempts for 30 minutes
            if user.failed_login_attempts >= 5:
                user.locked_until = datetime.utcnow() + timedelta(minutes=30)
                logger.warning(f"User locked due to failed attempts: {user.email}")
            
            db.commit()
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to increment failed attempts: {str(e)}")
            raise
    
    async def reset_failed_attempts(self, db: Session, user: User) -> None:
        """Reset failed login attempts"""
        try:
            user.failed_login_attempts = 0
            user.locked_until = None
            db.commit()
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to reset failed attempts: {str(e)}")
            raise
    
    async def change_password(self, db: Session, user_id: int, new_password: str) -> Optional[User]:
        """Change user password"""
        try:
            db_user = await self.get_by_id(db, user_id)
            if not db_user:
                return None
            
            # Hash new password
            db_user.hashed_password = auth_manager.hash_password(new_password)
            
            # Clear refresh token to force re-login
            db_user.refresh_token_jti = None
            
            db.commit()
            db.refresh(db_user)
            
            logger.info(f"Password changed for user: {db_user.email}")
            return db_user
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to change password: {str(e)}")
            raise
    
    async def regenerate_api_key(self, db: Session, user_id: int) -> Optional[User]:
        """Regenerate user API key"""
        try:
            db_user = await self.get_by_id(db, user_id)
            if not db_user:
                return None
            
            db_user.api_key = auth_manager.generate_api_key()
            
            db.commit()
            db.refresh(db_user)
            
            logger.info(f"API key regenerated for user: {db_user.email}")
            return db_user
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to regenerate API key: {str(e)}")
            raise
    
    async def search_users(
        self, 
        db: Session, 
        query: str, 
        skip: int = 0, 
        limit: int = 20
    ) -> List[User]:
        """Search users by username, email, or full name"""
        search_filter = or_(
            User.username.ilike(f"%{query}%"),
            User.email.ilike(f"%{query}%"),
            User.full_name.ilike(f"%{query}%")
        )
        
        return db.query(User).filter(search_filter).offset(skip).limit(limit).all()
    
    async def get_user_count(self, db: Session) -> int:
        """Get total user count"""
        return db.query(User).count()
    
    async def get_active_user_count(self, db: Session) -> int:
        """Get active user count"""
        return db.query(User).filter(User.is_active == True).count()
