"""
Analytics API routes
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging

from app.config.database import get_db
from app.core.auth import auth_manager
from app.models.user import User
from app.models.conversation import Conversation, Message
from app.models.document import Document, DocumentStatus
from app.services.analytics_service import AnalyticsService

router = APIRouter()
security = HTTPBearer()
logger = logging.getLogger(__name__)

# Initialize analytics service
analytics_service = AnalyticsService()


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


@router.get("/overview")
async def get_analytics_overview(
    period: str = Query("30d", description="Time period: 1d, 7d, 30d, 90d, 1y"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get analytics overview for user"""
    try:
        overview = await analytics_service.get_user_overview(db, current_user.id, period)
        return overview
        
    except Exception as e:
        logger.error(f"Get analytics overview error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analytics overview"
        )


@router.get("/conversations")
async def get_conversation_analytics(
    period: str = Query("30d", description="Time period: 1d, 7d, 30d, 90d, 1y"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get conversation analytics"""
    try:
        analytics = await analytics_service.get_conversation_analytics(db, current_user.id, period)
        return analytics
        
    except Exception as e:
        logger.error(f"Get conversation analytics error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation analytics"
        )


@router.get("/messages")
async def get_message_analytics(
    period: str = Query("30d", description="Time period: 1d, 7d, 30d, 90d, 1y"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get message analytics"""
    try:
        analytics = await analytics_service.get_message_analytics(db, current_user.id, period)
        return analytics
        
    except Exception as e:
        logger.error(f"Get message analytics error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve message analytics"
        )


@router.get("/documents")
async def get_document_analytics(
    period: str = Query("30d", description="Time period: 1d, 7d, 30d, 90d, 1y"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get document analytics"""
    try:
        analytics = await analytics_service.get_document_analytics(db, current_user.id, period)
        return analytics
        
    except Exception as e:
        logger.error(f"Get document analytics error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document analytics"
        )


@router.get("/usage-patterns")
async def get_usage_patterns(
    period: str = Query("30d", description="Time period: 1d, 7d, 30d, 90d, 1y"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user usage patterns"""
    try:
        patterns = await analytics_service.get_usage_patterns(db, current_user.id, period)
        return patterns
        
    except Exception as e:
        logger.error(f"Get usage patterns error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve usage patterns"
        )


@router.get("/topics")
async def get_topic_analysis(
    period: str = Query("30d", description="Time period: 1d, 7d, 30d, 90d, 1y"),
    limit: int = Query(20, description="Maximum number of topics to return"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get topic analysis from user conversations"""
    try:
        topics = await analytics_service.get_topic_analysis(db, current_user.id, period, limit)
        return topics
        
    except Exception as e:
        logger.error(f"Get topic analysis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve topic analysis"
        )


@router.get("/performance")
async def get_performance_metrics(
    period: str = Query("30d", description="Time period: 1d, 7d, 30d, 90d, 1y"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get performance metrics"""
    try:
        metrics = await analytics_service.get_performance_metrics(db, current_user.id, period)
        return metrics
        
    except Exception as e:
        logger.error(f"Get performance metrics error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance metrics"
        )


@router.get("/feedback")
async def get_feedback_analytics(
    period: str = Query("30d", description="Time period: 1d, 7d, 30d, 90d, 1y"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get feedback analytics"""
    try:
        feedback = await analytics_service.get_feedback_analytics(db, current_user.id, period)
        return feedback
        
    except Exception as e:
        logger.error(f"Get feedback analytics error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve feedback analytics"
        )


@router.get("/trends")
async def get_usage_trends(
    metric: str = Query("messages", description="Metric to analyze: messages, conversations, documents"),
    period: str = Query("30d", description="Time period: 1d, 7d, 30d, 90d, 1y"),
    granularity: str = Query("day", description="Granularity: hour, day, week, month"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get usage trends over time"""
    try:
        trends = await analytics_service.get_usage_trends(
            db, current_user.id, metric, period, granularity
        )
        return trends
        
    except Exception as e:
        logger.error(f"Get usage trends error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve usage trends"
        )


@router.get("/export")
async def export_analytics(
    format: str = Query("json", description="Export format: json, csv"),
    period: str = Query("30d", description="Time period: 1d, 7d, 30d, 90d, 1y"),
    include: List[str] = Query(["overview", "conversations", "messages"], description="Data to include"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Export analytics data"""
    try:
        export_data = await analytics_service.export_analytics(
            db, current_user.id, format, period, include
        )
        return export_data
        
    except Exception as e:
        logger.error(f"Export analytics error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export analytics data"
        )


# Admin endpoints (for admin users only)
@router.get("/admin/system-overview")
async def get_system_analytics_overview(
    period: str = Query("30d", description="Time period: 1d, 7d, 30d, 90d, 1y"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get system-wide analytics overview (admin only)"""
    try:
        if current_user.role.value != "ADMIN":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        overview = await analytics_service.get_system_overview(db, period)
        return overview
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get system analytics overview error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system analytics overview"
        )


@router.get("/admin/users")
async def get_user_analytics(
    period: str = Query("30d", description="Time period: 1d, 7d, 30d, 90d, 1y"),
    limit: int = Query(50, description="Maximum number of users to return"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user analytics (admin only)"""
    try:
        # Check if user is admin - handle both string and enum
        user_role = current_user.role
        if hasattr(user_role, 'value'):
            role_value = user_role.value.lower()
        else:
            role_value = str(user_role).lower()
        
        if role_value != "admin":
            logger.warning(f"Non-admin user {current_user.id} attempted to access user analytics. Role: {user_role}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        user_analytics = await analytics_service.get_user_analytics(db, period, limit)
        return user_analytics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user analytics error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user analytics"
        )