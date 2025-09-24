"""
Recommendation API routes
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import List, Optional
import logging

from app.config.database import get_db
from app.core.auth import auth_manager
from app.models.user import User
from app.services.recommendation_service import RecommendationService

router = APIRouter()
security = HTTPBearer()
logger = logging.getLogger(__name__)

# Initialize recommendation service
recommendation_service = RecommendationService()


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


@router.get("/personalized")
async def get_personalized_recommendations(
    limit: int = Query(10, description="Maximum number of recommendations"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get personalized recommendations for user"""
    try:
        recommendations = await recommendation_service.get_personalized_recommendations(
            db, current_user.id, limit
        )
        return recommendations
        
    except Exception as e:
        logger.error(f"Get personalized recommendations error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve personalized recommendations"
        )


@router.get("/quick")
async def get_quick_recommendations(
    context: str = Query(..., description="Current context or topic"),
    limit: int = Query(5, description="Maximum number of recommendations"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get quick recommendations based on context"""
    try:
        recommendations = await recommendation_service.get_quick_recommendations(
            db, current_user.id, context, limit
        )
        return {
            "context": context,
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Get quick recommendations error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve quick recommendations"
        )


@router.get("/seasonal")
async def get_seasonal_recommendations(
    limit: int = Query(5, description="Maximum number of recommendations"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get seasonal recommendations"""
    try:
        # Build basic user profile for seasonal recommendations
        user_profile = await recommendation_service._build_user_profile(db, current_user.id)
        
        seasonal_recommendations = await recommendation_service._get_seasonal_recommendations(
            user_profile, limit
        )
        
        return {
            "seasonal_recommendations": seasonal_recommendations,
            "user_season_profile": {
                "expertise_level": user_profile.get("expertise_level", "beginner"),
                "interests": user_profile.get("interests", {})
            }
        }
        
    except Exception as e:
        logger.error(f"Get seasonal recommendations error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve seasonal recommendations"
        )


@router.get("/trending")
async def get_trending_recommendations(
    limit: int = Query(5, description="Maximum number of recommendations"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get trending topic recommendations"""
    try:
        user_profile = await recommendation_service._build_user_profile(db, current_user.id)
        
        trending_recommendations = await recommendation_service._get_trending_recommendations(
            db, user_profile, limit
        )
        
        return {
            "trending_recommendations": trending_recommendations,
            "analysis_period": "30 derniers jours"
        }
        
    except Exception as e:
        logger.error(f"Get trending recommendations error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve trending recommendations"
        )


@router.get("/next-actions")
async def get_next_actions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get suggested next actions for user"""
    try:
        user_profile = await recommendation_service._build_user_profile(db, current_user.id)
        
        next_actions = await recommendation_service._suggest_next_actions(user_profile)
        
        return {
            "next_actions": next_actions,
            "user_profile_summary": {
                "activity_level": user_profile.get("activity_level", "new"),
                "expertise_level": user_profile.get("expertise_level", "beginner"),
                "top_interests": user_profile.get("preferred_topics", [])[:3]
            }
        }
        
    except Exception as e:
        logger.error(f"Get next actions error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve next actions"
        )


@router.get("/learning-resources")
async def get_learning_resources(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get suggested learning resources"""
    try:
        user_profile = await recommendation_service._build_user_profile(db, current_user.id)
        
        learning_resources = await recommendation_service._suggest_learning_resources(user_profile)
        
        return {
            "learning_resources": learning_resources,
            "personalized_for": {
                "expertise_level": user_profile.get("expertise_level", "beginner"),
                "interests": list(user_profile.get("interests", {}).keys())[:5]
            }
        }
        
    except Exception as e:
        logger.error(f"Get learning resources error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve learning resources"
        )


@router.get("/profile")
async def get_user_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's recommendation profile"""
    try:
        user_profile = await recommendation_service._build_user_profile(db, current_user.id)
        
        return {
            "user_profile": user_profile,
            "profile_completeness": _calculate_profile_completeness(user_profile),
            "last_updated": user_profile.get("profile_updated")
        }
        
    except Exception as e:
        logger.error(f"Get user profile error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user profile"
        )


def _calculate_profile_completeness(user_profile: dict) -> dict:
    """Calculate how complete the user profile is"""
    completeness_score = 0
    max_score = 5
    
    # Check different aspects of profile completeness
    if user_profile.get("total_messages", 0) > 0:
        completeness_score += 1
    
    if user_profile.get("interests"):
        completeness_score += 1
    
    if user_profile.get("expertise_level") != "beginner":
        completeness_score += 1
    
    if user_profile.get("conversation_patterns"):
        completeness_score += 1
    
    if len(user_profile.get("preferred_topics", [])) >= 3:
        completeness_score += 1
    
    percentage = (completeness_score / max_score) * 100
    
    return {
        "score": completeness_score,
        "max_score": max_score,
        "percentage": round(percentage, 1),
        "status": "complete" if percentage >= 80 else "partial" if percentage >= 40 else "minimal"
    }


@router.post("/feedback")
async def provide_recommendation_feedback(
    recommendation_id: str,
    feedback: str = Query(..., description="Feedback: helpful, not_helpful, irrelevant"),
    comment: Optional[str] = Query(None, description="Additional comment"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Provide feedback on a recommendation"""
    try:
        # In a real implementation, you would store this feedback
        # to improve the recommendation algorithm
        
        logger.info(f"Recommendation feedback from user {current_user.id}: "
                   f"recommendation_id={recommendation_id}, feedback={feedback}, comment={comment}")
        
        return {
            "message": "Feedback received successfully",
            "recommendation_id": recommendation_id,
            "feedback": feedback,
            "will_improve_future_recommendations": True
        }
        
    except Exception as e:
        logger.error(f"Provide recommendation feedback error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record feedback"
        )


@router.get("/categories")
async def get_recommendation_categories():
    """Get available recommendation categories"""
    try:
        categories = {
            "agriculture_categories": recommendation_service.agriculture_categories,
            "seasonal_advice": recommendation_service.seasonal_advice,
            "recommendation_types": [
                "content_based",
                "seasonal", 
                "collaborative",
                "trending",
                "contextual",
                "urgent"
            ]
        }
        
        return categories
        
    except Exception as e:
        logger.error(f"Get recommendation categories error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve recommendation categories"
        )