"""
Recommendation service for providing personalized agricultural advice and content
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_
from collections import defaultdict, Counter
import re
import math

from app.models.user import User
from app.models.conversation import Conversation, Message
from app.models.document import Document, DocumentStatus

logger = logging.getLogger(__name__)


class RecommendationService:
    """Service for generating personalized recommendations"""
    
    def __init__(self):
        # Agricultural knowledge categories and keywords
        self.agriculture_categories = {
            'cereals': {
                'keywords': ['sorgho', 'mil', 'maïs', 'riz', 'blé', 'orge', 'avoine'],
                'topics': ['variétés', 'semis', 'fertilisation', 'récolte', 'stockage'],
                'seasons': ['hivernage', 'saison sèche']
            },
            'legumes': {
                'keywords': ['niébé', 'arachide', 'haricot', 'soja', 'lentille', 'pois'],
                'topics': ['inoculation', 'nodulation', 'fixation azote', 'rotation'],
                'seasons': ['saison des pluies', 'contre-saison']
            },
            'techniques': {
                'keywords': ['irrigation', 'labour', 'semis', 'désherbage', 'fertilisant'],
                'topics': ['mechanisation', 'conservation sols', 'agroécologie'],
                'seasons': ['préparation', 'entretien', 'post-récolte']
            },
            'protection': {
                'keywords': ['maladie', 'parasite', 'insecte', 'champignon', 'virus', 'traitement'],
                'topics': ['prévention', 'lutte biologique', 'pesticide', 'diagnostic'],
                'seasons': ['surveillance', 'traitement préventif']
            },
            'climate': {
                'keywords': ['pluie', 'sécheresse', 'température', 'humidité', 'vent'],
                'topics': ['adaptation climatique', 'gestion eau', 'prévisions'],
                'seasons': ['saison pluvieuse', 'saison sèche']
            }
        }
        
        # Seasonal recommendations
        self.seasonal_advice = {
            'hivernage': {
                'priority_crops': ['sorgho', 'mil', 'maïs', 'niébé'],
                'key_activities': ['semis', 'fertilisation', 'sarclage'],
                'risks': ['excès eau', 'maladies fongiques', 'parasites']
            },
            'saison_seche': {
                'priority_crops': ['cultures irriguées', 'maraîchage'],
                'key_activities': ['irrigation', 'récolte', 'stockage'],
                'risks': ['stress hydrique', 'ravageurs stockage']
            }
        }
    
    async def get_personalized_recommendations(
        self, 
        db: Session, 
        user_id: int, 
        limit: int = 10
    ) -> Dict[str, Any]:
        """Get personalized recommendations for user"""
        try:
            # Analyze user's conversation history
            user_profile = await self._build_user_profile(db, user_id)
            
            # Get content-based recommendations
            content_recommendations = await self._get_content_recommendations(
                db, user_id, user_profile, limit
            )
            
            # Get seasonal recommendations
            seasonal_recommendations = await self._get_seasonal_recommendations(
                user_profile, limit // 2
            )
            
            # Get similar user recommendations (collaborative filtering)
            collaborative_recommendations = await self._get_collaborative_recommendations(
                db, user_id, user_profile, limit // 2
            )
            
            # Get trending topics
            trending_recommendations = await self._get_trending_recommendations(
                db, user_profile, limit // 3
            )
            
            return {
                "user_profile": user_profile,
                "recommendations": {
                    "content_based": content_recommendations,
                    "seasonal": seasonal_recommendations,
                    "collaborative": collaborative_recommendations,
                    "trending": trending_recommendations
                },
                "next_actions": await self._suggest_next_actions(user_profile),
                "learning_resources": await self._suggest_learning_resources(user_profile)
            }
            
        except Exception as e:
            logger.error(f"Get personalized recommendations error: {str(e)}")
            raise
    
    async def _build_user_profile(self, db: Session, user_id: int) -> Dict[str, Any]:
        """Build user profile from conversation history"""
        try:
            # Get user messages from last 90 days
            cutoff_date = datetime.utcnow() - timedelta(days=90)
            
            messages = db.query(Message).join(Conversation).filter(
                Conversation.user_id == user_id,
                Message.created_at >= cutoff_date
            ).all()
            
            if not messages:
                return {
                    "interests": {},
                    "activity_level": "new",
                    "preferred_topics": [],
                    "conversation_patterns": {},
                    "expertise_level": "beginner"
                }
            
            # Extract interests from message content
            all_content = " ".join([m.content.lower() for m in messages if m.content])
            interests = self._extract_interests(all_content)
            
            # Analyze conversation patterns
            conversation_patterns = self._analyze_conversation_patterns(messages)
            
            # Determine expertise level
            expertise_level = self._estimate_expertise_level(messages, interests)
            
            # Get preferred topics
            preferred_topics = sorted(interests.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                "interests": interests,
                "activity_level": self._classify_activity_level(len(messages)),
                "preferred_topics": [topic for topic, score in preferred_topics],
                "conversation_patterns": conversation_patterns,
                "expertise_level": expertise_level,
                "total_messages": len(messages),
                "profile_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Build user profile error: {str(e)}")
            return {}
    
    def _extract_interests(self, content: str) -> Dict[str, float]:
        """Extract user interests from conversation content"""
        interests = defaultdict(float)
        
        # Analyze content for agricultural categories
        for category, category_data in self.agriculture_categories.items():
            category_score = 0
            
            # Check keywords
            for keyword in category_data['keywords']:
                count = content.count(keyword.lower())
                category_score += count * 2  # Keywords have higher weight
            
            # Check topics
            for topic in category_data['topics']:
                count = content.count(topic.lower())
                category_score += count * 1.5  # Topics have medium weight
            
            # Check seasons
            for season in category_data.get('seasons', []):
                count = content.count(season.lower())
                category_score += count * 1  # Seasons have lower weight
            
            if category_score > 0:
                interests[category] = category_score
        
        # Normalize scores
        if interests:
            max_score = max(interests.values())
            interests = {k: v / max_score for k, v in interests.items()}
        
        return dict(interests)
    
    def _analyze_conversation_patterns(self, messages: List[Message]) -> Dict[str, Any]:
        """Analyze user conversation patterns"""
        if not messages:
            return {}
        
        # Time patterns
        hours = [m.created_at.hour for m in messages]
        most_active_hour = Counter(hours).most_common(1)[0][0] if hours else 12
        
        # Message characteristics
        avg_message_length = sum(len(m.content) for m in messages if m.content) / len(messages)
        
        # Feedback patterns
        helpful_ratio = sum(1 for m in messages if m.is_helpful == True) / len(messages)
        
        return {
            "most_active_hour": most_active_hour,
            "average_message_length": round(avg_message_length, 2),
            "helpful_feedback_ratio": round(helpful_ratio, 2),
            "total_conversations": len(set(m.conversation_id for m in messages))
        }
    
    def _estimate_expertise_level(self, messages: List[Message], interests: Dict[str, float]) -> str:
        """Estimate user expertise level"""
        if not messages:
            return "beginner"
        
        # Factors for expertise estimation
        message_count = len(messages)
        avg_complexity = self._estimate_question_complexity(messages)
        interest_diversity = len(interests)
        
        # Simple scoring system
        score = 0
        if message_count > 50:
            score += 2
        elif message_count > 20:
            score += 1
        
        if avg_complexity > 0.7:
            score += 2
        elif avg_complexity > 0.4:
            score += 1
        
        if interest_diversity > 3:
            score += 1
        
        if score >= 4:
            return "expert"
        elif score >= 2:
            return "intermediate"
        else:
            return "beginner"
    
    def _estimate_question_complexity(self, messages: List[Message]) -> float:
        """Estimate average complexity of user questions"""
        if not messages:
            return 0.0
        
        complexity_indicators = [
            'comment', 'pourquoi', 'méthode', 'technique', 'optimiser', 
            'améliorer', 'problème', 'solution', 'stratégie', 'analyse'
        ]
        
        total_complexity = 0
        for message in messages:
            if not message.content:
                continue
            
            content_lower = message.content.lower()
            complexity_score = sum(1 for indicator in complexity_indicators if indicator in content_lower)
            # Normalize by message length
            complexity_score = complexity_score / max(len(content_lower.split()), 1)
            total_complexity += complexity_score
        
        return total_complexity / len(messages)
    
    def _classify_activity_level(self, message_count: int) -> str:
        """Classify user activity level"""
        if message_count == 0:
            return "new"
        elif message_count < 10:
            return "low"
        elif message_count < 50:
            return "medium"
        else:
            return "high"
    
    async def _get_content_recommendations(
        self, 
        db: Session, 
        user_id: int, 
        user_profile: Dict[str, Any], 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Get content-based recommendations"""
        try:
            recommendations = []
            interests = user_profile.get("interests", {})
            expertise_level = user_profile.get("expertise_level", "beginner")
            
            # Recommend based on user interests
            for category, score in sorted(interests.items(), key=lambda x: x[1], reverse=True):
                if len(recommendations) >= limit:
                    break
                
                category_data = self.agriculture_categories.get(category, {})
                
                # Generate recommendations for this category
                for topic in category_data.get('topics', []):
                    if len(recommendations) >= limit:
                        break
                    
                    recommendation = {
                        "type": "content",
                        "category": category,
                        "title": f"Guide sur {topic} pour {category}",
                        "description": f"Approfondissez vos connaissances sur {topic} dans le domaine {category}",
                        "relevance_score": score,
                        "difficulty": expertise_level,
                        "estimated_read_time": "5-10 min",
                        "action_type": "learn"
                    }
                    recommendations.append(recommendation)
            
            # Fill remaining slots with general recommendations
            while len(recommendations) < limit:
                general_topics = [
                    ("Techniques de conservation des sols", "techniques", 0.8),
                    ("Gestion intégrée des parasites", "protection", 0.7),
                    ("Optimisation de l'irrigation", "techniques", 0.75),
                    ("Variétés améliorées de céréales", "cereals", 0.9),
                    ("Agriculture climatiquement intelligente", "climate", 0.85)
                ]
                
                for title, category, score in general_topics:
                    if len(recommendations) >= limit:
                        break
                    
                    recommendation = {
                        "type": "content",
                        "category": category,
                        "title": title,
                        "description": f"Découvrez les meilleures pratiques en {title.lower()}",
                        "relevance_score": score,
                        "difficulty": expertise_level,
                        "estimated_read_time": "8-15 min",
                        "action_type": "learn"
                    }
                    recommendations.append(recommendation)
            
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Get content recommendations error: {str(e)}")
            return []
    
    async def _get_seasonal_recommendations(
        self, 
        user_profile: Dict[str, Any], 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Get seasonal recommendations"""
        try:
            recommendations = []
            current_month = datetime.utcnow().month
            
            # Determine season (simplified for West Africa)
            if 6 <= current_month <= 10:  # Rainy season
                season = "hivernage"
                season_name = "Hivernage"
            else:  # Dry season
                season = "saison_seche"
                season_name = "Saison sèche"
            
            seasonal_data = self.seasonal_advice.get(season, {})
            interests = user_profile.get("interests", {})
            
            # Priority crops recommendations
            for crop in seasonal_data.get('priority_crops', []):
                if len(recommendations) >= limit:
                    break
                
                recommendation = {
                    "type": "seasonal",
                    "season": season_name,
                    "title": f"Culture de {crop} en {season_name.lower()}",
                    "description": f"Conseils spécialisés pour optimiser la culture de {crop} pendant {season_name.lower()}",
                    "urgency": "high",
                    "timeframe": "actuel",
                    "action_type": "implement"
                }
                recommendations.append(recommendation)
            
            # Key activities recommendations
            for activity in seasonal_data.get('key_activities', []):
                if len(recommendations) >= limit:
                    break
                
                recommendation = {
                    "type": "seasonal",
                    "season": season_name,
                    "title": f"Optimiser {activity} en {season_name.lower()}",
                    "description": f"Meilleures pratiques pour {activity} adaptées à {season_name.lower()}",
                    "urgency": "medium",
                    "timeframe": "2-4 semaines",
                    "action_type": "plan"
                }
                recommendations.append(recommendation)
            
            # Risk management recommendations
            for risk in seasonal_data.get('risks', []):
                if len(recommendations) >= limit:
                    break
                
                recommendation = {
                    "type": "seasonal",
                    "season": season_name,
                    "title": f"Prévention: {risk}",
                    "description": f"Stratégies de prévention et gestion des risques liés à {risk}",
                    "urgency": "high",
                    "timeframe": "immédiat",
                    "action_type": "prevent"
                }
                recommendations.append(recommendation)
            
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Get seasonal recommendations error: {str(e)}")
            return []
    
    async def _get_collaborative_recommendations(
        self, 
        db: Session, 
        user_id: int, 
        user_profile: Dict[str, Any], 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Get collaborative filtering recommendations"""
        try:
            recommendations = []
            user_interests = user_profile.get("interests", {})
            
            # Find similar users
            similar_users = await self._find_similar_users(db, user_id, user_interests)
            
            if not similar_users:
                return []
            
            # Get topics popular among similar users
            popular_topics = await self._get_popular_topics_among_users(db, similar_users)
            
            for topic, popularity_score in popular_topics[:limit]:
                recommendation = {
                    "type": "collaborative",
                    "title": f"Sujet populaire: {topic}",
                    "description": f"Ce sujet intéresse des utilisateurs avec des intérêts similaires aux vôtres",
                    "popularity_score": popularity_score,
                    "similar_users_count": len(similar_users),
                    "action_type": "explore"
                }
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Get collaborative recommendations error: {str(e)}")
            return []
    
    async def _find_similar_users(
        self, 
        db: Session, 
        user_id: int, 
        user_interests: Dict[str, float]
    ) -> List[int]:
        """Find users with similar interests"""
        try:
            # This is a simplified approach
            # In a real system, you'd use more sophisticated similarity metrics
            
            # Get all users who have had conversations
            active_users = db.query(func.distinct(Conversation.user_id)).filter(
                Conversation.user_id != user_id
            ).all()
            
            similar_users = []
            
            for (other_user_id,) in active_users:
                # Build profile for other user (simplified)
                other_messages = db.query(Message).join(Conversation).filter(
                    Conversation.user_id == other_user_id
                ).limit(50).all()  # Limit for performance
                
                if not other_messages:
                    continue
                
                other_content = " ".join([m.content.lower() for m in other_messages if m.content])
                other_interests = self._extract_interests(other_content)
                
                # Calculate similarity score
                similarity = self._calculate_interest_similarity(user_interests, other_interests)
                
                if similarity > 0.3:  # Threshold for similarity
                    similar_users.append(other_user_id)
                
                if len(similar_users) >= 10:  # Limit similar users
                    break
            
            return similar_users
            
        except Exception as e:
            logger.error(f"Find similar users error: {str(e)}")
            return []
    
    def _calculate_interest_similarity(
        self, 
        interests1: Dict[str, float], 
        interests2: Dict[str, float]
    ) -> float:
        """Calculate cosine similarity between two interest profiles"""
        if not interests1 or not interests2:
            return 0.0
        
        # Get common categories
        common_categories = set(interests1.keys()) & set(interests2.keys())
        
        if not common_categories:
            return 0.0
        
        # Calculate cosine similarity
        dot_product = sum(interests1[cat] * interests2[cat] for cat in common_categories)
        norm1 = math.sqrt(sum(v * v for v in interests1.values()))
        norm2 = math.sqrt(sum(v * v for v in interests2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def _get_popular_topics_among_users(
        self, 
        db: Session, 
        user_ids: List[int]
    ) -> List[Tuple[str, float]]:
        """Get topics popular among similar users"""
        try:
            if not user_ids:
                return []
            
            # Get messages from similar users
            messages = db.query(Message).join(Conversation).filter(
                Conversation.user_id.in_(user_ids)
            ).limit(500).all()  # Limit for performance
            
            if not messages:
                return []
            
            # Extract topics from their conversations
            all_content = " ".join([m.content.lower() for m in messages if m.content])
            topic_counts = Counter()
            
            # Simple topic extraction
            words = re.findall(r'\b[a-zA-Z]{4,}\b', all_content)
            agricultural_words = []
            
            for category_data in self.agriculture_categories.values():
                agricultural_words.extend(category_data['keywords'])
                agricultural_words.extend(category_data['topics'])
            
            for word in words:
                if word in agricultural_words:
                    topic_counts[word] += 1
            
            # Calculate popularity scores
            total_messages = len(messages)
            popular_topics = [
                (topic, count / total_messages)
                for topic, count in topic_counts.most_common(10)
            ]
            
            return popular_topics
            
        except Exception as e:
            logger.error(f"Get popular topics error: {str(e)}")
            return []
    
    async def _get_trending_recommendations(
        self, 
        db: Session, 
        user_profile: Dict[str, Any], 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Get trending topic recommendations"""
        try:
            recommendations = []
            
            # Get recent popular topics (last 30 days)
            cutoff_date = datetime.utcnow() - timedelta(days=30)
            
            recent_messages = db.query(Message).join(Conversation).filter(
                Message.created_at >= cutoff_date
            ).limit(1000).all()  # Limit for performance
            
            if not recent_messages:
                return []
            
            # Extract trending topics
            all_content = " ".join([m.content.lower() for m in recent_messages if m.content])
            trending_topics = self._extract_trending_topics(all_content)
            
            for topic, trend_score in trending_topics[:limit]:
                recommendation = {
                    "type": "trending",
                    "title": f"Sujet tendance: {topic}",
                    "description": f"Ce sujet génère beaucoup d'intérêt récemment dans la communauté",
                    "trend_score": trend_score,
                    "timeframe": "30 derniers jours",
                    "action_type": "explore"
                }
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Get trending recommendations error: {str(e)}")
            return []
    
    def _extract_trending_topics(self, content: str) -> List[Tuple[str, float]]:
        """Extract trending topics from recent content"""
        topic_counts = Counter()
        
        # Extract agricultural terms
        for category_data in self.agriculture_categories.values():
            for keyword in category_data['keywords']:
                count = content.count(keyword.lower())
                if count > 0:
                    topic_counts[keyword] += count
            
            for topic in category_data['topics']:
                count = content.count(topic.lower())
                if count > 0:
                    topic_counts[topic] += count
        
        # Calculate trend scores (simplified)
        total_count = sum(topic_counts.values())
        if total_count == 0:
            return []
        
        trending_topics = [
            (topic, count / total_count)
            for topic, count in topic_counts.most_common(10)
            if count >= 3  # Minimum threshold
        ]
        
        return trending_topics
    
    async def _suggest_next_actions(self, user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest next actions for the user"""
        actions = []
        expertise_level = user_profile.get("expertise_level", "beginner")
        interests = user_profile.get("interests", {})
        
        if expertise_level == "beginner":
            actions.extend([
                {
                    "action": "complete_profile",
                    "title": "Complétez votre profil agricole",
                    "description": "Aidez-nous à mieux vous conseiller en précisant vos cultures et régions",
                    "priority": "high"
                },
                {
                    "action": "explore_basics",
                    "title": "Explorez les bases de l'agriculture",
                    "description": "Découvrez les fondamentaux adaptés à votre région",
                    "priority": "medium"
                }
            ])
        
        elif expertise_level == "intermediate":
            actions.extend([
                {
                    "action": "advanced_techniques",
                    "title": "Découvrez des techniques avancées",
                    "description": "Approfondissez vos connaissances avec des méthodes spécialisées",
                    "priority": "medium"
                },
                {
                    "action": "connect_experts",
                    "title": "Connectez-vous avec des experts",
                    "description": "Échangez avec des professionnels de votre domaine",
                    "priority": "low"
                }
            ])
        
        else:  # expert
            actions.extend([
                {
                    "action": "share_knowledge",
                    "title": "Partagez votre expertise",
                    "description": "Aidez d'autres agriculteurs en partageant vos expériences",
                    "priority": "medium"
                },
                {
                    "action": "latest_research",
                    "title": "Suivez les dernières recherches",
                    "description": "Restez à jour avec les innovations agricoles",
                    "priority": "high"
                }
            ])
        
        # Add interest-specific actions
        if interests:
            top_interest = max(interests.items(), key=lambda x: x[1])[0]
            actions.append({
                "action": "deepen_interest",
                "title": f"Approfondissez vos connaissances en {top_interest}",
                "description": f"Explorez des ressources spécialisées sur {top_interest}",
                "priority": "high"
            })
        
        return actions[:5]  # Limit to 5 actions
    
    async def _suggest_learning_resources(self, user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest learning resources"""
        resources = []
        interests = user_profile.get("interests", {})
        expertise_level = user_profile.get("expertise_level", "beginner")
        
        # Base resources for all users
        base_resources = [
            {
                "type": "guide",
                "title": "Guide pratique de l'agriculture sahélienne",
                "description": "Manuel complet adapté aux conditions locales",
                "difficulty": "beginner",
                "estimated_time": "2-3 heures"
            },
            {
                "type": "video",
                "title": "Techniques de conservation des sols",
                "description": "Démonstrations pratiques des meilleures techniques",
                "difficulty": "intermediate",
                "estimated_time": "45 minutes"
            }
        ]
        
        resources.extend(base_resources)
        
        # Interest-specific resources
        for interest in list(interests.keys())[:3]:  # Top 3 interests
            interest_resource = {
                "type": "specialized_guide",
                "title": f"Guide spécialisé: {interest}",
                "description": f"Ressources approfondies sur {interest}",
                "difficulty": expertise_level,
                "estimated_time": "1-2 heures",
                "relevance": interests[interest]
            }
            resources.append(interest_resource)
        
        return resources[:6]  # Limit to 6 resources
    
    async def get_quick_recommendations(
        self, 
        db: Session, 
        user_id: int, 
        context: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get quick recommendations based on current context"""
        try:
            recommendations = []
            
            # Analyze context for immediate recommendations
            context_lower = context.lower()
            
            # Emergency/urgent recommendations
            urgent_keywords = ['problème', 'urgent', 'maladie', 'parasite', 'aide']
            if any(keyword in context_lower for keyword in urgent_keywords):
                recommendations.append({
                    "type": "urgent",
                    "title": "Diagnostic rapide de problème",
                    "description": "Outils pour identifier rapidement les problèmes agricoles",
                    "urgency": "immediate",
                    "action_type": "diagnose"
                })
            
            # Context-specific recommendations
            for category, category_data in self.agriculture_categories.items():
                if any(keyword in context_lower for keyword in category_data['keywords']):
                    recommendations.append({
                        "type": "contextual",
                        "category": category,
                        "title": f"Conseils sur {category}",
                        "description": f"Recommandations spécialisées basées sur votre question sur {category}",
                        "relevance": "high",
                        "action_type": "learn"
                    })
                    
                    if len(recommendations) >= limit:
                        break
            
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Get quick recommendations error: {str(e)}")
            return []