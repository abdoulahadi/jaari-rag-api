"""
Analytics service for conversation and usage analytics
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_, text
from collections import defaultdict, Counter
import json
import csv
import io
import re

from app.models.user import User
from app.models.conversation import Conversation, Message
from app.models.document import Document, DocumentStatus

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Service for generating analytics and insights"""
    
    def __init__(self):
        self.period_mapping = {
            "1d": 1,
            "7d": 7,
            "30d": 30,
            "90d": 90,
            "1y": 365
        }
    
    def _get_date_range(self, period: str) -> tuple[datetime, datetime]:
        """Get date range for analytics period"""
        days = self.period_mapping.get(period, 30)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        return start_date, end_date
    
    async def get_user_overview(self, db: Session, user_id: int, period: str) -> Dict[str, Any]:
        """Get user analytics overview"""
        try:
            start_date, end_date = self._get_date_range(period)
            
            # Get basic counts
            total_conversations = db.query(Conversation).filter(
                Conversation.user_id == user_id,
                Conversation.created_at >= start_date
            ).count()
            
            total_messages = db.query(Message).join(Conversation).filter(
                Conversation.user_id == user_id,
                Message.created_at >= start_date
            ).count()
            
            total_documents = db.query(Document).filter(
                Document.uploaded_by_id == user_id,
                Document.created_at >= start_date
            ).count()
            
            # Average messages per conversation
            avg_messages_per_conv = (
                db.query(func.avg(Conversation.total_messages))
                .filter(
                    Conversation.user_id == user_id,
                    Conversation.created_at >= start_date
                ).scalar() or 0
            )
            
            # Get feedback metrics
            helpful_messages = db.query(Message).join(Conversation).filter(
                Conversation.user_id == user_id,
                Message.is_helpful == True,
                Message.created_at >= start_date
            ).count()
            
            total_feedback = db.query(Message).join(Conversation).filter(
                Conversation.user_id == user_id,
                Message.is_helpful.isnot(None),
                Message.created_at >= start_date
            ).count()
            
            satisfaction_rate = (helpful_messages / total_feedback * 100) if total_feedback > 0 else 0
            
            # Activity level
            active_days = db.query(
                func.count(func.distinct(func.date(Message.created_at)))
            ).join(Conversation).filter(
                Conversation.user_id == user_id,
                Message.created_at >= start_date
            ).scalar() or 0
            
            return {
                "period": period,
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "totals": {
                    "conversations": total_conversations,
                    "messages": total_messages,
                    "documents": total_documents
                },
                "averages": {
                    "messages_per_conversation": round(float(avg_messages_per_conv), 2),
                    "conversations_per_day": round(total_conversations / self.period_mapping[period], 2)
                },
                "engagement": {
                    "active_days": active_days,
                    "activity_rate": round(active_days / self.period_mapping[period] * 100, 2),
                    "satisfaction_rate": round(satisfaction_rate, 2),
                    "total_feedback_received": total_feedback
                }
            }
            
        except Exception as e:
            logger.error(f"Get user overview error: {str(e)}")
            raise
    
    async def get_conversation_analytics(self, db: Session, user_id: int, period: str) -> Dict[str, Any]:
        """Get detailed conversation analytics"""
        try:
            start_date, end_date = self._get_date_range(period)
            
            # Get conversations with message counts
            conversations = db.query(
                Conversation.id,
                Conversation.title,
                Conversation.total_messages,
                Conversation.created_at,
                Conversation.is_active
            ).filter(
                Conversation.user_id == user_id,
                Conversation.created_at >= start_date
            ).order_by(desc(Conversation.created_at)).all()
            
            # Most active conversations
            most_active = sorted(conversations, key=lambda x: x.total_messages, reverse=True)[:10]
            
            # Conversation length distribution
            length_distribution = defaultdict(int)
            for conv in conversations:
                if conv.total_messages <= 5:
                    length_distribution["short (1-5)"] += 1
                elif conv.total_messages <= 15:
                    length_distribution["medium (6-15)"] += 1
                else:
                    length_distribution["long (16+)"] += 1
            
            # Daily conversation creation
            daily_creation = defaultdict(int)
            for conv in conversations:
                date_key = conv.created_at.date().isoformat()
                daily_creation[date_key] += 1
            
            return {
                "summary": {
                    "total_conversations": len(conversations),
                    "active_conversations": sum(1 for c in conversations if c.is_active),
                    "average_length": round(sum(c.total_messages for c in conversations) / len(conversations), 2) if conversations else 0
                },
                "most_active_conversations": [
                    {
                        "id": conv.id,
                        "title": conv.title,
                        "message_count": conv.total_messages,
                        "created_at": conv.created_at.isoformat()
                    }
                    for conv in most_active
                ],
                "length_distribution": dict(length_distribution),
                "daily_creation": dict(daily_creation)
            }
            
        except Exception as e:
            logger.error(f"Get conversation analytics error: {str(e)}")
            raise
    
    async def get_message_analytics(self, db: Session, user_id: int, period: str) -> Dict[str, Any]:
        """Get message analytics"""
        try:
            start_date, end_date = self._get_date_range(period)
            
            # Get messages with details
            messages = db.query(Message).join(Conversation).filter(
                Conversation.user_id == user_id,
                Message.created_at >= start_date
            ).all()
            
            if not messages:
                return {
                    "summary": {"total_messages": 0},
                    "response_times": {},
                    "feedback_distribution": {},
                    "hourly_distribution": {},
                    "token_usage": {}
                }
            
            # Response time analysis
            response_times = [m.response_time for m in messages if m.response_time]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            # Feedback distribution
            feedback_dist = {
                "helpful": sum(1 for m in messages if m.is_helpful == True),
                "not_helpful": sum(1 for m in messages if m.is_helpful == False),
                "no_feedback": sum(1 for m in messages if m.is_helpful is None)
            }
            
            # Hourly message distribution
            hourly_dist = defaultdict(int)
            for message in messages:
                hour = message.created_at.hour
                hourly_dist[f"{hour:02d}:00"] += 1
            
            # Token usage analysis
            token_usage = [m.tokens_used for m in messages if m.tokens_used]
            avg_tokens = sum(token_usage) / len(token_usage) if token_usage else 0
            total_tokens = sum(token_usage) if token_usage else 0
            
            return {
                "summary": {
                    "total_messages": len(messages),
                    "messages_with_feedback": feedback_dist["helpful"] + feedback_dist["not_helpful"],
                    "average_response_time": round(avg_response_time, 2),
                    "total_tokens_used": total_tokens
                },
                "response_times": {
                    "average": round(avg_response_time, 2),
                    "min": min(response_times) if response_times else 0,
                    "max": max(response_times) if response_times else 0
                },
                "feedback_distribution": feedback_dist,
                "hourly_distribution": dict(sorted(hourly_dist.items())),
                "token_usage": {
                    "total": total_tokens,
                    "average_per_message": round(avg_tokens, 2),
                    "max_tokens_in_message": max(token_usage) if token_usage else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Get message analytics error: {str(e)}")
            raise
    
    async def get_document_analytics(self, db: Session, user_id: int, period: str) -> Dict[str, Any]:
        """Get document analytics"""
        try:
            start_date, end_date = self._get_date_range(period)
            
            documents = db.query(Document).filter(
                Document.uploaded_by_id == user_id,
                Document.created_at >= start_date
            ).all()
            
            if not documents:
                return {
                    "summary": {"total_documents": 0},
                    "status_distribution": {},
                    "type_distribution": {},
                    "size_analysis": {}
                }
            
            # Status distribution
            status_dist = defaultdict(int)
            for doc in documents:
                status_dist[doc.status.value] += 1
            
            # Type distribution
            type_dist = defaultdict(int)
            for doc in documents:
                type_dist[doc.file_type.value] += 1
            
            # Size analysis
            total_size = sum(doc.file_size for doc in documents)
            avg_size = total_size / len(documents)
            
            return {
                "summary": {
                    "total_documents": len(documents),
                    "total_size_mb": round(total_size / (1024 * 1024), 2),
                    "average_size_mb": round(avg_size / (1024 * 1024), 2)
                },
                "status_distribution": dict(status_dist),
                "type_distribution": dict(type_dist),
                "size_analysis": {
                    "total_bytes": total_size,
                    "average_bytes": round(avg_size, 2),
                    "largest_file_mb": round(max(doc.file_size for doc in documents) / (1024 * 1024), 2) if documents else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Get document analytics error: {str(e)}")
            raise
    
    async def get_usage_patterns(self, db: Session, user_id: int, period: str) -> Dict[str, Any]:
        """Get user usage patterns"""
        try:
            start_date, end_date = self._get_date_range(period)
            
            # Get all user messages for pattern analysis
            messages = db.query(Message).join(Conversation).filter(
                Conversation.user_id == user_id,
                Message.created_at >= start_date
            ).all()
            
            if not messages:
                return {"patterns": {}, "peak_hours": [], "peak_days": []}
            
            # Daily patterns
            daily_counts = defaultdict(int)
            hourly_counts = defaultdict(int)
            weekday_counts = defaultdict(int)
            
            for message in messages:
                date_key = message.created_at.date().isoformat()
                hour_key = message.created_at.hour
                weekday_key = message.created_at.strftime('%A')
                
                daily_counts[date_key] += 1
                hourly_counts[hour_key] += 1
                weekday_counts[weekday_key] += 1
            
            # Find peak hours and days
            peak_hours = sorted(hourly_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            peak_days = sorted(weekday_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            
            return {
                "patterns": {
                    "daily_activity": dict(daily_counts),
                    "hourly_distribution": {str(k): v for k, v in hourly_counts.items()},
                    "weekday_distribution": dict(weekday_counts)
                },
                "peak_hours": [{"hour": f"{hour}:00", "count": count} for hour, count in peak_hours],
                "peak_days": [{"day": day, "count": count} for day, count in peak_days]
            }
            
        except Exception as e:
            logger.error(f"Get usage patterns error: {str(e)}")
            raise
    
    async def get_topic_analysis(self, db: Session, user_id: int, period: str, limit: int = 20) -> Dict[str, Any]:
        """Get topic analysis from user messages"""
        try:
            start_date, end_date = self._get_date_range(period)
            
            # Get user messages
            messages = db.query(Message).join(Conversation).filter(
                Conversation.user_id == user_id,
                Message.created_at >= start_date
            ).all()
            
            if not messages:
                return {"topics": [], "keywords": [], "categories": {}}
            
            # Simple keyword extraction from messages
            all_text = " ".join([m.content.lower() for m in messages if m.content])
            
            # Basic keyword extraction (in a real implementation, use NLP libraries)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text)
            word_counts = Counter(words)
            
            # Filter out common words
            stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'was', 'one', 'our', 'out', 'day', 'get', 'use', 'how', 'new', 'now', 'way', 'may', 'say'}
            keywords = [(word, count) for word, count in word_counts.most_common(limit) if word not in stop_words]
            
            # Simple topic categorization based on agricultural keywords
            agriculture_keywords = {
                'cereals': ['sorgho', 'mil', 'maïs', 'riz', 'blé', 'orge'],
                'legumes': ['niébé', 'arachide', 'haricot', 'soja', 'lentille'],
                'techniques': ['irrigation', 'semis', 'récolte', 'fertilisant', 'pesticide'],
                'weather': ['pluie', 'saison', 'sécheresse', 'climat', 'température'],
                'disease': ['maladie', 'parasite', 'insecte', 'champignon', 'virus']
            }
            
            categories = {}
            for category, keywords_list in agriculture_keywords.items():
                count = sum(word_counts.get(keyword, 0) for keyword in keywords_list)
                if count > 0:
                    categories[category] = count
            
            return {
                "topics": [{"keyword": word, "frequency": count} for word, count in keywords[:limit]],
                "keywords": keywords[:limit],
                "categories": categories
            }
            
        except Exception as e:
            logger.error(f"Get topic analysis error: {str(e)}")
            raise
    
    async def get_performance_metrics(self, db: Session, user_id: int, period: str) -> Dict[str, Any]:
        """Get performance metrics"""
        try:
            start_date, end_date = self._get_date_range(period)
            
            messages = db.query(Message).join(Conversation).filter(
                Conversation.user_id == user_id,
                Message.created_at >= start_date
            ).all()
            
            if not messages:
                return {"response_times": {}, "success_rate": 0, "efficiency": {}}
            
            # Response time metrics
            response_times = [m.response_time for m in messages if m.response_time and m.response_time > 0]
            
            # Success rate (based on helpful feedback)
            helpful_count = sum(1 for m in messages if m.is_helpful == True)
            total_feedback = sum(1 for m in messages if m.is_helpful is not None)
            success_rate = (helpful_count / total_feedback * 100) if total_feedback > 0 else 0
            
            # Token efficiency
            tokens_used = [m.tokens_used for m in messages if m.tokens_used and m.tokens_used > 0]
            avg_tokens = sum(tokens_used) / len(tokens_used) if tokens_used else 0
            
            return {
                "response_times": {
                    "average_seconds": round(sum(response_times) / len(response_times), 2) if response_times else 0,
                    "fastest_seconds": min(response_times) if response_times else 0,
                    "slowest_seconds": max(response_times) if response_times else 0,
                    "total_responses": len(response_times)
                },
                "success_rate": round(success_rate, 2),
                "efficiency": {
                    "average_tokens_per_response": round(avg_tokens, 2),
                    "total_tokens_used": sum(tokens_used) if tokens_used else 0,
                    "responses_analyzed": len(tokens_used)
                }
            }
            
        except Exception as e:
            logger.error(f"Get performance metrics error: {str(e)}")
            raise
    
    async def get_feedback_analytics(self, db: Session, user_id: int, period: str) -> Dict[str, Any]:
        """Get feedback analytics"""
        try:
            start_date, end_date = self._get_date_range(period)
            
            messages = db.query(Message).join(Conversation).filter(
                Conversation.user_id == user_id,
                Message.created_at >= start_date
            ).all()
            
            # Feedback analysis
            total_messages = len(messages)
            helpful_messages = sum(1 for m in messages if m.is_helpful == True)
            not_helpful_messages = sum(1 for m in messages if m.is_helpful == False)
            no_feedback_messages = sum(1 for m in messages if m.is_helpful is None)
            
            # Feedback comments
            feedback_comments = [m.feedback_comment for m in messages if m.feedback_comment]
            
            return {
                "summary": {
                    "total_messages": total_messages,
                    "messages_with_feedback": helpful_messages + not_helpful_messages,
                    "feedback_rate": round((helpful_messages + not_helpful_messages) / total_messages * 100, 2) if total_messages > 0 else 0
                },
                "feedback_distribution": {
                    "helpful": helpful_messages,
                    "not_helpful": not_helpful_messages,
                    "no_feedback": no_feedback_messages
                },
                "satisfaction_score": round(helpful_messages / (helpful_messages + not_helpful_messages) * 100, 2) if (helpful_messages + not_helpful_messages) > 0 else 0,
                "comments_received": len(feedback_comments),
                "recent_feedback": feedback_comments[-10:] if feedback_comments else []
            }
            
        except Exception as e:
            logger.error(f"Get feedback analytics error: {str(e)}")
            raise
    
    async def get_usage_trends(self, db: Session, user_id: int, metric: str, period: str, granularity: str) -> Dict[str, Any]:
        """Get usage trends over time"""
        try:
            start_date, end_date = self._get_date_range(period)
            
            # Define granularity mapping
            granularity_mapping = {
                "hour": "%Y-%m-%d %H:00",
                "day": "%Y-%m-%d",
                "week": "%Y-W%W",
                "month": "%Y-%m"
            }
            
            date_format = granularity_mapping.get(granularity, "%Y-%m-%d")
            
            if metric == "messages":
                query = db.query(
                    func.strftime(date_format, Message.created_at).label('period'),
                    func.count(Message.id).label('count')
                ).join(Conversation).filter(
                    Conversation.user_id == user_id,
                    Message.created_at >= start_date
                ).group_by('period').order_by('period')
            
            elif metric == "conversations":
                query = db.query(
                    func.strftime(date_format, Conversation.created_at).label('period'),
                    func.count(Conversation.id).label('count')
                ).filter(
                    Conversation.user_id == user_id,
                    Conversation.created_at >= start_date
                ).group_by('period').order_by('period')
            
            elif metric == "documents":
                query = db.query(
                    func.strftime(date_format, Document.created_at).label('period'),
                    func.count(Document.id).label('count')
                ).filter(
                    Document.uploaded_by_id == user_id,
                    Document.created_at >= start_date
                ).group_by('period').order_by('period')
            
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            results = query.all()
            
            return {
                "metric": metric,
                "granularity": granularity,
                "period": period,
                "data": [{"period": r.period, "count": r.count} for r in results]
            }
            
        except Exception as e:
            logger.error(f"Get usage trends error: {str(e)}")
            raise
    
    async def export_analytics(self, db: Session, user_id: int, format: str, period: str, include: List[str]) -> Dict[str, Any]:
        """Export analytics data"""
        try:
            export_data = {}
            
            if "overview" in include:
                export_data["overview"] = await self.get_user_overview(db, user_id, period)
            
            if "conversations" in include:
                export_data["conversations"] = await self.get_conversation_analytics(db, user_id, period)
            
            if "messages" in include:
                export_data["messages"] = await self.get_message_analytics(db, user_id, period)
            
            if "documents" in include:
                export_data["documents"] = await self.get_document_analytics(db, user_id, period)
            
            if "performance" in include:
                export_data["performance"] = await self.get_performance_metrics(db, user_id, period)
            
            if format == "csv":
                # Convert to CSV format
                csv_data = self._convert_to_csv(export_data)
                return {
                    "format": "csv",
                    "data": csv_data,
                    "filename": f"analytics_export_{user_id}_{period}.csv"
                }
            
            return {
                "format": "json",
                "data": export_data,
                "filename": f"analytics_export_{user_id}_{period}.json"
            }
            
        except Exception as e:
            logger.error(f"Export analytics error: {str(e)}")
            raise
    
    def _convert_to_csv(self, data: Dict[str, Any]) -> str:
        """Convert analytics data to CSV format"""
        output = io.StringIO()
        
        # Write a simplified CSV version
        writer = csv.writer(output)
        writer.writerow(["Section", "Metric", "Value"])
        
        for section, section_data in data.items():
            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    if isinstance(value, (int, float, str)):
                        writer.writerow([section, key, value])
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            writer.writerow([f"{section}_{key}", sub_key, sub_value])
        
        return output.getvalue()
    
    # Admin-only methods
    async def get_system_overview(self, db: Session, period: str) -> Dict[str, Any]:
        """Get system-wide analytics overview (admin only)"""
        try:
            start_date, end_date = self._get_date_range(period)
            
            total_users = db.query(User).count()
            active_users = db.query(func.count(func.distinct(Conversation.user_id))).filter(
                Conversation.created_at >= start_date
            ).scalar() or 0
            
            total_conversations = db.query(Conversation).filter(
                Conversation.created_at >= start_date
            ).count()
            
            total_messages = db.query(Message).filter(
                Message.created_at >= start_date
            ).count()
            
            total_documents = db.query(Document).filter(
                Document.created_at >= start_date
            ).count()
            
            return {
                "period": period,
                "users": {
                    "total_users": total_users,
                    "active_users": active_users,
                    "activity_rate": round(active_users / total_users * 100, 2) if total_users > 0 else 0
                },
                "content": {
                    "conversations": total_conversations,
                    "messages": total_messages,
                    "documents": total_documents
                },
                "averages": {
                    "conversations_per_user": round(total_conversations / active_users, 2) if active_users > 0 else 0,
                    "messages_per_user": round(total_messages / active_users, 2) if active_users > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Get system overview error: {str(e)}")
            raise
    
    async def get_user_analytics(self, db: Session, period: str, limit: int = 50) -> Dict[str, Any]:
        """Get analytics for all users (admin only)"""
        try:
            start_date, end_date = self._get_date_range(period)
            
            # Get top users by activity
            user_activity = db.query(
                User.id,
                User.username,
                User.full_name,
                func.count(Message.id).label('message_count'),
                func.count(func.distinct(Conversation.id)).label('conversation_count')
            ).outerjoin(Conversation, User.id == Conversation.user_id)\
             .outerjoin(Message, Conversation.id == Message.conversation_id)\
             .filter(or_(Message.created_at >= start_date, Message.created_at.is_(None)))\
             .group_by(User.id)\
             .order_by(desc('message_count'))\
             .limit(limit).all()
            
            return {
                "period": period,
                "top_users": [
                    {
                        "user_id": u.id,
                        "username": u.username,
                        "full_name": u.full_name,
                        "message_count": u.message_count,
                        "conversation_count": u.conversation_count
                    }
                    for u in user_activity
                ]
            }
            
        except Exception as e:
            logger.error(f"Get user analytics error: {str(e)}")
            raise