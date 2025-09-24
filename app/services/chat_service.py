"""
Chat service for conversation and message management
"""
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import logging

from app.models.conversation import Conversation, Message
from app.models.user import User
from app.schemas.chat import (
    ConversationCreate, ConversationResponse, ConversationWithMessages,
    MessageResponse, ChatStats
)

logger = logging.getLogger(__name__)


class ChatService:
    """Chat and conversation management service"""
    
    def _message_to_response(self, message: Message) -> MessageResponse:
        """Convert database Message to MessageResponse with proper source deserialization"""
        sources_used = None
        if message.sources_used:
            try:
                sources_used = json.loads(message.sources_used)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Failed to parse sources for message {message.id}")
                sources_used = None
        
        # Convert response_time to float if it's a string
        response_time = message.response_time
        if response_time and isinstance(response_time, str):
            try:
                response_time = float(response_time)
            except (ValueError, TypeError):
                response_time = message.response_time  # Keep as string if conversion fails
        
        return MessageResponse(
            id=message.id,
            conversation_id=message.conversation_id,
            question=message.question,
            answer=message.answer,
            answer_wolof=message.answer_wolof,  # Include Wolof translation
            sources_used=sources_used,
            response_time=response_time,
            tokens_used=message.tokens_used,
            model_used=message.model_used,
            temperature_used=message.temperature_used,
            is_helpful=message.is_helpful,
            feedback_comment=message.feedback_comment,
            created_at=message.created_at
        )
    
    async def create_conversation(
        self, 
        db: Session, 
        user_id: int, 
        conversation_data: ConversationCreate
    ) -> Conversation:
        """Create a new conversation"""
        try:
            db_conversation = Conversation(
                user_id=user_id,
                title=conversation_data.title,
                description=conversation_data.description,
                llm_model=conversation_data.llm_model,
                temperature=conversation_data.temperature,
                max_tokens=conversation_data.max_tokens
            )
            
            db.add(db_conversation)
            db.commit()
            db.refresh(db_conversation)
            
            logger.info(f"Conversation created: {db_conversation.id} for user {user_id}")
            return db_conversation
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create conversation: {str(e)}")
            raise
    
    async def get_conversation(
        self, 
        db: Session, 
        conversation_id: int, 
        user_id: int
    ) -> Optional[Conversation]:
        """Get conversation by ID for specific user (or any conversation for anonymous user)"""
        if user_id == -1:  # Anonymous user can access any conversation
            return db.query(Conversation).filter(
                Conversation.id == conversation_id,
                Conversation.is_active == True
            ).first()
        else:
            return db.query(Conversation).filter(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
                Conversation.is_active == True
            ).first()
    
    async def get_conversation_with_messages(
        self, 
        db: Session, 
        conversation_id: int, 
        user_id: int
    ) -> Optional[ConversationWithMessages]:
        """Get conversation with all messages"""
        conversation = await self.get_conversation(db, conversation_id, user_id)
        if not conversation:
            return None
        
        # Get messages
        db_messages = db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.created_at).all()
        
        # Convert messages to proper response format with deserialized sources
        messages = [self._message_to_response(msg) for msg in db_messages]
        
        # Convert to response format
        conversation_dict = {
            "id": conversation.id,
            "title": conversation.title,
            "description": conversation.description,
            "user_id": conversation.user_id,
            "is_active": conversation.is_active,
            "total_messages": conversation.total_messages,
            "total_tokens_used": conversation.total_tokens_used,
            "llm_model": conversation.llm_model,
            "temperature": conversation.temperature,
            "max_tokens": conversation.max_tokens,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
            "last_message_at": conversation.last_message_at,
            "messages": messages
        }
        
        return ConversationWithMessages(**conversation_dict)
    
    async def get_user_conversations(
        self, 
        db: Session, 
        user_id: int, 
        skip: int = 0, 
        limit: int = 20
    ) -> List[Conversation]:
        """Get user conversations"""
        return db.query(Conversation).filter(
            Conversation.user_id == user_id,
            Conversation.is_active == True
        ).order_by(desc(Conversation.updated_at)).offset(skip).limit(limit).all()
    
    async def create_message(
        self, 
        db: Session, 
        conversation_id: int, 
        question: str, 
        rag_response: Dict[str, Any]
    ) -> MessageResponse:
        """Create a new message from RAG response"""
        try:
            # Parse sources for storage
            sources_json = None
            if rag_response.get("sources"):
                sources_json = json.dumps(rag_response["sources"])
            
            db_message = Message(
                conversation_id=conversation_id,
                question=question,
                answer=rag_response["answer"],
                answer_wolof=rag_response.get("answer_wolof"),  # Save Wolof translation
                sources_used=sources_json,
                response_time=str(rag_response.get("response_time", 0)),
                tokens_used=rag_response.get("tokens_used"),
                model_used=rag_response.get("model"),
                temperature_used=rag_response.get("temperature_used")
            )
            
            db.add(db_message)
            db.commit()
            db.refresh(db_message)
            
            logger.info(f"Message created: {db_message.id} in conversation {conversation_id}")
            
            # Return properly formatted MessageResponse with deserialized sources
            return MessageResponse(
                id=db_message.id,
                conversation_id=db_message.conversation_id,
                question=db_message.question,
                answer=db_message.answer,
                answer_wolof=db_message.answer_wolof,  # Include Wolof translation
                sources_used=rag_response.get("sources"),  # Use original sources list, not serialized JSON
                response_time=db_message.response_time,
                tokens_used=db_message.tokens_used,
                model_used=db_message.model_used,
                temperature_used=db_message.temperature_used,
                is_helpful=db_message.is_helpful,
                feedback_comment=db_message.feedback_comment,
                created_at=db_message.created_at
            )
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create message: {str(e)}")
            raise
    
    async def update_conversation_stats(self, db: Session, conversation_id: int):
        """Update conversation statistics"""
        try:
            conversation = db.query(Conversation).filter(
                Conversation.id == conversation_id
            ).first()
            
            if not conversation:
                return
            
            # Count messages
            message_count = db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).count()
            
            # Sum tokens used
            total_tokens = db.query(func.sum(Message.tokens_used)).filter(
                Message.conversation_id == conversation_id,
                Message.tokens_used.isnot(None)
            ).scalar() or 0
            
            # Get last message time
            last_message = db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(desc(Message.created_at)).first()
            
            # Update conversation
            conversation.total_messages = message_count
            conversation.total_tokens_used = total_tokens
            conversation.last_message_at = last_message.created_at if last_message else None
            conversation.updated_at = datetime.utcnow()
            
            db.commit()
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update conversation stats: {str(e)}")
    
    async def get_user_stats(self, db: Session, user_id: int) -> ChatStats:
        """Get user chat statistics"""
        try:
            # Total conversations
            total_conversations = db.query(Conversation).filter(
                Conversation.user_id == user_id,
                Conversation.is_active == True
            ).count()
            
            # Total messages
            total_messages = db.query(Message).join(Conversation).filter(
                Conversation.user_id == user_id,
                Conversation.is_active == True
            ).count()
            
            # Average messages per conversation
            avg_messages = total_messages / total_conversations if total_conversations > 0 else 0
            
            # Total tokens used
            total_tokens = db.query(func.sum(Message.tokens_used)).join(Conversation).filter(
                Conversation.user_id == user_id,
                Conversation.is_active == True,
                Message.tokens_used.isnot(None)
            ).scalar() or 0
            
            # Average response time
            avg_response_time = db.query(func.avg(Message.response_time)).join(Conversation).filter(
                Conversation.user_id == user_id,
                Conversation.is_active == True,
                Message.response_time.isnot(None)
            ).scalar()
            
            # Most active day (simplified - would need more complex query for real implementation)
            most_active_day = "Monday"  # Placeholder
            
            return ChatStats(
                total_conversations=total_conversations,
                total_messages=total_messages,
                avg_messages_per_conversation=round(avg_messages, 2),
                most_active_day=most_active_day,
                total_tokens_used=total_tokens,
                avg_response_time=float(avg_response_time) if avg_response_time else None
            )
            
        except Exception as e:
            logger.error(f"Failed to get user stats: {str(e)}")
            raise
    
    async def delete_conversation(self, db: Session, conversation_id: int, user_id: int) -> bool:
        """Delete conversation (soft delete)"""
        try:
            conversation = await self.get_conversation(db, conversation_id, user_id)
            if not conversation:
                return False
            
            conversation.is_active = False
            db.commit()
            
            logger.info(f"Conversation deleted: {conversation_id} by user {user_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete conversation: {str(e)}")
            raise
    
    async def search_conversations(
        self, 
        db: Session, 
        user_id: int, 
        query: str, 
        skip: int = 0, 
        limit: int = 20
    ) -> List[Conversation]:
        """Search user conversations by title or description"""
        return db.query(Conversation).filter(
            Conversation.user_id == user_id,
            Conversation.is_active == True,
            (Conversation.title.ilike(f"%{query}%") | 
             Conversation.description.ilike(f"%{query}%"))
        ).order_by(desc(Conversation.updated_at)).offset(skip).limit(limit).all()
    
    async def get_message(self, db: Session, message_id: int, user_id: int) -> Optional[MessageResponse]:
        """Get message by ID for specific user"""
        db_message = db.query(Message).join(Conversation).filter(
            Message.id == message_id,
            Conversation.user_id == user_id,
            Conversation.is_active == True
        ).first()
        
        if not db_message:
            return None
            
        return self._message_to_response(db_message)
    
    async def update_message_feedback(
        self, 
        db: Session, 
        message_id: int, 
        user_id: int, 
        is_helpful: bool, 
        feedback_comment: Optional[str] = None
    ) -> Optional[MessageResponse]:
        """Update message feedback"""
        try:
            # Get the raw database message first
            db_message = db.query(Message).join(Conversation).filter(
                Message.id == message_id,
                Conversation.user_id == user_id,
                Conversation.is_active == True
            ).first()
            
            if not db_message:
                return None
            
            db_message.is_helpful = is_helpful
            db_message.feedback_comment = feedback_comment
            
            db.commit()
            db.refresh(db_message)
            
            logger.info(f"Message feedback updated: {message_id}")
            return self._message_to_response(db_message)
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update message feedback: {str(e)}")
            raise
