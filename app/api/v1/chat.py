"""
Chat and RAG API routes - Email-based Authentication
"""
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import logging
import asyncio
import hashlib

from app.config.database import get_db
from app.core.instances import rag_engine
from app.models.user import User
from app.models.conversation import Conversation, Message
from app.schemas.chat import (
    ConversationCreate, ConversationResponse, ConversationWithMessages,
    MessageCreate, MessageResponse, QuickQuery, QuickQueryResponse,
    MessageFeedback, ChatStats
)
from app.services.chat_service import ChatService
from app.services.tts_service import tts_service  # Import du service TTS

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize chat service
chat_service = ChatService()


async def generate_audio_if_wolof(answer_wolof: Optional[str]) -> Optional[str]:
    """
    Générer automatiquement l'audio si une traduction wolof est disponible
    
    Args:
        answer_wolof: Texte en wolof à synthétiser
        
    Returns:
        Chemin relatif vers le fichier audio ou None
    """
    if not answer_wolof or not answer_wolof.strip():
        return None
    
    try:
        logger.info("🎤 Generating automatic Wolof audio...")
        
        # Générer l'audio en arrière-plan
        audio_result = await tts_service.generate_wolof_audio(
            wolof_text=answer_wolof,
            format="mp3",
            use_cache=True
        )
        
        if audio_result.get("success") and audio_result.get("filename"):
            audio_path = f"/audio/files/{audio_result['filename']}"
            logger.info(f"✅ Wolof audio generated: {audio_path}")
            return audio_path
        else:
            logger.warning(f"⚠️ Audio generation failed: {audio_result.get('error')}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error generating Wolof audio: {str(e)}")
        return None


def get_or_create_user_by_email(db: Session, email: str) -> User:
    """Get or create user by email"""
    # Chercher l'utilisateur par email
    user = db.query(User).filter(User.email == email).first()
    
    if not user:
        # Créer un nouvel utilisateur
        # Générer un nom d'utilisateur basé sur l'email
        username = email.split('@')[0]
        
        # S'assurer que le nom d'utilisateur est unique
        base_username = username
        counter = 1
        while db.query(User).filter(User.username == username).first():
            username = f"{base_username}_{counter}"
            counter += 1
        
        # Générer un hash par défaut pour les utilisateurs email (pas de mot de passe requis)
        # Utiliser l'email comme base pour un hash unique mais non-sensible
        default_hash = hashlib.sha256(f"email_user_{email}".encode()).hexdigest()
        
        user = User(
            email=email,
            username=username,
            hashed_password=default_hash,  # Hash par défaut pour satisfaire la contrainte NOT NULL
            full_name=email.split('@')[0].title(),
            is_active=True
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        
        logger.info(f"New user created: {email}")
    
    return user


@router.get("/conversations", response_model=List[ConversationResponse])
async def get_conversations(
    skip: int = 0,
    limit: int = 20,
    user_email: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get user conversations by email"""
    try:
        if not user_email:
            return []
        
        # Récupérer ou créer l'utilisateur
        user = get_or_create_user_by_email(db, user_email)
        
        # Récupérer les conversations de l'utilisateur
        conversations = db.query(Conversation).filter(
            Conversation.user_id == user.id,
            Conversation.is_active == True
        ).order_by(Conversation.updated_at.desc()).offset(skip).limit(limit).all()
        
        return conversations
        
    except Exception as e:
        logger.error(f"Get conversations error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversations"
        )


@router.post("/conversations", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    conversation_data: ConversationCreate,
    user_email: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Create a new conversation for user identified by email"""
    try:
        if not user_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User email is required"
            )
        
        # Récupérer ou créer l'utilisateur
        user = get_or_create_user_by_email(db, user_email)
        
        conversation = await chat_service.create_conversation(
            db, user.id, conversation_data
        )
        logger.info(f"New conversation created: {conversation.id} for user {user_email}")
        return conversation
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create conversation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create conversation"
        )


@router.get("/conversations/{conversation_id}", response_model=ConversationWithMessages)
async def get_conversation(
    conversation_id: int,
    user_email: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get conversation with messages (email-based access)"""
    try:
        if not user_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User email is required"
            )
        
        # Récupérer l'utilisateur
        user = get_or_create_user_by_email(db, user_email)
        
        conversation = await chat_service.get_conversation_with_messages(
            db, conversation_id, user.id
        )
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        return conversation
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get conversation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation"
        )


@router.post("/ask", response_model=MessageResponse)
async def ask_question(
    message_data: MessageCreate,
    background_tasks: BackgroundTasks,
    user_email: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Ask a question to the RAG system (email-based authentication)"""
    try:
        # Check if RAG engine is initialized
        status_info = rag_engine.get_status()
        if not status_info.get("initialized", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG system is not available"
            )
        
        if not user_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User email is required"
            )
        
        # Récupérer ou créer l'utilisateur
        user = get_or_create_user_by_email(db, user_email)
        
        # Get or create conversation
        conversation = None
        conversation_history = []
        
        if message_data.conversation_id:
            conversation = await chat_service.get_conversation(
                db, message_data.conversation_id, user.id
            )
            if not conversation:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found"
                )
            
            # Récupérer l'historique des messages de cette conversation
            existing_messages = db.query(Message).filter(
                Message.conversation_id == conversation.id
            ).order_by(Message.created_at.desc()).limit(10).all()  # Derniers 10 messages
            
            # Formater l'historique pour le RAG engine
            conversation_history = [
                {
                    "question": msg.question,
                    "answer": msg.answer
                }
                for msg in reversed(existing_messages)  # Ordre chronologique
            ]
        else:
            # Create new conversation with question as title
            conv_data = ConversationCreate(
                title=message_data.question[:50] + "..." if len(message_data.question) > 50 else message_data.question
            )
            conversation = await chat_service.create_conversation(
                db, user.id, conv_data
            )
        
        # Query RAG engine with conversation history
        rag_response = rag_engine.query(
            message_data.question, 
            conversation_history=conversation_history,
            return_sources=True
        )
        
        # Create message
        message = await chat_service.create_message(
            db, conversation.id, message_data.question, rag_response
        )
        
        # Update conversation stats in background
        background_tasks.add_task(
            chat_service.update_conversation_stats, 
            db, conversation.id
        )
        
        logger.info(f"Question answered for user {user_email} in conversation {conversation.id}")
        
        return message
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ask question error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process question"
        )


@router.post("/quick", response_model=QuickQueryResponse)
async def quick_query(
    query_data: QuickQuery,
    user_email: Optional[str] = None
):
    """Quick query with optional email for history tracking and automatic Wolof audio"""
    try:
        # Check if RAG engine is initialized
        status_info = rag_engine.get_status()
        if not status_info.get("initialized", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG system is not available"
            )
        
        # Query RAG engine with more sources
        rag_response = rag_engine.query(
            question=query_data.question,
            return_sources=query_data.include_sources
        )
        
        # Récupérer la traduction wolof depuis la réponse RAG
        answer_wolof = rag_response.get("answer_wolof")
        
        # Générer automatiquement l'audio wolof si disponible
        # audio_wolof_path = None
        # if answer_wolof:
        #     try:
        #         audio_wolof_path = await generate_audio_if_wolof(answer_wolof)
        #     except Exception as audio_error:
        #         logger.warning(f"⚠️ Audio generation non-blocking error: {audio_error}")
        #         # L'erreur audio ne doit pas faire échouer la requête
        
        # Format response
        response = QuickQueryResponse(
            question=query_data.question,
            answer=rag_response["answer"],
            answer_wolof=answer_wolof,
            # audio_wolof_path=audio_wolof_path,  # Inclure le chemin audio généré
            sources=rag_response["sources"] if query_data.include_sources else None,
            response_time=rag_response["response_time"],
            model_used=rag_response["model"],
            temperature_used=query_data.temperature,
            tokens_used=rag_response.get("tokens_used")
        )
        
        user_info = f"user {user_email}" if user_email else "anonymous user"
        logger.info(f"Quick query processed for {user_info} (audio: {'✅' })")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quick query error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query"
        )


@router.post("/messages/{message_id}/feedback")
async def add_message_feedback(
    message_id: int,
    feedback_data: MessageFeedback,
    db: Session = Depends(get_db)
):
    """Add feedback to a message (anonymous access)"""
    try:
        # Get message without user verification for anonymous access
        message = db.query(Message).filter(
            Message.id == message_id
        ).first()
        
        if not message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Message not found"
            )
        
        # Update feedback
        message.is_helpful = feedback_data.is_helpful
        message.feedback_comment = feedback_data.feedback_comment
        db.commit()
        
        logger.info(f"Feedback added to message {message_id} by anonymous user")
        
        return {"message": "Feedback added successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add feedback error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add feedback"
        )


@router.get("/stats", response_model=ChatStats)
async def get_chat_stats(
    db: Session = Depends(get_db)
):
    """Get anonymous user chat statistics"""
    try:
        # For anonymous access, return default/empty stats
        stats = ChatStats(
            total_conversations=0,
            total_messages=0,
            avg_messages_per_conversation=0.0,
            most_active_day=None,
            total_tokens_used=0,
            avg_response_time=None
        )
        return stats
    except Exception as e:
        logger.error(f"Get chat stats error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    user_email: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Delete a conversation (email-based access)"""
    try:
        if not user_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User email is required"
            )
        
        # Récupérer l'utilisateur
        user = get_or_create_user_by_email(db, user_email)
        
        # Get conversation with user verification
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id,
            Conversation.is_active == True
        ).first()
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        # Soft delete (mark as inactive instead of hard delete)
        conversation.is_active = False
        db.commit()
        
        logger.info(f"Conversation {conversation_id} deleted by user {user_email}")
        
        return {"message": "Conversation deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete conversation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete conversation"
        )
