"""
WebSocket chat implementation for real-time communication - No Authentication Required
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, List, Set, Optional
import json
import logging
import asyncio
from datetime import datetime

from app.config.database import get_db
from app.core.rag_engine import rag_engine
from app.models.user import User
from app.models.conversation import Conversation
from app.services.chat_service import ChatService
from app.schemas.chat import ConversationCreate

logger = logging.getLogger(__name__)
router = APIRouter()

# Anonymous user ID for non-authenticated requests
ANONYMOUS_USER_ID = -1

# Connection manager for WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[int, Set[WebSocket]] = {}  # user_id -> set of websockets
        self.conversation_rooms: Dict[int, Set[int]] = {}  # conversation_id -> set of user_ids
        
    async def connect(self, websocket: WebSocket, user_id: int):
        """Accept websocket connection and add to user connections"""
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        self.active_connections[user_id].add(websocket)
        logger.info(f"User {user_id} connected via WebSocket")
    
    def disconnect(self, websocket: WebSocket, user_id: int):
        """Remove websocket connection"""
        if user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        logger.info(f"User {user_id} disconnected from WebSocket")
    
    async def send_personal_message(self, message: dict, user_id: int):
        """Send message to specific user's connections"""
        if user_id in self.active_connections:
            disconnected_websockets = []
            for websocket in self.active_connections[user_id]:
                try:
                    await websocket.send_text(json.dumps(message))
                except:
                    disconnected_websockets.append(websocket)
            
            # Clean up disconnected websockets
            for ws in disconnected_websockets:
                self.active_connections[user_id].discard(ws)
    
    async def join_conversation(self, user_id: int, conversation_id: int):
        """Join user to conversation room"""
        if conversation_id not in self.conversation_rooms:
            self.conversation_rooms[conversation_id] = set()
        self.conversation_rooms[conversation_id].add(user_id)
    
    async def leave_conversation(self, user_id: int, conversation_id: int):
        """Remove user from conversation room"""
        if conversation_id in self.conversation_rooms:
            self.conversation_rooms[conversation_id].discard(user_id)
            if not self.conversation_rooms[conversation_id]:
                del self.conversation_rooms[conversation_id]
    
    async def broadcast_to_conversation(self, message: dict, conversation_id: int):
        """Broadcast message to all users in conversation"""
        if conversation_id in self.conversation_rooms:
            for user_id in self.conversation_rooms[conversation_id]:
                await self.send_personal_message(message, user_id)

# Global connection manager
manager = ConnectionManager()
chat_service = ChatService()


@router.websocket("/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat (no authentication required)"""
    # Get database session
    db = next(get_db())
    try:
        # Connect anonymous user
        user_id = ANONYMOUS_USER_ID
        await manager.connect(websocket, user_id)
        
        # Send welcome message
        welcome_message = {
            "type": "connection_established",
            "message": "Welcome! You are now connected to the chat system.",
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        await manager.send_personal_message(welcome_message, user_id)
        
        try:
            while True:
                # Receive message from client
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                # Handle different message types
                await handle_websocket_message(websocket, user_id, message_data, db)
                
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"WebSocket error for anonymous user: {str(e)}")
            error_message = {
                "type": "error",
                "message": "An error occurred processing your message",
                "timestamp": datetime.utcnow().isoformat()
            }
            await manager.send_personal_message(error_message, user_id)
    
    finally:
        manager.disconnect(websocket, user_id)
        db.close()


async def handle_websocket_message(websocket: WebSocket, user_id: int, message_data: dict, db: Session):
    """Handle different types of WebSocket messages"""
    message_type = message_data.get("type")
    
    if message_type == "ping":
        # Heartbeat/ping message
        response = {
            "type": "pong",
            "timestamp": datetime.utcnow().isoformat()
        }
        await manager.send_personal_message(response, user_id)
    
    elif message_type == "join_conversation":
        # Join a conversation room
        conversation_id = message_data.get("conversation_id")
        if conversation_id:
            # For anonymous access, allow joining any conversation
            conversation = db.query(Conversation).filter(
                Conversation.id == conversation_id
            ).first()
            
            if conversation:
                await manager.join_conversation(user_id, conversation_id)
                response = {
                    "type": "joined_conversation",
                    "conversation_id": conversation_id,
                    "message": f"Joined conversation: {conversation.title}",
                    "timestamp": datetime.utcnow().isoformat()
                }
                await manager.send_personal_message(response, user_id)
            else:
                error_response = {
                    "type": "error",
                    "message": "Conversation not found",
                    "timestamp": datetime.utcnow().isoformat()
                }
                await manager.send_personal_message(error_response, user_id)
    
    elif message_type == "leave_conversation":
        # Leave a conversation room
        conversation_id = message_data.get("conversation_id")
        if conversation_id:
            await manager.leave_conversation(user_id, conversation_id)
            response = {
                "type": "left_conversation",
                "conversation_id": conversation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            await manager.send_personal_message(response, user_id)
    
    elif message_type == "chat_message":
        # Process chat message with RAG
        await handle_chat_message(user_id, message_data, db)
    
    elif message_type == "typing":
        # Handle typing indicator
        conversation_id = message_data.get("conversation_id")
        if conversation_id:
            typing_message = {
                "type": "user_typing",
                "user_id": user_id,
                "username": "Anonymous",
                "conversation_id": conversation_id,
                "typing": message_data.get("typing", True),
                "timestamp": datetime.utcnow().isoformat()
            }
            await manager.broadcast_to_conversation(typing_message, conversation_id)
    
    else:
        # Unknown message type
        error_response = {
            "type": "error",
            "message": f"Unknown message type: {message_type}",
            "timestamp": datetime.utcnow().isoformat()
        }
        await manager.send_personal_message(error_response, user_id)


async def handle_chat_message(user_id: int, message_data: dict, db: Session):
    """Handle chat message processing with RAG"""
    try:
        question = message_data.get("message")
        conversation_id = message_data.get("conversation_id")
        
        if not question:
            error_response = {
                "type": "error",
                "message": "Message content is required",
                "timestamp": datetime.utcnow().isoformat()
            }
            await manager.send_personal_message(error_response, user_id)
            return
        
        # Send typing indicator
        typing_message = {
            "type": "assistant_typing",
            "conversation_id": conversation_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        await manager.send_personal_message(typing_message, user_id)
        
        # Check if RAG engine is available
        if not rag_engine.initialized:
            error_response = {
                "type": "error",
                "message": "RAG system is currently unavailable. Please try again later.",
                "timestamp": datetime.utcnow().isoformat()
            }
            await manager.send_personal_message(error_response, user_id)
            return
        
        # Get or create conversation
        conversation = None
        if conversation_id:
            conversation = db.query(Conversation).filter(
                Conversation.id == conversation_id
            ).first()
        
        if not conversation:
            # Create new conversation
            conv_data = ConversationCreate(
                title=question[:50] + "..." if len(question) > 50 else question,
                description="WebSocket conversation (Anonymous)"
            )
            conversation = await chat_service.create_conversation(db, user_id, conv_data)
            
            # Send conversation created message
            conv_message = {
                "type": "conversation_created",
                "conversation": {
                    "id": conversation.id,
                    "title": conversation.title,
                    "description": conversation.description,
                    "created_at": conversation.created_at.isoformat()
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            await manager.send_personal_message(conv_message, user_id)
        
        # Join conversation room
        await manager.join_conversation(user_id, conversation.id)
        
        # Query RAG engine
        rag_response = await rag_engine.query(question)
        
        # Create message in database
        message = await chat_service.create_message(
            db, conversation.id, question, rag_response
        )
        
        # Update conversation stats
        await chat_service.update_conversation_stats(db, conversation.id)
        
        # Send response back to user
        response_message = {
            "type": "chat_response",
            "conversation_id": conversation.id,
            "message": {
                "id": message.id,
                "question": question,
                "answer": rag_response["answer"],
                "sources": rag_response.get("sources", []),
                "response_time": rag_response.get("response_time", 0),
                "model_used": rag_response.get("model"),
                "tokens_used": rag_response.get("tokens_used"),
                "created_at": message.created_at.isoformat()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Send to user
        await manager.send_personal_message(response_message, user_id)
        
        # Stop typing indicator
        stop_typing_message = {
            "type": "assistant_typing",
            "conversation_id": conversation.id,
            "typing": False,
            "timestamp": datetime.utcnow().isoformat()
        }
        await manager.send_personal_message(stop_typing_message, user_id)
        
        logger.info(f"WebSocket message processed for anonymous user in conversation {conversation.id}")
        
    except Exception as e:
        logger.error(f"Handle chat message error: {str(e)}")
        error_response = {
            "type": "error",
            "message": "Failed to process your message. Please try again.",
            "timestamp": datetime.utcnow().isoformat()
        }
        await manager.send_personal_message(error_response, user_id)


@router.websocket("/status")
async def websocket_status(websocket: WebSocket):
    """WebSocket endpoint for system status updates"""
    await websocket.accept()
    
    try:
        while True:
            # Send status update every 30 seconds
            status_data = {
                "type": "system_status",
                "rag_engine": {
                    "initialized": rag_engine.initialized,
                    "status": rag_engine.get_status()
                },
                "active_connections": len(manager.active_connections),
                "active_conversations": len(manager.conversation_rooms),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await websocket.send_text(json.dumps(status_data))
            await asyncio.sleep(30)
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Status WebSocket error: {str(e)}")
    finally:
        await websocket.close()