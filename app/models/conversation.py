"""
Conversation model for chat functionality
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Boolean, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.config.database import Base


class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # User relationship
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    user = relationship("User", back_populates="conversations")
    
    # Conversation metadata
    is_active = Column(Boolean, default=True, nullable=False)
    total_messages = Column(Integer, default=0, nullable=False)
    total_tokens_used = Column(Integer, default=0, nullable=False)
    
    # Configuration for this conversation
    llm_model = Column(String(100), nullable=True)
    temperature = Column(String(10), nullable=True)  # Store as string for flexibility
    max_tokens = Column(Integer, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    last_message_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, title='{self.title}', user_id={self.user_id})>"


class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Conversation relationship
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, index=True)
    conversation = relationship("Conversation", back_populates="messages")
    
    # Message content
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    answer_wolof = Column(Text, nullable=True)  # Wolof translation
    
    # RAG metadata
    sources_used = Column(JSON, nullable=True)  # Store source documents metadata
    response_time = Column(String(20), nullable=True)  # Response time in seconds
    tokens_used = Column(Integer, nullable=True)
    
    # Model information
    model_used = Column(String(100), nullable=True)
    temperature_used = Column(String(10), nullable=True)
    
    # Message metadata
    is_helpful = Column(Boolean, nullable=True)  # User feedback
    feedback_comment = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<Message(id={self.id}, conversation_id={self.conversation_id})>"
    
    @property
    def question_preview(self) -> str:
        """Get preview of question (first 100 chars)"""
        return self.question[:100] + "..." if len(self.question) > 100 else self.question
    
    @property
    def answer_preview(self) -> str:
        """Get preview of answer (first 200 chars)"""
        return self.answer[:200] + "..." if len(self.answer) > 200 else self.answer
