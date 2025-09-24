"""
Chat and conversation schemas
"""
from pydantic import BaseModel, field_validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime


class ConversationBase(BaseModel):
    title: str
    description: Optional[str] = None
    llm_model: Optional[str] = None
    temperature: Optional[str] = None
    max_tokens: Optional[int] = None


class ConversationCreate(ConversationBase):
    @field_validator('title')
    @classmethod
    def validate_title(cls, v):
        if len(v.strip()) < 3:
            raise ValueError('Title must be at least 3 characters long')
        return v.strip()


class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    llm_model: Optional[str] = None
    temperature: Optional[str] = None
    max_tokens: Optional[int] = None


class ConversationResponse(ConversationBase):
    id: int
    user_id: int
    is_active: bool
    total_messages: int
    total_tokens_used: int
    created_at: datetime
    updated_at: datetime
    last_message_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class MessageBase(BaseModel):
    question: str
    
    @field_validator('question')
    @classmethod
    def validate_question(cls, v):
        if len(v.strip()) < 3:
            raise ValueError('Question must be at least 3 characters long')
        return v.strip()


class MessageCreate(MessageBase):
    conversation_id: Optional[int] = None  # If None, creates new conversation


class MessageResponse(BaseModel):
    id: int
    conversation_id: int
    question: str
    answer: str
    answer_wolof: Optional[str] = None  # Wolof translation
    sources_used: Optional[List[Dict[str, Any]]] = None
    response_time: Optional[Union[str, float]] = None
    tokens_used: Optional[int] = None
    model_used: Optional[str] = None
    temperature_used: Optional[str] = None
    is_helpful: Optional[bool] = None
    feedback_comment: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class MessageFeedback(BaseModel):
    is_helpful: bool
    feedback_comment: Optional[str] = None


class ConversationWithMessages(ConversationResponse):
    messages: List[MessageResponse] = []


class QuickQuery(BaseModel):
    """For quick queries without conversation context"""
    question: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    include_sources: bool = True
    
    @field_validator('question')
    @classmethod
    def validate_question(cls, v):
        if len(v.strip()) < 3:
            raise ValueError('Question must be at least 3 characters long')
        return v.strip()


class QuickQueryResponse(BaseModel):
    question: str
    answer: str
    answer_wolof: Optional[str] = None  # Traduction en wolof
    audio_wolof_path: Optional[str] = None  # Chemin vers l'audio wolof
    sources: Optional[List[Dict[str, Any]]] = None
    response_time: float
    model_used: str
    temperature_used: Optional[float] = None
    tokens_used: Optional[int] = None


class ChatStats(BaseModel):
    """Chat usage statistics"""
    total_conversations: int
    total_messages: int
    avg_messages_per_conversation: float
    most_active_day: Optional[str] = None
    total_tokens_used: int
    avg_response_time: Optional[float] = None
    
    
class ConversationExport(BaseModel):
    """For exporting conversation data"""
    conversation: ConversationResponse
    messages: List[MessageResponse]
    export_format: str = "json"  # json, csv, txt
    include_metadata: bool = True


# WebSocket schemas
class WSMessageType(str):
    QUESTION = "question"
    ANSWER = "answer" 
    ERROR = "error"
    STATUS = "status"
    TYPING = "typing"


class WSMessage(BaseModel):
    type: str
    data: Dict[str, Any]
    conversation_id: Optional[int] = None
    timestamp: datetime = datetime.utcnow()


class WSQuestion(BaseModel):
    question: str
    conversation_id: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class WSAnswer(BaseModel):
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None
    response_time: float
    tokens_used: Optional[int] = None
    message_id: int


class WSError(BaseModel):
    error: str
    error_code: str
    details: Optional[str] = None


class WSStatus(BaseModel):
    status: str  # "thinking", "processing", "ready", "error"
    message: Optional[str] = None
