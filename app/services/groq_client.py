"""
Client Groq pour l'intégration LLM
"""
import os
import logging
from typing import Dict, Any, List, Optional
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class GroqClient:
    """Client pour l'API Groq avec support LangChain"""
    
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.temperature = float(os.getenv("GROQ_TEMPERATURE", "0.1"))
        self.max_tokens = int(os.getenv("GROQ_MAX_TOKENS", "2000"))
        
        # Client Groq direct pour tests simples
        self.client = Groq(api_key=self.api_key)
        
        # Client LangChain pour intégration RAG
        self.langchain_client = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        logger.info(f"✅ Groq client initialized with model: {self.model}")
    
    def test_connection(self) -> bool:
        """Test la connexion à l'API Groq"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": "Test de connexion - réponds juste 'OK'"
                }],
                max_tokens=10,
                temperature=0
            )
            
            result = response.choices[0].message.content
            logger.info(f"✅ Groq connection test successful: {result}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Groq connection test failed: {str(e)}")
            return False
    
    def generate_response(self, prompt: str, context: str = None) -> str:
        """Génère une réponse simple avec l'API Groq"""
        try:
            messages = []
            
            if context:
                messages.append({
                    "role": "system",
                    "content": f"Contexte: {context}"
                })
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"❌ Groq API error: {str(e)}")
            raise
    
    def get_langchain_llm(self):
        """Retourne le client LangChain pour l'intégration RAG"""
        return self.langchain_client
    
    def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """Completion de chat générique"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"❌ Groq chat completion error: {str(e)}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Retourne la liste des modèles Groq disponibles"""
        try:
            # Note: Cette API peut nécessiter des permissions spécifiques
            # Pour l'instant, retourner les modèles connus
            return [
                "llama-3.3-70b-versatile",
                "llama-3.1-70b-versatile", 
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
                "gemma-7b-it",
                "gemma2-9b-it"
            ]
        except Exception as e:
            logger.warning(f"Could not fetch Groq models: {str(e)}")
            return [self.model]
    
    def get_status(self) -> Dict[str, Any]:
        """Retourne le statut du client Groq"""
        try:
            connection_ok = self.test_connection()
            
            return {
                "provider": "groq",
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "connection_ok": connection_ok,
                "api_key_configured": bool(self.api_key)
            }
        except Exception as e:
            return {
                "provider": "groq",
                "model": self.model,
                "connection_ok": False,
                "error": str(e)
            }
