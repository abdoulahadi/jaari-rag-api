"""
RAG Engine Professionnel pour Corpus Massif
------------------------------------------
Optimisé pour la gestion de gros corpus avec vectorstore avancé, batching, reranking et configuration flexible.

Fonctionnalités :
- Initialisation configurable du LLM et du vectorstore
- Support du choix d'index FAISS (Flat, IVF, HNSW)
- Indexation par batch pour la scalabilité
- Option de reranking pour améliorer la pertinence
- Monitoring et logs détaillés
- Documentation professionnelle
- Support async/sync pour API et traitement batch
"""
from typing import List, Dict, Any, Optional, Tuple
import time
import logging
import asyncio
from pathlib import Path

# LangChain imports
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

# Import Groq support
try:
    from app.services.groq_client import GroqClient
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    GroqClient = None

from app.core.document_loader import DocumentLoader, DocumentSplitter
from app.core.vectorstore import VectorStoreManager
from app.core.intelligent_search import IntelligentQueryProcessor
from app.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RAGEngine:
    """
    RAG Engine Unifié - Version Complète
    ------------------------------------
    Moteur RAG optimisé pour corpus volumineux avec support async/sync :
    - Initialisation flexible du LLM et du vectorstore
    - Support du batching et du choix d'index FAISS
    - Option de reranking (cross-encoder)
    - Monitoring et logs professionnels
    - API asynchrone et synchrone
    """
    
    def __init__(self):
        # Core components
        self.document_loader = DocumentLoader()
        self.document_splitter = DocumentSplitter()
        self.vectorstore_manager = VectorStoreManager()
        self.query_processor = IntelligentQueryProcessor()
        
        # LLM and chain components
        self.llm = None
        self.groq_client = None
        self.qa_chain = None
        self.initialized = False
        
        # Configuration du provider LLM
        self.llm_provider = getattr(settings, 'LLM_PROVIDER', 'groq').lower()
        logger.info(f"🔧 LLM Provider configured: {self.llm_provider}")
        
        # Configuration LLM avec settings par défaut
        self.llm_config = {
            "model": getattr(settings, 'OLLAMA_MODEL', "llama3.1:latest"),
            "temperature": getattr(settings, 'LLM_TEMPERATURE', 0.1),
            "base_url": getattr(settings, 'OLLAMA_BASE_URL', "http://localhost:11434"),
            "timeout": getattr(settings, 'OLLAMA_TIMEOUT', 180),
            "keep_alive": getattr(settings, 'OLLAMA_KEEP_ALIVE', "5m"),
            "max_tokens": getattr(settings, 'LLM_MAX_TOKENS', 2000)
        }
        
        # Configuration Groq
        self.groq_config = {
            "model": getattr(settings, 'GROQ_MODEL', "llama-3.3-70b-versatile"),
            "temperature": float(getattr(settings, 'GROQ_TEMPERATURE', 0.1)),
            "max_tokens": int(getattr(settings, 'GROQ_MAX_TOKENS', 2000))
        }
        
        # Configuration vectorstore (amélioration pour performance)
        self.vectorstore_config = {
            "persist_directory": getattr(settings, 'VECTORSTORE_PATH', "data/vectorstore"),
            "chunk_size": getattr(settings, 'CHUNK_SIZE', 1000),
            "chunk_overlap": getattr(settings, 'CHUNK_OVERLAP', 200),
            "search_k": getattr(settings, 'SEARCH_K', 12),  # Augmenté pour plus de contexte
            "search_k_enhanced": 20,  # Pour recherche approfondie
            "max_context_length": 8000  # Limite de contexte pour le LLM
        }
        
        # Configuration avancée
        self.index_type = getattr(settings, 'FAISS_INDEX_TYPE', "IVF")
        self.batch_size = getattr(settings, 'BATCH_SIZE', 1000)
    
    # =================
    # ASYNC API METHODS
    # =================
    
    async def initialize(self) -> bool:
        """Initialize all RAG components asynchronously"""
        try:
            logger.info("Initializing RAG Engine...")
            
            # Initialize LLM
            await self._initialize_llm_async()
            
            # Load existing vectorstore if available
            await self._load_vectorstore_async()
            
            self.initialized = True
            logger.info("RAG Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG Engine: {str(e)}")
            return False
    
    async def _initialize_llm_async(self):
        """Initialize LLM based on configured provider asynchronously"""
        try:
            if self.llm_provider == "groq":
                await self._initialize_groq_async()
            else:
                await self._initialize_ollama_async()
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider '{self.llm_provider}': {str(e)}")
            # Try fallback to other provider
            try:
                if self.llm_provider == "groq":
                    logger.info("🔄 Groq failed, falling back to Ollama...")
                    await self._initialize_ollama_async()
                else:
                    logger.info("🔄 Ollama failed, falling back to Groq...")
                    await self._initialize_groq_async()
            except Exception as fallback_error:
                logger.error(f"❌ All LLM providers failed: {str(fallback_error)}")
                self.llm = None
    
    async def _initialize_groq_async(self):
        """Initialize Groq LLM asynchronously"""
        if not GROQ_AVAILABLE:
            raise ImportError("Groq not available - missing dependencies")
        
        try:
            self.groq_client = GroqClient()
            
            # Test connection
            connection_ok = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.groq_client.test_connection
            )
            
            if connection_ok:
                self.llm = self.groq_client.get_langchain_llm()
                logger.info(f"✅ Groq LLM initialized successfully: {self.groq_config['model']}")
            else:
                raise ConnectionError("Groq connection test failed")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize Groq: {str(e)}")
            raise
    
    async def _initialize_ollama_async(self):
        """Initialize Ollama LLM asynchronously"""
        try:
            self.llm = OllamaLLM(
                model=self.llm_config["model"],
                base_url=self.llm_config["base_url"],
                timeout=self.llm_config["timeout"],
                keep_alive=self.llm_config["keep_alive"]
            )
            
            # Test connection
            test_response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.llm.invoke("Test connection - respond OK")
            )
            
            logger.info(f"✅ Ollama LLM initialized successfully: {self.llm_config['model']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Ollama: {str(e)}")
            raise
    
    async def _load_vectorstore_async(self):
        """Load existing vectorstore asynchronously"""
        try:
            if self.vectorstore_manager.load_vectorstore():
                logger.info("Vectorstore loaded successfully")
                
                # Setup QA chain if LLM is available
                if self.llm:
                    await self._setup_qa_chain_async()
                    
        except Exception as e:
            logger.warning(f"Could not load vectorstore: {str(e)}")
    
    async def _setup_qa_chain_async(self):
        """Setup the RAG QA chain asynchronously"""
        if not (self.llm and self.vectorstore_manager.vectorstore):
            raise ValueError("Both LLM and vectorstore must be initialized")
        
        try:
            retriever = self.vectorstore_manager.get_retriever(
                search_kwargs={"k": self.vectorstore_config["search_k"]}
            )
            
            # Create QA chain with agricultural prompt
            prompt_template = """
Tu es un assistant agricole expert du nom de Jaari, spécialisé dans la vulgarisation claire et rapide.
Ta mission : fournir des réponses précises, pédagogiques et directement actionnables pour les agriculteurs.

Contexte fourni :
{context}

Question de l'utilisateur :
{question}

Consignes :
- se présenter comme un expert agricole sénégalais nommé Jaari pour instaurer la confiance avec les utilisateurs locaux
- utiliser un langage simple et accessible, adapté au public cible
- Réponds uniquement sur la base du contexte fourni, sans extrapoler.
- Si l'information n'est pas présente, indique clairement : "Je n'ai pas suffisamment d'informations pour répondre à cette question."
- Utilise un langage simple, des phrases courtes et des exemples concrets si possible.
- Mets en avant les points clés, les dates, les recommandations pratiques et les actions à entreprendre.
- Structure ta réponse en étapes ou en liste si pertinent.
- Termine par un résumé ou une recommandation principale.

Réponse claire et concise :
"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            logger.info("QA Chain setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup QA chain: {str(e)}")
            raise
    
    async def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vectorstore asynchronously"""
        try:
            # Split documents into chunks
            chunks = self.document_splitter.split_documents(documents)
            
            # Add to vectorstore using manager
            if self.vectorstore_manager.vectorstore is None:
                self.vectorstore_manager.create_vectorstore(chunks)
            else:
                self.vectorstore_manager.add_documents(chunks)
            
            # Setup QA chain if LLM is available
            if self.llm:
                await self._setup_qa_chain_async()
                
            logger.info(f"Added {len(documents)} documents ({len(chunks)} chunks)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            return False
    
    async def query_async(
        self, 
        question: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """Query the RAG system asynchronously"""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Add documents first.")
        
        try:
            start_time = time.time()
            
            # Run query in executor to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.qa_chain({"query": question})
            )
            
            response_time = time.time() - start_time
            
            # Format response
            response = {
                "answer": result["result"],
                "sources": [
                    {
                        "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in result.get("source_documents", [])
                ],
                "response_time": response_time,
                "model": self.llm_config["model"],
                "question": question
            }
            
            logger.info(f"Query processed in {response_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise
    
    async def search_documents_async(
        self, 
        query: str, 
        k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search similar documents asynchronously"""
        if not self.vectorstore_manager.vectorstore:
            raise ValueError("Vectorstore not initialized")
        
        try:
            # Run search in executor
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vectorstore_manager.vectorstore.similarity_search_with_score(query, k=k)
            )
            
            # Filter by score threshold and format
            filtered_results = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(score)
                }
                for doc, score in results
                if score >= score_threshold
            ]
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Document search failed: {str(e)}")
            raise
    
    # =================
    # SYNC METHODS (Compatible avec l'ancienne API)
    # =================
    
    def initialize_llm(self, model: Optional[str] = None, **kwargs) -> Any:
        """Initialize LLM with enhanced error handling and configurable timeout"""
        provider = kwargs.get("provider", self.llm_provider)
        
        if provider == "groq":
            return self._initialize_groq_sync(model, **kwargs)
        else:
            return self._initialize_ollama_sync(model, **kwargs)
    
    def _initialize_groq_sync(self, model: Optional[str] = None, **kwargs):
        """Initialize Groq LLM synchronously"""
        if not GROQ_AVAILABLE:
            raise ImportError("Groq not available - please install: pip install groq langchain-groq")
        
        try:
            logger.info(f"🚀 Attempting to initialize Groq LLM...")
            self.groq_client = GroqClient()
            
            # Test connection
            if self.groq_client.test_connection():
                self.llm = self.groq_client.get_langchain_llm()
                logger.info(f"✅ Groq LLM initialized successfully: {self.groq_config['model']}")
                return self.llm
            else:
                raise ConnectionError("Groq connection test failed")
                
        except Exception as e:
            error_msg = f"❌ Erreur lors de l'initialisation de Groq: {str(e)}"
            
            if "api" in str(e).lower() or "key" in str(e).lower():
                error_msg += "\n💡 Suggestions:"
                error_msg += "\n- Vérifiez votre clé API Groq dans GROQ_API_KEY"
                error_msg += "\n- Vérifiez votre connexion internet"
            
            logger.error(error_msg)
            raise ConnectionError(error_msg)
    
    def _initialize_ollama_sync(self, model: Optional[str] = None, **kwargs):
        """Initialize Ollama LLM synchronously"""
        model = model or self.llm_config["model"]
        
        llm_params = {
            "model": model,
            "temperature": kwargs.get("temperature", self.llm_config["temperature"]),
            "base_url": kwargs.get("base_url", self.llm_config["base_url"]),
            "timeout": kwargs.get("timeout", self.llm_config["timeout"]),
            "keep_alive": kwargs.get("keep_alive", self.llm_config["keep_alive"]),
        }
        
        # Add max_tokens if specified
        if "max_tokens" in kwargs or "max_tokens" in self.llm_config:
            llm_params["num_predict"] = kwargs.get("max_tokens", self.llm_config["max_tokens"])
        
        try:
            logger.info(f"🚀 Attempting to initialize Ollama LLM: {model} at {llm_params['base_url']}")
            self.llm = OllamaLLM(**llm_params)
            
            # Test de connexion simple
            try:
                test_response = self.llm.invoke("Test de connexion - réponds juste 'OK'")
                logger.info(f"✅ Ollama connection test successful: {model}")
            except Exception as test_error:
                logger.warning(f"⚠️ Ollama connection test failed but initialization succeeded: {test_error}")
            
            logger.info(f"✅ Successfully initialized Ollama LLM: {model}")
            return self.llm
            
        except Exception as e:
            error_msg = f"❌ Erreur lors de l'initialisation d'Ollama {model}: {str(e)}"
            
            # Messages d'erreur plus spécifiques
            if "Connection" in str(e) or "connection" in str(e):
                error_msg += "\n💡 Suggestions:"
                error_msg += f"\n- Vérifiez qu'Ollama est démarré (ollama serve)"
                error_msg += f"\n- Vérifiez l'adresse: {llm_params['base_url']}"
                error_msg += f"\n- Testez avec: ollama list pour voir les modèles disponibles"
            elif "model" in str(e).lower():
                error_msg += f"\n💡 Le modèle '{model}' n'est peut-être pas installé."
                error_msg += f"\n- Installez-le avec: ollama pull {model}"
            
            logger.error(error_msg)
            raise ConnectionError(error_msg)
    
    def setup_rag_chain(self, custom_prompt: Optional[str] = None, use_rerank: bool = False) -> RetrievalQA:
        """
        Initialise la chaîne RAG avec retriever optimisé et option de reranking.
        Args:
            custom_prompt (str, optional): Prompt personnalisé.
            use_rerank (bool): Activer le reranking si disponible.
        Returns:
            RetrievalQA: Chaîne QA prête à l'emploi.
        """
        if not self.llm:
            self.initialize_llm()
        if not self.vectorstore_manager.vectorstore:
            logger.error("No vectorstore available. Please load documents first.")
            raise ValueError("No vectorstore available")
            
        try:
            retriever = self.vectorstore_manager.get_retriever(
                search_type="similarity",  # On peut aussi essayer "mmr" pour plus de diversité
                search_kwargs={"k": self.vectorstore_config["search_k"]}
            )
            
            # Option de reranking (cross-encoder)
            if use_rerank and hasattr(self.vectorstore_manager, 'reranker') and self.vectorstore_manager.reranker:
                retriever = self.vectorstore_manager.reranker.wrap_retriever(retriever)
                
            prompt_template = custom_prompt or (
                """
Tu es Jaari, assistant agricole expert du Sénégal. Tu maîtrises parfaitement l'agriculture sénégalaise et tu donnes des conseils pratiques et détaillés.

CONTEXTE DOCUMENTAIRE :
{context}

QUESTION :
{question}

INSTRUCTIONS PRÉCISES :
1. **ANALYSE APPROFONDIE** : Utilise TOUTES les informations disponibles dans le contexte pour donner une réponse complète et détaillée.

2. **STRUCTURE CLAIRE** : 
   - Commence par une introduction contextuelle
   - Développe en sections organisées (variétés, techniques, calendrier, rendements, etc.)
   - Termine par des recommandations concrètes

3. **INFORMATIONS SPÉCIFIQUES AU SÉNÉGAL** :
   - Mentionne les zones de culture (Niayes, Vallée du fleuve, Casamance, etc.)
   - Précise les saisons et calendriers agricoles
   - Indique les variétés adaptées au climat local
   - Donne des chiffres de rendement et de production quand disponibles

4. **CONSEIL PRATIQUE** : 
   - Donne des étapes concrètes à suivre
   - Mentionne les défis spécifiques et leurs solutions
   - Propose des bonnes pratiques locales

5. **SOURCES ET CRÉDIBILITÉ** :
   - Base-toi uniquement sur les informations du contexte
   - Si une information manque, indique clairement ce qui n'est pas disponible
   - Suggère des sources d'information complémentaires (centres agricoles, formations, etc.)

Réponse détaillée et structurée :
"""
            )
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            logger.info("RAG chain setup completed (index_type=%s, batch_size=%d)", 
                       self.index_type, self.batch_size)
            return self.qa_chain
            
        except Exception as e:
            logger.error(f"Error setting up RAG chain: {str(e)}")
            raise
    
    def _format_conversation_history(self, conversation_history: List[Dict[str, str]]) -> str:
        """Formater l'historique de conversation pour le prompt"""
        if not conversation_history:
            return ""
        
        history_text = "\n--- HISTORIQUE DE LA CONVERSATION ---\n"
        for i, exchange in enumerate(conversation_history[-5:], 1):  # Garder seulement les 5 derniers échanges
            history_text += f"\n{i}. UTILISATEUR: {exchange.get('question', '')}\n"
            history_text += f"   JAARI: {exchange.get('answer', '')}\n"
        
        history_text += "\n--- FIN DE L'HISTORIQUE ---\n"
        return history_text

    def _is_simple_greeting(self, question: str) -> bool:
        """Détecter si la question est une simple salutation pour traitement LLM optimisé"""
        question_lower = question.lower().strip()
        
        # Mots clés de salutations simples
        greeting_keywords = [
            'bonjour', 'bonsoir', 'salut', 'hello', 'hi', 'hey',
            'salam', 'asalaam', 'alaikum', 'nanga def', 'na nga def',
            'comment vas-tu', 'comment allez-vous', 'ça va', 'ca va',
            'comment tu vas', 'how are you', 'good morning', 'good evening'
        ]
        
        # Détecter si c'est une salutation simple
        words = question_lower.split()
        
        # Si la question contient des mots de salutation et fait moins de 10 mots
        if len(words) <= 10:
            for keyword in greeting_keywords:
                if keyword in question_lower:
                    return True
        
        return False
    
    def _generate_greeting_with_llm(self, question: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Générer une salutation via le LLM avec instructions pour Jaari et traduction wolof"""
        try:
            # Formater l'historique si disponible
            history_context = ""
            if conversation_history:
                history_context = self._format_conversation_history(conversation_history)
                history_context += "\nNOTE: Puisque nous avons déjà parlé, sois plus direct dans ta salutation.\n"
            
            # Prompt spécialisé pour les salutations avec instructions pour la traduction wolof
            greeting_prompt = f"""Tu es Jaari, assistant agricole expert du Sénégal. L'utilisateur te salue avec : "{question}"

{history_context}

INSTRUCTIONS STRICTES :
1. Réponds comme Jaari, assistant agricole sénégalais spécialisé
2. {"Salue brièvement puisque nous nous connaissons déjà" if conversation_history else "Salue chaleureusement et présente-toi brièvement"}
3. {"Propose directement ton aide" if conversation_history else "Mentionne tes domaines d'expertise agricole"}
4. Propose ton aide pour l'agriculture au Sénégal
5. Sois concis mais accueillant (maximum 100 mots)
6. Utilise un ton professionnel mais chaleureux

Réponse de salutation :"""

            # Générer la réponse via le LLM
            if not self.llm:
                self.initialize_llm()
            
            start_time = time.time()
            llm_response = self.llm.invoke(greeting_prompt)
            
            # Extraire le contenu textuel (compatible Groq et Ollama)
            if hasattr(llm_response, 'content'):
                french_response = llm_response.content
            elif isinstance(llm_response, str):
                french_response = llm_response
            else:
                french_response = str(llm_response)
            
            response_time = time.time() - start_time
            
            # Traduction automatique en wolof via Google Cloud
            wolof_answer = None
            try:
                from app.utils.google_credentials import get_google_credentials_path
                credentials_path = get_google_credentials_path()
                
                if credentials_path:
                    from google.cloud import translate_v2 as translate
                    translate_client = translate.Client()
                    
                    translation = translate_client.translate(
                        french_response, 
                        target_language="wo",
                        source_language="fr"
                    )
                    wolof_answer = translation["translatedText"]
                    logger.info(f"✅ Wolof translation completed for greeting")
                else:
                    logger.warning("🔐 Google Cloud credentials not configured - Wolof translation disabled")
            except Exception as translation_error:
                logger.warning(f"🔐 Wolof translation failed: {translation_error}")
                wolof_answer = None
            
            return {
                "question": question,
                "answer": french_response.strip(),
                "answer_wolof": wolof_answer,
                "audio_wolof_path": None,  # Audio will be generated separately when requested
                "response_time": response_time,
                "timestamp": time.time(),
                "model": self.llm_config["model"],
                "sources": [],
                "num_sources": 0,
                "is_greeting": True
            }
            
        except Exception as e:
            logger.error(f"Failed to generate LLM greeting: {str(e)}")
            # Fallback en cas d'erreur
            return {
                "question": question,
                "answer": "Bonjour ! Je suis Jaari, votre assistant agricole sénégalais. Comment puis-je vous aider aujourd'hui ?",
                "answer_wolof": "Asalaam alaikum ! Ana Jaari, assistant-bi agriculture Sénégal. Lan la bëgg ma wallal la ?",
                "audio_wolof_path": None,  # Audio will be generated separately when requested
                "response_time": 0.1,
                "timestamp": time.time(),
                "model": "fallback",
                "sources": [],
                "num_sources": 0,
                "is_greeting": True,
                "error": str(e)
            }

    def query(self, question: str, conversation_history: Optional[List[Dict[str, str]]] = None, return_sources: bool = True, optimize_timeout: bool = True) -> Dict[str, Any]:
        """Execute RAG query and return results with metadata, including Wolof translation and conversation history"""
        
        # 🎯 DÉTECTION DE SALUTATIONS SIMPLES EN PREMIER
        if self._is_simple_greeting(question):
            logger.info(f"Simple greeting detected: '{question}' - using LLM generation")
            return self._generate_greeting_with_llm(question, conversation_history)
        
        # Ensure all components are initialized
        try:
            if not self.llm:
                self.initialize_llm()
            
            if not self.vectorstore_manager.vectorstore:
                if not self.vectorstore_manager.load_vectorstore():
                    # Try to load from corpus directory
                    corpus_dir = getattr(settings, 'CORPUS_DIR', './data/corpus')
                    if not self.load_documents_from_directory(corpus_dir):
                        raise ValueError("No vectorstore available and no documents to load")
            
            if not self.qa_chain:
                self.setup_rag_chain()
                
        except Exception as init_error:
            logger.error(f"Failed to initialize RAG components: {str(init_error)}")
            return {
                "question": question,
                "answer": f"Erreur d'initialisation du système RAG: {str(init_error)}",
                "error": True,
                "response_time": 0,
                "model": self.llm_config.get("model", "unknown")
            }

        start_time = time.time()

        try:
            # 🔍 RECHERCHE AMÉLIORÉE DE DOCUMENTS
            logger.info(f"🔍 Processing query: '{question}'")
            
            # Déterminer le type de question pour optimiser la recherche
            question_type = self._classify_question(question)
            logger.info(f"📝 Question type: {question_type}")
            
            # Adapter la stratégie de recherche selon le type
            if question_type == "greeting":
                # Pour les salutations, recherche minimale
                search_k = 2
                use_enhanced_search = False
            elif question_type == "specific_crop":
                # Pour les questions sur des cultures spécifiques, recherche approfondie
                search_k = self.vectorstore_config["search_k_enhanced"]
                use_enhanced_search = True
            elif question_type == "technical":
                # Pour les questions techniques, recherche étendue
                search_k = self.vectorstore_config["search_k"]
                use_enhanced_search = True
            else:
                # Questions générales
                search_k = self.vectorstore_config["search_k"]
                use_enhanced_search = True
            
            # Effectuer la recherche de documents
            if use_enhanced_search:
                enhanced_docs = self._enhanced_document_search(question, search_k)
                # Convertir en format compatible avec le retriever
                retrieved_docs = [
                    type('Document', (), {
                        'page_content': doc['content'],
                        'metadata': doc['metadata']
                    })()
                    for doc in enhanced_docs
                ]
                logger.info(f"📚 Enhanced search retrieved {len(retrieved_docs)} documents")
            else:
                # Utiliser le retriever standard pour les cas simples
                retriever = self.vectorstore_manager.get_retriever(
                    search_kwargs={"k": search_k}
                )
                retrieved_docs = retriever.get_relevant_documents(question)
                logger.info(f"📚 Standard search retrieved {len(retrieved_docs)} documents")
            
            # Créer un prompt optimisé selon le type de question
            optimized_prompt = self._create_optimized_prompt(question, question_type, conversation_history)
            
            # Préparer le contexte avec les documents récupérés
            context = self._prepare_enhanced_context(retrieved_docs, question)
            
            # Utiliser le LLM directement avec le contexte préparé
            full_prompt = optimized_prompt.format(
                context=context,
                question=question
            )
            
            # Générer la réponse
            start_llm = time.time()
            llm_response = self.llm.invoke(full_prompt)
            
            # Extraire le contenu textuel de la réponse (compatible Groq et Ollama)
            if hasattr(llm_response, 'content'):
                result_answer = llm_response.content
            elif isinstance(llm_response, str):
                result_answer = llm_response
            else:
                result_answer = str(llm_response)
            
            llm_time = time.time() - start_llm
            
            logger.info(f"🤖 LLM response generated in {llm_time:.2f}s")
            
            response_time = time.time() - start_time

            # Traduction en wolof via Google Cloud Translate API (optionnel)
            wolof_answer = None
            
            try:
                # Vérifier que les credentials Google Cloud sont configurés  
                from app.utils.google_credentials import get_google_credentials_path
                credentials_path = get_google_credentials_path()
                
                if credentials_path:
                    from google.cloud import translate_v2 as translate
                    translate_client = translate.Client()
                    
                    # Traduire en wolof
                    translation = translate_client.translate(
                        result_answer, 
                        target_language="wo",
                        source_language="fr"
                    )
                    wolof_answer = translation["translatedText"]
                    logger.info(f"✅ Wolof translation completed: {len(wolof_answer)} characters")
                    
                    # Note: Audio generation is handled separately via dedicated endpoint
                    # No automatic audio generation to improve response time
                        
                else:
                    logger.warning("🔐 Google Cloud credentials not configured - Wolof translation disabled")
                    
            except ImportError:
                logger.warning("🔐 Google Cloud Translate not available - Wolof translation disabled")
            except Exception as translation_error:
                logger.warning(f"🔐 Wolof translation failed: {translation_error}")
                wolof_answer = None

            # Format response avec informations détaillées
            response = {
                "question": question,
                "answer": result_answer,
                "answer_wolof": wolof_answer,
                "audio_wolof_path": None,  # Audio will be generated separately when requested
                "audio_info": None,       # Audio info will be provided when audio is generated
                "response_time": response_time,
                "timestamp": time.time(),
                "model": self._get_current_model(),
                "search_strategy": question_type,
                "documents_analyzed": len(retrieved_docs)
            }

            if return_sources and retrieved_docs:
                response["sources"] = [
                    {
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        "metadata": doc.metadata,
                        "full_content": doc.page_content
                    }
                    for doc in retrieved_docs
                ]
                response["num_sources"] = len(retrieved_docs)

            logger.info(f"✅ Query processed successfully in {response_time:.2f}s - Strategy: {question_type}")
            return response

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "question": question,
                "answer": f"Erreur lors du traitement de la question: {str(e)}",
                "error": True,
                "response_time": time.time() - start_time,
                "model": self._get_current_model()
            }
    
    def _classify_question(self, question: str) -> str:
        """
        Classifie le type de question pour optimiser la stratégie de recherche
        """
        question_lower = question.lower().strip()
        
        # Détection des salutations
        if self._is_simple_greeting(question):
            return "greeting"
        
        # Détection des questions sur des cultures spécifiques
        specific_crops = [
            "pomme de terre", "riz", "mil", "arachide", "tomate", "oignon", 
            "chou", "maïs", "haricot", "niébé", "sorgho", "pastèque", "mangue"
        ]
        
        for crop in specific_crops:
            if crop in question_lower:
                return "specific_crop"
        
        # Détection des questions techniques
        technical_keywords = [
            "comment", "technique", "méthode", "procédure", "étapes", 
            "plantation", "culture", "irrigation", "fertilisation", "traitement",
            "maladie", "ravageur", "rendement", "production", "récolte"
        ]
        
        if any(keyword in question_lower for keyword in technical_keywords):
            return "technical"
        
        # Questions générales par défaut
        return "general"
    
    def _create_optimized_prompt(self, question: str, question_type: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Crée un prompt optimisé selon le type de question
        """
        # Formater l'historique de conversation
        history_context = ""
        if conversation_history:
            history_context = self._format_conversation_history(conversation_history)
        
        base_system = "Tu es Jaari, assistant agricole expert du Sénégal, spécialisé dans l'agriculture sénégalaise."
        
        if question_type == "greeting":
            return f"""{base_system}

{history_context}

Question : {{question}}

Si c'est une salutation, réponds brièvement en te présentant et en proposant ton aide.
Sinon, utilise le contexte suivant : {{context}}

Réponse concise :"""
        
        elif question_type == "specific_crop":
            return f"""{base_system} Tu excelles dans la connaissance des cultures spécifiques.

{history_context}

CONTEXTE DOCUMENTAIRE DÉTAILLÉ :
{{context}}

QUESTION SUR CULTURE SPÉCIFIQUE :
{{question}}

INSTRUCTIONS POUR RÉPONSE EXPERTE :
1. **ANALYSE COMPLÈTE** : Utilise TOUTES les informations disponibles dans le contexte
2. **STRUCTURE DÉTAILLÉE** :
   - Introduction avec contexte sénégalais
   - Variétés adaptées au Sénégal (si disponible)
   - Zones de culture (Niayes, Vallée, Casamance, etc.)
   - Calendrier cultural et saisons
   - Techniques de culture spécifiques
   - Rendements et production
   - Défis et solutions
   - Recommandations pratiques

3. **INFORMATIONS LOCALES** :
   - Mentionne les spécificités climatiques sénégalaises
   - Indique les pratiques locales recommandées
   - Cite les zones de production principales
   - Donne des chiffres concrets quand disponibles

4. **CONSEILS PRATIQUES** :
   - Étapes détaillées de mise en œuvre
   - Calendrier précis des opérations
   - Solutions aux problèmes courants
   - Sources d'approvisionnement et formation

Réponse experte et détaillée :"""
        
        elif question_type == "technical":
            return f"""{base_system} Tu maîtrises toutes les techniques agricoles.

{history_context}

CONTEXTE TECHNIQUE :
{{context}}

QUESTION TECHNIQUE :
{{question}}

INSTRUCTIONS POUR RÉPONSE TECHNIQUE :
1. **APPROCHE MÉTHODIQUE** : Explique la technique étape par étape
2. **ADAPTATION LOCALE** : Adapte les conseils au contexte sénégalais
3. **DÉTAILS PRATIQUES** : Donne des informations concrètes (doses, timing, coûts)
4. **ALTERNATIVES** : Propose plusieurs approches si possible
5. **PRÉVENTION** : Inclus les mesures préventives importantes

Réponse technique détaillée :"""
        
        else:  # général
            return f"""{base_system}

{history_context}

CONTEXTE :
{{context}}

QUESTION :
{{question}}

INSTRUCTIONS :
- Réponds de manière complète en utilisant toutes les informations disponibles
- Structure ta réponse clairement
- Adapte au contexte sénégalais
- Donne des conseils pratiques

Réponse structurée :"""
    
    def _prepare_enhanced_context(self, documents: List, question: str) -> str:
        """
        Prépare un contexte enrichi à partir des documents récupérés
        """
        if not documents:
            return "Aucun document pertinent trouvé."
        
        # Trier les documents par pertinence (si possible)
        # et limiter la taille totale du contexte
        max_context_length = self.vectorstore_config["max_context_length"]
        current_length = 0
        selected_docs = []
        
        for doc in documents:
            doc_content = f"\n--- SOURCE: {doc.metadata.get('source', 'Unknown')} ---\n{doc.page_content}\n"
            if current_length + len(doc_content) < max_context_length:
                selected_docs.append(doc_content)
                current_length += len(doc_content)
            else:
                # Ajouter partiellement le document si il reste de la place
                remaining_space = max_context_length - current_length
                if remaining_space > 200:  # Minimum viable
                    truncated_content = doc.page_content[:remaining_space-100] + "..."
                    doc_content = f"\n--- SOURCE: {doc.metadata.get('source', 'Unknown')} ---\n{truncated_content}\n"
                    selected_docs.append(doc_content)
                break
        
        logger.info(f"📄 Prepared context with {len(selected_docs)} documents ({current_length} chars)")
        
        return "\n".join(selected_docs)
    
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Process multiple queries"""
        results = []
        
        for question in questions:
            result = self.query(question)
            results.append(result)
        
        logger.info(f"Processed {len(questions)} batch queries")
        return results
    
    def load_documents_from_directory(self, directory_path: str, force_recreate: bool = False) -> bool:
        """
        Charge les documents depuis un dossier et crée le vectorstore optimisé.
        Args:
            directory_path (str): Chemin du dossier.
            force_recreate (bool): Forcer la recréation de l'index.
        Returns:
            bool: Succès du chargement.
        """
        try:
            documents = self.document_loader.load_from_directory(directory_path)
            if not documents:
                logger.warning(f"No documents found in {directory_path}")
                return False
                
            chunks = self.document_splitter.split_documents(documents)
            self.vectorstore_manager.create_vectorstore(chunks, force_recreate=force_recreate)
            logger.info(f"Successfully loaded {len(documents)} documents from {directory_path} (index_type=%s, batch_size=%d)", 
                       self.index_type, self.batch_size)
            return True
        except Exception as e:
            logger.error(f"Error loading documents from directory: {str(e)}")
            return False
    
    def load_documents_from_upload(self, uploaded_files, add_to_existing: bool = True) -> bool:
        """
        Charge les documents uploadés et met à jour le vectorstore optimisé.
        Args:
            uploaded_files: Fichiers uploadés.
            add_to_existing (bool): Ajouter au vectorstore existant ou créer un nouveau.
        Returns:
            bool: Succès du chargement.
        """
        try:
            documents = self.document_loader.load_from_uploaded_files(uploaded_files)
            if not documents:
                logger.warning("No documents loaded from upload")
                return False
                
            chunks = self.document_splitter.split_documents(documents)
            
            if add_to_existing and self.vectorstore_manager.vectorstore:
                self.vectorstore_manager.add_documents(chunks)
            else:
                vectorstore = self.vectorstore_manager.create_vectorstore(chunks)
                self.vectorstore_manager.vectorstore = vectorstore
                
            if self.llm:
                try:
                    self.setup_rag_chain()
                except Exception as chain_error:
                    logger.warning(f"Could not setup RAG chain: {chain_error}")
                    
            logger.info(f"Successfully loaded {len(documents)} documents from upload (index_type=%s, batch_size=%d)", 
                       self.index_type, self.batch_size)
            return True
        except Exception as e:
            logger.error(f"Error loading documents from upload: {str(e)}")
            return False
    
    def _enhanced_document_search(self, question: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Recherche de documents améliorée avec approche hybride
        """
        if not self.vectorstore_manager.vectorstore:
            return []
        
        k = k or self.vectorstore_config["search_k_enhanced"]
        
        try:
            # 1. Recherche vectorielle standard
            vector_results = self.vectorstore_manager.vectorstore.similarity_search_with_score(
                question, k=k
            )
            
            # 2. Extraction de mots-clés pour recherche supplémentaire
            key_terms = self._extract_key_terms(question)
            
            # 3. Recherche par mots-clés dans les métadonnées et contenu
            keyword_results = []
            if key_terms:
                for term in key_terms:
                    try:
                        term_results = self.vectorstore_manager.vectorstore.similarity_search(
                            term, k=int(k/2)
                        )
                        keyword_results.extend(term_results)
                    except:
                        continue
            
            # 4. Fusionner et dédupliquer les résultats
            all_docs = {}
            
            # Ajouter les résultats vectoriels avec scores
            for doc, score in vector_results:
                doc_key = f"{doc.page_content[:100]}_{doc.metadata.get('source', '')}"
                if doc_key not in all_docs:
                    all_docs[doc_key] = {
                        "document": doc,
                        "vector_score": float(score),
                        "relevance_score": float(score)
                    }
            
            # Ajouter les résultats par mots-clés
            for doc in keyword_results:
                doc_key = f"{doc.page_content[:100]}_{doc.metadata.get('source', '')}"
                if doc_key not in all_docs:
                    all_docs[doc_key] = {
                        "document": doc,
                        "vector_score": 1.0,  # Score par défaut
                        "relevance_score": 0.8  # Bonus pour correspondance mot-clé
                    }
                else:
                    # Augmenter le score si trouvé par les deux méthodes
                    all_docs[doc_key]["relevance_score"] *= 0.9
            
            # 5. Trier par score de pertinence et retourner les meilleurs
            sorted_results = sorted(
                all_docs.values(), 
                key=lambda x: x["relevance_score"]
            )[:k]
            
            logger.info(f"🔍 Enhanced search found {len(sorted_results)} relevant documents")
            
            return [
                {
                    "content": result["document"].page_content,
                    "metadata": result["document"].metadata,
                    "relevance_score": result["relevance_score"]
                }
                for result in sorted_results
            ]
            
        except Exception as e:
            logger.error(f"Enhanced document search failed: {str(e)}")
            # Fallback vers recherche simple
            try:
                simple_results = self.vectorstore_manager.vectorstore.similarity_search(question, k=k)
                return [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "relevance_score": 0.7
                    }
                    for doc in simple_results
                ]
            except:
                return []
    
    def _extract_key_terms(self, question: str) -> List[str]:
        """
        Extrait les termes clés d'une question pour améliorer la recherche
        """
        # Termes spécifiques à l'agriculture sénégalaise
        agriculture_terms = {
            "pomme de terre": ["pomme de terre", "tubercule", "plantation", "récolte"],
            "riz": ["riz", "riziculture", "paddy", "irrigation"],
            "mil": ["mil", "millet", "céréale", "sorgho"],
            "arachide": ["arachide", "cacahuète", "légumineuse"],
            "tomate": ["tomate", "maraîchage", "culture"],
            "oignon": ["oignon", "bulbe", "maraîchage"],
            "chou": ["chou", "khoulougné", "légume"],
            "maïs": ["maïs", "céréale", "épis"],
        }
        
        # Termes techniques agricoles
        technical_terms = [
            "variété", "variétés", "culture", "plantation", "semis", "récolte",
            "rendement", "production", "technique", "irrigation", "fertilisation",
            "pesticide", "maladie", "ravageur", "saison", "calendrier",
            "niayes", "vallée", "casamance", "sénégal"
        ]
        
        question_lower = question.lower()
        extracted_terms = []
        
        # Chercher les cultures spécifiques
        for crop, related_terms in agriculture_terms.items():
            if crop in question_lower:
                extracted_terms.extend(related_terms)
        
        # Ajouter les termes techniques présents
        for term in technical_terms:
            if term in question_lower:
                extracted_terms.append(term)
        
        # Ajouter les mots importants de la question (plus de 3 caractères)
        question_words = [
            word.strip(".,!?;:()[]{}\"'")
            for word in question_lower.split()
            if len(word) > 3 and word.isalpha()
        ]
        extracted_terms.extend(question_words)
        
        # Retourner les termes uniques
        return list(set(extracted_terms))[:10]  # Limiter à 10 termes max
    
    def search_documents(
        self, 
        query: str, 
        k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search similar documents with enhanced search"""
        if not self.vectorstore_manager.vectorstore:
            raise ValueError("Vectorstore not initialized")
        
        try:
            # Utiliser la recherche améliorée
            enhanced_results = self._enhanced_document_search(query, k)
            
            # Filtrer par seuil de score
            filtered_results = [
                result for result in enhanced_results
                if result["relevance_score"] >= score_threshold
            ]
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Document search failed: {str(e)}")
            raise
    
    def update_configuration(self, updates: Dict[str, Any]) -> None:
        """Update engine configuration"""
        try:
            # Update document splitter if chunk parameters changed
            if "document" in updates:
                doc_config = updates["document"]
                if "chunk_size" in doc_config or "chunk_overlap" in doc_config:
                    self.document_splitter.update_parameters(
                        doc_config.get("chunk_size", self.document_splitter.chunk_size),
                        doc_config.get("chunk_overlap", self.document_splitter.chunk_overlap)
                    )
            
            # Update LLM if parameters changed
            if "llm" in updates:
                self.llm_config.update(updates["llm"])
                self.llm = None  # Force re-initialization
                self.qa_chain = None
            
            logger.info("Configuration updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating configuration: {str(e)}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            vectorstore_info = self.vectorstore_manager.get_vectorstore_info()
            
            # Vérifier si le vectorstore est vraiment chargé
            vectorstore_loaded = (self.vectorstore_manager.vectorstore is not None and 
                                vectorstore_info.get('exists', False))
            
            status = {
                "llm_initialized": self.llm is not None,
                "llm_provider": self.llm_provider,
                "llm_model": self._get_current_model(),
                "qa_chain_ready": self.qa_chain is not None and vectorstore_loaded,
                "vectorstore": {
                    **vectorstore_info,
                    "loaded": vectorstore_loaded
                },
                "document_splitter": {
                    "chunk_size": self.document_splitter.chunk_size,
                    "chunk_overlap": self.document_splitter.chunk_overlap
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {
                "llm_initialized": False,
                "llm_model": None,
                "qa_chain_ready": False,
                "vectorstore": {"loaded": False, "exists": False, "num_documents": 0},
                "document_splitter": {"chunk_size": 1000, "chunk_overlap": 200}
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status (alias for API compatibility)"""
        status = self.get_system_status()
        return {
            "initialized": self.initialized,
            "llm_available": status["llm_initialized"],
            "llm_provider": self.llm_provider,
            "embeddings_available": self.vectorstore_manager.embeddings is not None,
            "vectorstore_available": status["vectorstore"]["loaded"],
            "qa_chain_available": status["qa_chain_ready"],
            "documents_count": status["vectorstore"].get("num_documents", 0),
            "model_name": status["llm_model"],
            "embeddings_model": getattr(self.vectorstore_manager, 'embeddings_config', {}).get("model_name", "N/A")
        }
    
    def ensure_vectorstore_loaded(self) -> bool:
        """Ensure vectorstore is loaded and ready"""
        try:
            # Vérifier si le vectorstore existe déjà
            if self.vectorstore_manager.vectorstore is not None:
                return True
            
            # Essayer de charger depuis le disque
            if self.vectorstore_manager.load_vectorstore():
                logger.info("Vectorstore loaded from disk")
                return True
            
            # Pas de vectorstore disponible
            logger.warning("No vectorstore available - please load documents first")
            return False
            
        except Exception as e:
            logger.error(f"Error ensuring vectorstore loaded: {str(e)}")
            return False
    
    def get_vectorstore_statistics(self) -> Dict[str, Any]:
        """Get detailed vectorstore statistics"""
        try:
            if not self.vectorstore_manager.vectorstore:
                return {"available": False}
            
            vectorstore_info = self.vectorstore_manager.get_vectorstore_info()
            
            # Essayer d'obtenir des statistiques supplémentaires
            stats = {
                "available": True,
                **vectorstore_info,
                "embedding_model": getattr(self.vectorstore_manager, 'embeddings_config', {}).get("model_name", "N/A")
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting vectorstore statistics: {str(e)}")
            return {"available": False, "error": str(e)}

    async def build_vectorstore_from_directory(self, directory_path: str) -> bool:
        """Build vectorstore from documents in a directory asynchronously"""
        try:
            logger.info(f"🔄 Building vectorstore from directory: {directory_path}")
            
            # Load documents from directory
            documents = self.document_loader.load_from_directory(directory_path)
            
            if not documents:
                logger.warning("❌ No documents found in directory")
                return False
                
            logger.info(f"📄 Loaded {len(documents)} documents")
            
            # Split documents with intelligent chunking
            logger.info("🧠 Applying intelligent chunking...")
            chunks = self.document_splitter.split_documents(documents)
            logger.info(f"✂️ Created {len(chunks)} intelligent chunks")
            
            # Create vectorstore
            logger.info("🔄 Creating vectorstore...")
            vectorstore = self.vectorstore_manager.create_vectorstore(chunks)
            
            if vectorstore:
                # Save vectorstore
                logger.info("💾 Saving vectorstore...")
                self.vectorstore_manager.save_vectorstore()
                
                # Setup QA chain if LLM is available
                if self.llm:
                    await self._setup_qa_chain_async()
                    
                logger.info("✅ Vectorstore built and saved successfully")
                return True
            else:
                logger.error("❌ Failed to create vectorstore")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to build vectorstore: {str(e)}")
            return False

    def _create_dynamic_prompt(self, question: str, concepts: Dict, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Crée un prompt dynamique adapté aux concepts détectés dans la question et incluant l'historique
        """
        # Formater l'historique de conversation
        history_context = ""
        if conversation_history:
            history_context = self._format_conversation_history(conversation_history)
        
        base_prompt = f"""Tu es Jaari, assistant agricole expert du Sénégal, spécialisé dans la vulgarisation claire et rapide.

{history_context}

Contexte fourni :
{{context}}

Question actuelle de l'utilisateur :
{{question}}

"""
        
        # Adaptations selon les concepts détectés
        if concepts.get('crops'):
            crops = concepts['crops']
            if 'pomme_de_terre' in crops:
                base_prompt += """INSTRUCTIONS SPÉCIALES POMME DE TERRE :
- Tu as accès aux documents TPS (Techniques de Production de Semences) spécialisés
- Concentre-toi sur les informations pratiques : variétés, cycle de production, techniques culturales
- Mentionne les spécificités pour le Sénégal (saisons, zones adaptées, rendements)
- Structure ta réponse avec des sections claires : Culture, Production, Techniques

"""
            elif 'haricot' in crops:
                base_prompt += """INSTRUCTIONS SPÉCIALES HARICOT :
- Utilise les documents TPS HARICOT pour donner des informations précises
- Focus sur les techniques de production, les variétés adaptées au Sénégal
- Mentionne les cycles de culture et les rendements attendus

"""
            # Autres cultures...
        
        if concepts.get('techniques'):
            if 'production' in concepts['techniques'] or 'semences' in concepts['techniques']:
                base_prompt += """FOCUS TECHNIQUES DE PRODUCTION :
- Donne des étapes concrètes et pratiques
- Mentionne les calendriers, durées de cycle, rendements
- Inclus les recommandations spécifiques pour le contexte sénégalais

"""
        
        base_prompt += """CONSIGNES GÉNÉRALES :
- Réponds UNIQUEMENT sur la base du contexte fourni
- Structure ta réponse clairement avec des titres et listes
- Donne des informations pratiques et actionnables
- Si l'information n'est pas dans le contexte, dis-le clairement
- Termine par une recommandation principale

Réponse structurée :"""

        return base_prompt
    
    def _get_current_model(self) -> Optional[str]:
        """Get the currently active model name"""
        if self.llm_provider == "groq" and self.groq_client:
            return self.groq_config["model"]
        elif self.llm_provider == "ollama":
            return self.llm_config["model"]
        else:
            return None


# Global RAG engine instance
rag_engine = RAGEngine()
