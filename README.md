# 🚀 Jaari RAG API - Agriculture Intelligence Platform

**Version:** 1.0.0  
**Status:** En Développement  
**Author:** Xelkoom Team  

## 📖 Description

**Jaari RAG API** est une plateforme API complète basée sur FastAPI qui offre des services d'intelligence artificielle pour l'agriculture africaine. Cette API combine la puissance des modèles de langage (LLM) avec une base de connaissances spécialisée pour fournir des conseils agricoles précis et contextuels.

## 🏗️ Architecture Technique

### 🔧 Stack Technologique
- **Backend:** FastAPI 0.104.1 avec Python 3.9+
- **Base de données:** SQLAlchemy avec support PostgreSQL/SQLite
- **Authentification:** JWT avec refresh tokens
- **Cache:** Redis pour les sessions et le cache
- **AI/ML:** Ollama (Llama 3.1:8b) + Sentence Transformers
- **Vector Store:** FAISS pour la recherche sémantique
- **Task Queue:** Celery pour le traitement asynchrone
- **Monitoring:** Prometheus pour les métriques

### 🗂️ Structure du Projet
```
app/
├── main.py                 # Point d'entrée FastAPI
├── config/                 # Configuration (DB, settings)
├── core/                   # Services centraux (auth, RAG)
├── models/                 # Modèles SQLAlchemy
├── schemas/                # Schémas Pydantic
├── services/               # Services métier
├── api/                    # Routes API
│   ├── v1/                # API v1 (auth, chat, users)
│   ├── documents.py       # Gestion documentaire
│   ├── analytics.py       # Analytics
│   └── websocket.py       # WebSocket temps réel
└── utils/                  # Utilitaires et middleware
```

## ✨ Fonctionnalités Implémentées

### 🔐 Authentification & Sécurité
- **JWT Authentication** avec access/refresh tokens
- **Gestion des rôles** (Admin, Expert, User)
- **Hashage sécurisé** des mots de passe (bcrypt)
- **Middleware de sécurité** et headers sécurisés
- **Protection contre les attaques** (CORS, validation)

### 💬 Chat Intelligence
- **API REST complète** pour les conversations
- **Intégration RAG** avec contexte agricole
- **Historique des conversations** persistant
- **Système de feedback** pour les messages
- **Support WebSocket** (en cours de développement)

### � Gestion Utilisateurs
- **Inscription/Connexion** utilisateur
- **Profils utilisateur** avec métadonnées
- **Gestion des sessions** et tokens
- **Système de rôles** hiérarchique

### 📚 Base de Connaissances
- **Corpus agricole** préchargé (sorgho, niébé, etc.)
- **Vectorisation automatique** des documents
- **Recherche sémantique** optimisée
- **Processing asynchrone** des documents

### 📊 Infrastructure & Monitoring
- **Health checks** complets (/health, /ready)
- **Métriques Prometheus** (/metrics)
- **Logs structurés** avec niveaux configurables
- **Middleware de monitoring** des performances

## 🚀 Installation et Configuration

### Prérequis
- **Python 3.9+** (testé avec Python 3.11)
- **PostgreSQL 14+** ou SQLite (dev)
- **Redis 6+** pour le cache et Celery
- **Ollama** installé avec le modèle Llama 3.1:8b
- **Docker & Docker Compose** (optionnel)

### 1. Installation Locale

#### Cloner le projet
```bash
git clone <repo-url>
cd jaari-rag-api
```

#### Créer l'environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

#### Installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. Configuration

#### Variables d'environnement
```bash
cp .env.example .env
```

Modifier le fichier `.env` avec vos paramètres :
```bash
# Base de données
DATABASE_URL=sqlite:///./data/jaari_rag.db  # Dev
# DATABASE_URL=postgresql://user:pass@localhost:5432/jaari_rag_db  # Prod

# JWT
SECRET_KEY=your-super-secret-key-here

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# Redis
REDIS_URL=redis://localhost:6379/0
```

### 3. Initialisation

#### Préparer la base de données
```bash
# Créer les répertoires de données
mkdir -p data logs uploads cache

# Initialiser la base de données (si Alembic configuré)
# alembic upgrade head
```

#### Démarrer Ollama et télécharger le modèle
```bash
ollama serve
ollama pull llama3.1:8b
```

### 4. Lancement

#### Mode développement
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Mode production
```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

#### Avec Docker Compose
```bash
# Développement
docker-compose -f docker-compose.dev.yml up --build

# Production
docker-compose up -d --build
```

### 5. Vérification

Accéder à l'API :
- **Documentation:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health
- **Métriques:** http://localhost:8000/metrics

## 📊 Documentation API

### URL de Base
```
http://localhost:8000/api/v1
```

### 🔐 Authentification

#### Inscription utilisateur
```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "username": "username",
  "password": "securepassword",
  "full_name": "Nom Complet",
  "organization": "Organisation"
}
```

#### Connexion
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword"
}
```

**Réponse:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

#### Autres endpoints d'authentification
```http
POST   /api/v1/auth/refresh          # Renouveler le token
POST   /api/v1/auth/logout           # Déconnexion
GET    /api/v1/auth/me               # Profil utilisateur
POST   /api/v1/auth/change-password  # Changer mot de passe
```

### 💬 Chat & Conversations

#### Lister les conversations
```http
GET /api/v1/chat/conversations?skip=0&limit=20
Authorization: Bearer {access_token}
```

#### Créer une conversation
```http
POST /api/v1/chat/conversations
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "title": "Questions sur le sorgho",
  "description": "Conseils pour la culture du sorgho"
}
```

#### Poser une question
```http
POST /api/v1/chat/ask
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "conversation_id": 1,
  "message": "Comment améliorer le rendement du sorgho ?"
}
```

#### Question rapide (sans conversation)
```http
POST /api/v1/chat/quick
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "question": "Quand planter le niébé ?"
}
```

#### Feedback sur un message
```http
POST /api/v1/chat/messages/{message_id}/feedback
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "rating": 5,
  "comment": "Réponse très utile"
}
```

#### Autres endpoints chat
```http
GET    /api/v1/chat/conversations/{id}    # Détail conversation avec messages
GET    /api/v1/chat/stats                 # Statistiques utilisateur
DELETE /api/v1/chat/conversations/{id}    # Supprimer conversation
```

### 👥 Gestion Utilisateurs

```http
GET /api/v1/users/me                # Profil utilisateur actuel
GET /api/v1/users/                  # Liste utilisateurs (admin)
```

### 📚 Documents & Corpus

```http
GET  /api/v1/documents/             # Liste documents (à implémenter)
POST /api/v1/documents/upload       # Upload document (à implémenter)

GET /api/v1/corpus/status           # Statut du corpus
GET /api/v1/corpus/                 # Informations corpus
```

### 📊 Analytics & Monitoring

```http
GET /api/v1/analytics/              # Analytics (à implémenter)
GET /health                         # Health check
GET /ready                          # Readiness check
GET /metrics                        # Métriques Prometheus
```

### 🔌 WebSocket

```
WS /ws/chat                         # Chat temps réel (en développement)
```

### 📝 Codes de Réponse

| Code | Description |
|------|-------------|
| 200  | Succès |
| 201  | Créé avec succès |
| 400  | Requête invalide |
| 401  | Non autorisé |
| 403  | Accès interdit |
| 404  | Ressource non trouvée |
| 422  | Erreur de validation |
| 500  | Erreur serveur |

## 🐳 Déploiement avec Docker

### Configuration Docker

Le projet inclut une configuration Docker complète pour le développement et la production.

#### Fichiers Docker
- `Dockerfile` - Image de l'application
- `docker-compose.yml` - Configuration production
- `docker-compose.dev.yml` - Configuration développement

#### Services inclus
- **jaari-api** - Application FastAPI
- **redis** - Cache et broker Celery
- **celery-worker** - Traitement asynchrone (optionnel)

### Déploiement rapide

```bash
# Production
docker-compose up -d --build

# Développement avec hot-reload
docker-compose -f docker-compose.dev.yml up --build
```

### Variables d'environnement Docker

```yaml
environment:
  - DATABASE_URL=sqlite:///./data/jaari_rag.db
  - SECRET_KEY=your-super-secret-key
  - OLLAMA_BASE_URL=http://host.docker.internal:11434
  - REDIS_URL=redis://redis:6379/0
```

## 🔧 Développement

### Structure des Modèles de Données

#### Utilisateur (`User`)
```python
class User:
    id: int
    email: str (unique)
    username: str (unique)
    hashed_password: str
    full_name: str
    role: UserRole (ADMIN, EXPERT, USER)
    is_active: bool
    organization: str
    created_at: datetime
```

#### Conversation (`Conversation`)
```python
class Conversation:
    id: int
    title: str
    description: str
    user_id: int (FK)
    is_active: bool
    total_messages: int
    created_at: datetime
    messages: List[Message]
```

#### Message (`Message`)
```python
class Message:
    id: int
    conversation_id: int (FK)
    content: str
    response: str
    role: MessageRole (USER, ASSISTANT)
    sources: List[str]
    tokens_used: int
    created_at: datetime
```

#### Document (`Document`)
```python
class Document:
    id: int
    filename: str
    file_path: str
    file_size: int
    file_type: DocumentType (PDF, TXT, DOCX, MD)
    status: DocumentStatus (UPLOADED, PROCESSING, INDEXED, FAILED)
    uploaded_by_id: int (FK)
    created_at: datetime
```

### Services Principaux

#### RAG Engine (`enhanced_rag_engine.py`)
- Intégration avec le corpus agricole
- Recherche sémantique avec FAISS
- Processing des documents
- Interface avec Ollama

#### Chat Service (`chat_service.py`)
- Gestion des conversations
- Historique des messages
- Intégration RAG pour les réponses

#### User Service (`user_service.py`)
- Authentification et autorisation
- Gestion des profils utilisateur
- Sessions et tokens

### Middleware Personnalisés

- **SecurityHeadersMiddleware** - Headers de sécurité
- **LoggingMiddleware** - Logs des requêtes
- **RateLimitMiddleware** - Limitation du taux (désactivé temporairement)

## 🧪 Tests et Validation

### Tests unitaires
```bash
# Installation des dépendances de test
pip install pytest pytest-asyncio httpx

# Exécution des tests
pytest tests/
```

### Tests manuels avec curl
```bash
# Health check
curl http://localhost:8000/health

# Inscription
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","username":"test","password":"password123","full_name":"Test User"}'

# Connexion
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password123"}'
```

## 🔍 Monitoring et Logs

### Métriques Prometheus
Accès via `/metrics` pour :
- Nombre de requêtes par endpoint
- Temps de réponse
- Erreurs HTTP
- Utilisation mémoire

### Logs structurés
Configuration dans `settings.py` :
```python
LOG_LEVEL=INFO
LOG_FORMAT=json  # ou "standard"
```

### Health Checks
- `/health` - Statut général de l'application
- `/ready` - Disponibilité pour recevoir du trafic

## 🚀 Fonctionnalités en Développement

### ⏳ En cours
- [ ] **WebSocket temps réel** pour le chat
- [ ] **Upload de documents** utilisateur
- [ ] **Analytics avancées** des conversations
- [ ] **Système de recommandations**

### 📋 Planifié
- [ ] **Multi-langues** (Anglais, Wolof)
- [ ] **API de recherche** sémantique
- [ ] **Intégration S3** pour le stockage
- [ ] **Dashboard admin** complet
- [ ] **Notifications push**
- [ ] **Export des conversations**

## ⚠️ Limitations Actuelles

1. **WebSocket** - Implémentation basique, pas de chat temps réel complet
2. **Upload documents** - Endpoints définis mais non implémentés
3. **Rate limiting** - Middleware désactivé temporairement
4. **Tests** - Couverture de tests à améliorer
5. **Alembic** - Migrations de base de données à configurer

## 🤝 Contribution

### Pour contribuer :
1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commit les changements (`git commit -am 'Ajout nouvelle fonctionnalité'`)
4. Push vers la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Créer une Pull Request

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 📞 Support

- **Documentation:** http://localhost:8000/docs
- **Issues:** Créer un issue sur le repository
- **Email:** support@xelkoom.com

---

**Développé avec ❤️ par l'équipe Xelkoom pour l'agriculture africaine**

*Dernière mise à jour: Août 2025*
# Force rebuild for Render native Python
