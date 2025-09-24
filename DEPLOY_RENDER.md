# 🚀 Guide de Déploiement Render - Jaari RAG API

Ce guide explique comment déployer Jaari RAG API sur [Render](https://render.com).

## 📋 Prérequis

1. **Compte Render** : Créer un compte sur [render.com](https://render.com)
2. **Dépôt Git** : Code source sur GitHub/GitLab
3. **Variables d'environnement** : Clés API nécessaires

## 🛠️ Configuration Automatique (Recommandée)

### Option 1 : Déploiement avec render.yaml

1. **Push ton code** vers ton dépôt Git avec le fichier `render.yaml` inclus
2. **Connecte ton dépôt** sur Render Dashboard
3. **Configure les variables d'environnement** (voir section Variables ci-dessous)
4. **Déploie automatiquement**

### Option 2 : Déploiement Manuel

1. **Nouveau Web Service**
   - Va sur [Render Dashboard](https://dashboard.render.com)
   - Clique sur "New +" → "Web Service"
   - Connecte ton dépôt Git

2. **Configuration du Service**
   ```
   Name: jaari-rag-api
   Environment: Docker
   Region: Oregon (ou plus proche de toi)
   Branch: main
   Dockerfile Path: ./Dockerfile
   ```

3. **Configuration des Variables d'Environnement**
   ```bash
   # Required
   GROQ_API_KEY=your_groq_api_key_here
   LLM_PROVIDER=groq
   
   # Database (PostgreSQL fourni par Render)
   DATABASE_URL=${{DATABASE_URL}}  # Auto-générée par Render
   
   # Application
   ENVIRONMENT=production
   DEBUG=false
   LOG_LEVEL=INFO
   
   # Authentication
   SECRET_KEY=your_secret_key_here
   DEFAULT_ADMIN_EMAIL=admin@jaari.com
   DEFAULT_ADMIN_USERNAME=admin
   DEFAULT_ADMIN_PASSWORD=secure_password_here
   DEFAULT_ADMIN_FULL_NAME=Jaari Administrator
   
   # Optional - Redis
   REDIS_URL=${{REDIS_URL}}  # Si tu utilises Redis de Render
   ```

## 🗄️ Base de Données

### PostgreSQL (Recommandée)

1. **Créer une base PostgreSQL**
   - Dashboard Render → "New +" → "PostgreSQL"
   - Name: `jaari-postgres`
   - Region: Same as your web service

2. **Connecter à ton Web Service**
   - Dans les paramètres de ton web service
   - Environment Variables → Add
   - `DATABASE_URL` = `${{DATABASE_URL}}`

### Alternative : Base SQLite (Développement seulement)
```bash
DATABASE_URL=sqlite:///app/data/jaari.db
```

## 🚨 Variables d'Environnement Obligatoires

### Render Dashboard → Ton Service → Environment

```bash
# 🤖 LLM Provider (OBLIGATOIRE)
GROQ_API_KEY=gsk_your_groq_api_key_here
LLM_PROVIDER=groq

# 🔐 Security (OBLIGATOIRE)
SECRET_KEY=ton_secret_key_super_securise_ici_minimum_32_chars

# 👤 Admin par défaut
DEFAULT_ADMIN_EMAIL=ton_email@exemple.com
DEFAULT_ADMIN_USERNAME=admin
DEFAULT_ADMIN_PASSWORD=motdepasse_securise
DEFAULT_ADMIN_FULL_NAME=Ton Nom Complet

# 🌍 Application
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# 🗄️ Base de données (Auto-configurée si tu uses PostgreSQL Render)
DATABASE_URL=${{DATABASE_URL}}
```

## 🔧 Configuration des Services Externes

### 1. Obtenir une clé API Groq
1. Va sur [console.groq.com](https://console.groq.com)
2. Créer un compte / Se connecter
3. Generate API Key
4. Copie la clé dans `GROQ_API_KEY` sur Render

### 2. Google Cloud Translation (Optionnel)
Si tu veux les traductions Wolof :
1. Télécharge ton fichier JSON de credentials Google Cloud
2. Sur Render, tu peux l'ajouter comme "Secret File" ou utiliser les variables d'environnement

## 🚀 Déploiement

### Automatique (avec render.yaml)
```bash
git add .
git commit -m "Ready for Render deployment"
git push origin main
```

### Manuel
1. Push ton code vers ton dépôt
2. Sur Render Dashboard → Ton service → "Manual Deploy"

## 📊 Monitoring et Logs

### Health Check
Ton API aura automatiquement un health check sur `/health`

### Logs
```bash
# Voir les logs en temps réel
Render Dashboard → Ton Service → Logs
```

### Métriques
- `/health` - Status de l'API
- `/metrics` - Métriques Prometheus (si configuré)

## 🔗 URLs Importantes

Après déploiement, ton API sera disponible sur :
```
https://ton-service-name.onrender.com

# Endpoints
https://ton-service-name.onrender.com/docs          # Swagger UI
https://ton-service-name.onrender.com/health        # Health check
https://ton-service-name.onrender.com/api/v1/       # API base
```

## 🐛 Troubleshooting

### Problème : Build Failed
```bash
# Check les logs de build sur Render Dashboard
# Vérifier que le Dockerfile est correct
# S'assurer que requirements.txt est à jour
```

### Problème : Health Check Failed
```bash
# Vérifier les variables d'environnement
# Check les logs de l'application
# Tester l'endpoint /health localement
```

### Problème : Database Connection
```bash
# Vérifier que DATABASE_URL est correctement configurée
# S'assurer que PostgreSQL service est actif
# Check les logs pour des erreurs de connexion
```

## 🔄 Mise à Jour

### Auto-Deploy
Si configuré, chaque push sur `main` déclenche un nouveau déploiement.

### Manuel
Render Dashboard → Ton Service → "Manual Deploy"

## 💰 Estimation des Coûts

### Plan Starter (Gratuit)
- Web Service: Gratuit (avec limitations)
- PostgreSQL: Gratuit (1GB)
- Idéal pour développement/test

### Plan Standard ($7/mois)
- Web Service: $7/mois
- PostgreSQL: $7/mois  
- Idéal pour production

## 📞 Support

En cas de problème :
1. Check les logs sur Render Dashboard
2. Vérifier la documentation officielle Render
3. Tester localement avec Docker
4. Contacter le support Render si nécessaire

---

🌾 **Bonne chance avec ton déploiement Jaari RAG API !**
