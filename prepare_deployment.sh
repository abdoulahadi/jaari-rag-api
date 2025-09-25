#!/bin/bash

# Script de préparation pour le déploiement sur Render
echo "🚀 Préparation du déploiement Jaari RAG API sur Render"
echo "================================================="

# Vérifier que nous sommes dans le bon répertoire
if [ ! -f "app/main.py" ]; then
    echo "❌ Erreur : Ce script doit être exécuté depuis la racine du projet"
    exit 1
fi

echo "✅ Répertoire correct détecté"

# Génerer une SECRET_KEY
echo ""
echo "📋 SECRET_KEY générée pour Render :"
echo "================================================="
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
echo "================================================="

# Rechercher le fichier Google Cloud credentials
echo ""
echo "🔍 Recherche des credentials Google Cloud..."

if [ -f "$HOME/backup_sensitive_files/translate-jaari-065fa764be8a.json" ]; then
    echo "📋 Valeur base64 pour GOOGLE_APPLICATION_CREDENTIALS_JSON :"
    echo "================================================="
    base64 -i "$HOME/backup_sensitive_files/translate-jaari-065fa764be8a.json"
    echo "================================================="
elif [ -f "translate-jaari-065fa764be8a.json" ]; then
    echo "📋 Valeur base64 pour GOOGLE_APPLICATION_CREDENTIALS_JSON :"
    echo "================================================="
    base64 -i translate-jaari-065fa764be8a.json
    echo "================================================="
else
    echo "⚠️  Fichier Google Cloud credentials non trouvé"
    echo "   Vérifiez : $HOME/backup_sensitive_files/"
fi

echo ""
echo "🌟 OPTIONS DE DÉPLOIEMENT RENDER :"
echo "=================================="
echo ""
echo "📦 OPTION 1 - Build Python natif (RECOMMANDÉ) :"
echo "- Runtime: Python 3"
echo "- Build Command: pip install -r requirements.txt"
echo "- Start Command: python -m uvicorn app.main:app --host 0.0.0.0 --port \$PORT"
echo ""
echo "🐳 OPTION 2 - Build Docker :"
echo "- Runtime: Docker"
echo "- Build Command: (automatique avec Dockerfile)"
echo "- Start Command: (automatique avec Dockerfile)"
echo ""
echo "📝 Variables d'environnement OBLIGATOIRES :"
echo "============================================"
echo "GROQ_API_KEY = [votre clé Groq]"
echo "GOOGLE_APPLICATION_CREDENTIALS_JSON = [valeur base64 ci-dessus]" 
echo "SECRET_KEY = [valeur générée ci-dessus]"
echo "DATABASE_URL = [URL PostgreSQL de Render - auto-générée]"
echo ""
echo "📝 Variables optionnelles :"
echo "ENVIRONMENT = production"
echo "DEBUG = False"
echo "LOG_LEVEL = INFO"
echo "DEFAULT_ADMIN_EMAIL = admin@jaari.com"
echo "DEFAULT_ADMIN_USERNAME = admin"
echo "DEFAULT_ADMIN_PASSWORD = [mot de passe sécurisé]"
echo ""
echo "✅ Prêt pour le déploiement !"