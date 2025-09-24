#!/bin/bash
# Script de démarrage pour Jaari RAG API

echo "🚀 Démarrage de Jaari RAG API..."

# Aller dans le répertoire du projet
cd "$(dirname "$0")"

# Fonction pour choisir le provider LLM
choose_llm_provider() {
    echo ""
    echo "🤖 Choisissez votre provider LLM :"
    echo "1) Groq (rapide, cloud) - [PAR DÉFAUT]"
    echo "2) Ollama (local)"
    echo ""
    
    # Timeout de 10 secondes, défaut = Groq
    echo "Votre choix (1-2) ? [défaut: 1 dans 10s]"
    read -t 10 choice
    
    case $choice in
        2)
            echo "✅ Ollama sélectionné"
            export LLM_PROVIDER=ollama
            ;;
        1|"")
            echo "✅ Groq sélectionné (défaut)"
            export LLM_PROVIDER=groq
            ;;
        *)
            echo "⚠️ Choix invalide, utilisation de Groq par défaut"
            export LLM_PROVIDER=groq
            ;;
    esac
}

# Charger les variables d'environnement depuis .env
if [ -f .env ]; then
    echo "📝 Chargement des variables d'environnement depuis .env"
    # Utiliser une méthode plus sûre pour charger les variables avec des espaces
    while IFS='=' read -r key value; do
        # Ignorer les lignes vides et les commentaires
        if [[ ! -z "$key" && ! "$key" =~ ^[[:space:]]*# ]]; then
            # Supprimer les espaces autour de la clé
            key=$(echo "$key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            # Supprimer les espaces au début de la valeur et les guillemets
            value=$(echo "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//;s/^"//;s/"$//')
            export "$key=$value"
        fi
    done < .env
fi

# Choisir le provider LLM
choose_llm_provider

# Afficher le provider choisi
echo "🔧 Provider LLM configuré: $LLM_PROVIDER"

# Vérifications spécifiques selon le provider
if [ "$LLM_PROVIDER" = "groq" ]; then
    echo "🌐 Configuration Groq..."
    if [ -z "$GROQ_API_KEY" ]; then
        echo "❌ GROQ_API_KEY n'est pas définie !"
        echo "💡 Ajoutez votre clé API Groq dans le fichier .env"
        exit 1
    fi
    echo "✅ Clé API Groq configurée"
elif [ "$LLM_PROVIDER" = "ollama" ]; then
    echo "🏠 Configuration Ollama..."
    echo "💡 Assurez-vous qu'Ollama est démarré avec: ollama serve"
    echo "📋 Modèles disponibles: ollama list"
fi

# Vérifier que Google Cloud est configuré
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "⚠️  GOOGLE_APPLICATION_CREDENTIALS n'est pas défini"
    export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/translate-jaari-065fa764be8a.json"
    echo "✅ Configuré automatiquement: $GOOGLE_APPLICATION_CREDENTIALS"
fi

# Vérifier que le fichier existe
if [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "❌ Fichier de credentials Google Cloud introuvable: $GOOGLE_APPLICATION_CREDENTIALS"
    exit 1
fi

echo "✅ Google Cloud Translate configuré"
echo "🌐 Démarrage du serveur sur http://localhost:8000"

# Démarrer le serveur
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
