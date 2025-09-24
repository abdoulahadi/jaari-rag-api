#!/bin/bash
# Script d'aide pour configurer l'environnement Jaari RAG API

echo "Configuration de l'environnement Jaari RAG API..."

# Chemin vers le projet
PROJECT_DIR="/Users/caambdiop/Desktop/first doc/Projet Xelkoom/AGRO_SPACE/jaari-rag-api"

# Configuration Google Cloud
export GOOGLE_APPLICATION_CREDENTIALS="$PROJECT_DIR/translate-jaari-065fa764be8a.json"

# Option 1: Ajouter au profil shell de façon permanente
echo "Voulez-vous ajouter GOOGLE_APPLICATION_CREDENTIALS à votre profil shell de façon permanente? (y/n)"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    # Détecter le shell utilisé
    if [[ $SHELL == *"zsh"* ]]; then
        PROFILE_FILE="$HOME/.zshrc"
    elif [[ $SHELL == *"bash"* ]]; then
        PROFILE_FILE="$HOME/.bash_profile"
    else
        PROFILE_FILE="$HOME/.profile"
    fi
    
    # Ajouter la variable d'environnement
    echo "" >> "$PROFILE_FILE"
    echo "# Jaari RAG API - Google Cloud Translate" >> "$PROFILE_FILE"
    echo "export GOOGLE_APPLICATION_CREDENTIALS=\"$PROJECT_DIR/translate-jaari-065fa764be8a.json\"" >> "$PROFILE_FILE"
    
    echo "✅ Variable ajoutée à $PROFILE_FILE"
    echo "Redémarrez votre terminal ou exécutez: source $PROFILE_FILE"
else
    echo "Configuration temporaire uniquement pour cette session."
fi

echo "✅ GOOGLE_APPLICATION_CREDENTIALS configuré: $GOOGLE_APPLICATION_CREDENTIALS"
