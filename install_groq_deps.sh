#!/bin/bash
# Installation des dépendances Groq pour Jaari RAG API

echo "📦 Installation des dépendances Groq..."

# Aller dans le répertoire du projet
cd "$(dirname "$0")"

# Installer les nouvelles dépendances
echo "🔄 Installation de groq et langchain-groq..."

pip install groq>=0.4.0
pip install langchain-groq>=0.2.0

# Vérifier l'installation
echo "🧪 Vérification de l'installation..."

python -c "
try:
    import groq
    print('✅ groq installé')
except ImportError:
    print('❌ groq non installé')

try:
    import langchain_groq
    print('✅ langchain-groq installé')
except ImportError:
    print('❌ langchain-groq non installé')

try:
    from groq import Groq
    print('✅ Client Groq importable')
except ImportError as e:
    print(f'❌ Erreur import Client Groq: {e}')

try:
    from langchain_groq import ChatGroq
    print('✅ ChatGroq importable')
except ImportError as e:
    print(f'❌ Erreur import ChatGroq: {e}')
"

echo ""
echo "✅ Installation terminée !"
echo "💡 Vous pouvez maintenant tester avec: python test_groq_integration.py"
