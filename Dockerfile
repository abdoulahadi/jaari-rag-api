# Utiliser une image Python officielle légère
FROM python:3.11-slim

# Définir les variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=10000 \
    TRANSFORMERS_CACHE=/app/cache/transformers \
    HF_HOME=/app/cache/huggingface

# Créer un utilisateur non-root pour la sécurité
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Créer le répertoire de travail
WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY . .

# Créer les répertoires nécessaires
RUN mkdir -p /app/data/logs \
             /app/data/audio_output \
             /app/data/vectorstore \
             /app/cache/embeddings \
             /app/cache/transformers \
             /app/cache/huggingface \
             /app/uploads && \
    chown -R appuser:appuser /app

# Changer vers l'utilisateur non-root
USER appuser

# Exposer le port
EXPOSE $PORT

# Vérification de santé
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Commande par défaut - utilise la variable PORT de Render
CMD python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
