#!/bin/bash

echo "🚀 Starting Jaari RAG API on Render..."
echo "🔧 Provider LLM configuré: ${LLM_PROVIDER:-groq}"
echo "🌐 Port: ${PORT:-8000}"

# Verify required environment variables
if [ "$LLM_PROVIDER" = "groq" ] && [ -z "$GROQ_API_KEY" ]; then
    echo "❌ GROQ_API_KEY is required for Groq provider"
    exit 1
fi

# Verify Google Cloud credentials
if [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "⚠️ Google Cloud credentials file not found: $GOOGLE_APPLICATION_CREDENTIALS"
    echo "💡 Some translation features may be limited"
fi

echo "✅ Starting server on port ${PORT:-8000}"
exec python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
