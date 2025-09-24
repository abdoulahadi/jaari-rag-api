# Use Python slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
# Set default LLM provider to Groq (option 1 from start_server.sh)
ENV LLM_PROVIDER=groq

# Set work directory
WORKDIR /app

# Install system dependencies including OCR support
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        git \
        tesseract-ocr \
        tesseract-ocr-fra \
        tesseract-ocr-eng \
        poppler-utils \
        libmagic1 \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with enhanced retry and error handling
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --timeout 1000 --retries 10 --default-timeout=100 -r requirements.txt

# Copy application code
COPY . .

# Create directories for data persistence
RUN mkdir -p /app/data/logs /app/data/vectorstore /app/data/cache

# Set permissions
RUN chmod -R 755 /app

# Setup Google Cloud credentials path
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/translate-jaari-065fa764be8a.json

# Health check (use PORT env var if available, fallback to 8000)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Make startup script executable
RUN chmod +x /app/start_render.sh

# Run the application using Render-compatible startup script
CMD ["/app/start_render.sh"]
