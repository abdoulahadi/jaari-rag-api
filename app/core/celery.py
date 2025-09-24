"""
Celery configuration for Jaari RAG API
"""
from celery import Celery
from app.config.settings import settings

# Create Celery instance
celery_app = Celery(
    "jaari-rag-api",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.tasks"]
)

# Configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    result_expires=3600,
)

# Auto-discover tasks
celery_app.autodiscover_tasks()