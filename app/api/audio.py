"""
API pour la génération et la diffusion d'audio Wolof
==================================================
Endpoints pour générer et servir les fichiers audio en wolof
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import os
from pathlib import Path

# Import conditionnel du service TTS
try:
    from app.services.tts_service import tts_service
    TTS_AVAILABLE = True
except ImportError as e:
    TTS_AVAILABLE = False
    tts_service = None
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ TTS service not available: {e}")

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audio", tags=["audio"])

class AudioGenerationRequest(BaseModel):
    text: str
    voice_description: Optional[str] = "A clear and educational voice, with a flow adapted to learning"
    format: Optional[str] = "mp3"
    use_cache: Optional[bool] = True

class AudioGenerationResponse(BaseModel):
    success: bool
    file_path: Optional[str] = None
    filename: Optional[str] = None
    format: Optional[str] = None
    duration_seconds: Optional[float] = None
    generation_time: Optional[float] = None
    chunks_count: Optional[int] = None
    error: Optional[str] = None

@router.get("/test")
async def test_audio_endpoint():
    """
    Endpoint de test pour vérifier que l'API audio fonctionne
    """
    try:
        return {
            "status": "ok",
            "message": "API Audio accessible",
            "endpoints": [
                "/audio/test",
                "/audio/generate-wolof", 
                "/audio/files/{filename}",
                "/audio/status"
            ]
        }
    except Exception as e:
        logger.error(f"❌ Error in test endpoint: {str(e)}")
        return {"status": "error", "error": str(e)}

@router.post("/generate-wolof", response_model=AudioGenerationResponse)
async def generate_wolof_audio(request: AudioGenerationRequest):
    """
    Générer un fichier audio à partir d'un texte wolof
    """
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide")
        
        logger.info(f"🎤 Audio generation request: '{request.text[:50]}...'")
        
        # Vérification si le service TTS est disponible
        if not TTS_AVAILABLE or tts_service is None:
            logger.error("❌ TTS service not available")
            return AudioGenerationResponse(
                success=False,
                error="Service TTS non disponible - dépendances manquantes. Installez: pip install -r requirements_tts.txt"
            )
        
        # Générer l'audio avec le service TTS
        result = await tts_service.generate_wolof_audio(
            wolof_text=request.text.strip(),
            voice_description=request.voice_description,
            format=request.format,
            use_cache=request.use_cache
        )
        
        if result.get("success"):
            # Retourner le chemin relatif pour l'API
            filename = result.get("filename")
            return AudioGenerationResponse(
                success=True,
                file_path=f"/audio/files/{filename}",
                filename=filename,
                format=result.get("format"),
                duration_seconds=result.get("duration_seconds"),
                generation_time=result.get("generation_time"),
                chunks_count=result.get("chunks_count")
            )
        else:
            logger.error(f"❌ Audio generation failed: {result.get('error')}")
            return AudioGenerationResponse(
                success=False,
                error=result.get("error", "Erreur inconnue lors de la génération audio")
            )
            
    except Exception as e:
        logger.error(f"❌ Audio generation API error: {str(e)}")
        return AudioGenerationResponse(
            success=False,
            error=f"Erreur interne: {str(e)}"
        )

@router.get("/files/{filename}")
async def serve_audio_file(filename: str):
    """
    Servir un fichier audio généré
    """
    try:
        # Vérifier que le fichier existe
        file_path = tts_service.audio_output_dir / filename
        
        if not file_path.exists():
            logger.warning(f"📁 Audio file not found: {filename}")
            raise HTTPException(status_code=404, detail="Fichier audio non trouvé")
        
        # Déterminer le type MIME
        if filename.lower().endswith('.mp3'):
            media_type = "audio/mpeg"
        elif filename.lower().endswith('.wav'):
            media_type = "audio/wav"
        else:
            media_type = "application/octet-stream"
        
        logger.info(f"🎵 Serving audio file: {filename}")
        
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=filename,
            headers={
                "Cache-Control": "public, max-age=3600",  # Cache 1 heure
                "Accept-Ranges": "bytes"  # Support du streaming
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error serving audio file {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur lors de la lecture du fichier")

@router.get("/cache/info")
async def get_cache_info():
    """
    Obtenir des informations sur le cache audio
    """
    try:
        cache_info = tts_service.get_cache_info()
        
        # Ajouter des informations sur les fichiers
        audio_dir = tts_service.audio_output_dir
        files_info = []
        total_size = 0
        
        if audio_dir.exists():
            for file_path in audio_dir.glob("*"):
                if file_path.is_file():
                    stat = file_path.stat()
                    files_info.append({
                        "filename": file_path.name,
                        "size_bytes": stat.st_size,
                        "created_timestamp": stat.st_ctime
                    })
                    total_size += stat.st_size
        
        return {
            "cache": cache_info,
            "files": {
                "count": len(files_info),
                "total_size_bytes": total_size,
                "files": files_info[:10]  # Limiter à 10 pour éviter des réponses trop grandes
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting cache info: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des informations de cache")

@router.delete("/cache/clear")
async def clear_audio_cache(background_tasks: BackgroundTasks):
    """
    Vider le cache audio
    """
    try:
        def clear_cache_task():
            tts_service.clear_cache()
            logger.info("🧹 Audio cache cleared via API")
        
        background_tasks.add_task(clear_cache_task)
        
        return {"message": "Cache audio vidé avec succès"}
        
    except Exception as e:
        logger.error(f"❌ Error clearing cache: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur lors du vidage du cache")

@router.delete("/files/cleanup")
async def cleanup_old_files(background_tasks: BackgroundTasks, max_age_hours: int = 24):
    """
    Nettoyer les anciens fichiers audio
    """
    try:
        def cleanup_task():
            cleaned_count = tts_service.cleanup_old_files(max_age_hours)
            logger.info(f"🧹 Cleaned {cleaned_count} old audio files via API")
            return cleaned_count
        
        background_tasks.add_task(cleanup_task)
        
        return {"message": f"Nettoyage des fichiers de plus de {max_age_hours}h démarré"}
        
    except Exception as e:
        logger.error(f"❌ Error starting cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur lors du démarrage du nettoyage")

@router.get("/status")
async def get_tts_status():
    """
    Obtenir le statut du service TTS
    """
    try:
        if not TTS_AVAILABLE or tts_service is None:
            return {
                "status": "unavailable",
                "error": "TTS service not available - missing dependencies",
                "available_endpoints": ["/audio/test"],
                "installation_required": "pip install -r requirements_tts.txt"
            }
            
        # Vérifier si le modèle est chargé
        model_loaded = tts_service.model is not None and tts_service.tokenizer is not None
        
        # Informations sur le dispositif
        device_info = {
            "device": tts_service.device,
            "cuda_available": tts_service.device.startswith("cuda")
        }
        
        # Informations sur la configuration
        config_info = {
            "model_name": tts_service.model_name,
            "max_chunk_length": tts_service.max_chunk_length,
            "overlap_words": tts_service.overlap_words,
            "audio_format": tts_service.audio_config["output_format"],
            "sample_rate": tts_service.audio_config.get("sample_rate", "Not loaded"),
        }
        
        return {
            "status": "ready" if model_loaded else "not_loaded",
            "model_loaded": model_loaded,
            "device": device_info,
            "config": config_info,
            "cache": tts_service.get_cache_info()
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting TTS status: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

@router.post("/preload")
async def preload_tts_model():
    """
    Précharger le modèle TTS
    """
    try:
        logger.info("🔄 Preloading TTS model...")
        
        success = await tts_service.load_model()
        
        if success:
            return {"message": "Modèle TTS chargé avec succès", "status": "ready"}
        else:
            raise HTTPException(status_code=500, detail="Échec du chargement du modèle TTS")
            
    except Exception as e:
        logger.error(f"❌ Error preloading TTS model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du chargement: {str(e)}")
