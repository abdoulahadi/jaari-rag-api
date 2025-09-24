"""
Service de Synthèse Vocale Wolof avec ADIA TTS
============================================
Service optimisé pour la génération audio en wolof avec:
- Traitement par chunks pour améliorer la clarté
- Génération parallèle pour optimiser les performances
- Support des formats web (MP3, WAV)
- Configuration avancée pour la qualité audio
"""

import torch
import asyncio
import logging
import hashlib
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

# Audio processing
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize

# TTS model
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Avoid tokenizers parallelism warning when forking worker processes
# See: https://github.com/huggingface/tokenizers/issues/192
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

class AdiaWolofTTSService:
    """
    Service de synthèse vocale Wolof avec ADIA TTS
    Optimisé pour les textes longs avec traitement par chunks
    """
    
    def __init__(self, 
                 model_name: str = "CONCREE/Adia_TTS",
                 audio_output_dir: str = "data/audio_output",
                 max_chunk_length: int = 150,
                 overlap_words: int = 3):
        """
        Initialiser le service TTS

        Args:
            model_name: Nom du modèle Hugging Face
            audio_output_dir: Dossier de sortie pour les fichiers audio
            max_chunk_length: Longueur maximale d'un chunk (en caractères)
            overlap_words: Nombre de mots de chevauchement entre chunks
        """
        self.model_name = model_name
        self.audio_output_dir = Path(audio_output_dir)
        self.max_chunk_length = max_chunk_length
        self.overlap_words = overlap_words

        # Configuration GPU/CPU
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"TTS Service initialized on device: {self.device}")

        # Modèle et tokenizer (chargement lazy)
        self.model = None
        self.tokenizer = None
        self._model_lock = threading.Lock()
        self._is_loading = False

        # Configuration de génération optimisée
        self.generation_config = {
            "temperature": 0.8,
            "min_new_tokens": 50,      # Minimum plus élevé pour assurer la lecture complète
            "max_new_tokens": 1000,    # Augmenté pour les textes plus longs
            "do_sample": True,
            "top_k": 50,
            "repetition_penalty": 1.2,
            "pad_token_id": None,      # Sera défini après le chargement du tokenizer
            "eos_token_id": None,      # Sera défini après le chargement du tokenizer
            "early_stopping": False    # Désactivé par défaut pour lecture complète
        }
        
        # Configuration audio
        self.audio_config = {
            "sample_rate": 24000,  # Sera défini après le chargement du modèle
            "output_format": "wav",
            "normalize_audio": True,
            "fade_duration": 0.1  # Fondu entre chunks (en secondes)
        }
        
        # Description vocale par défaut
        self.default_voice_description = "A clear and educational voice, with a flow adapted to learning"
        
        # Créer le dossier de sortie
        self.audio_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache pour éviter la régénération
        self._audio_cache = {}
        self._cache_lock = threading.Lock()
        
        # Eviter de lancer plusieurs générations simultanées pour le même texte
        # clé -> asyncio.Future
        self._in_progress: Dict[str, Any] = {}
        self._in_progress_lock = threading.Lock()
    
    async def load_model(self) -> bool:
        """Charger le modèle ADIA TTS de manière asynchrone"""
        if self.model is not None and self.tokenizer is not None:
            return True
        
        with self._model_lock:
            if self._is_loading:
                # Attendre que le chargement en cours se termine
                while self._is_loading:
                    await asyncio.sleep(0.1)
                return self.model is not None
            
            self._is_loading = True
        
        try:
            logger.info(f"🔄 Loading ADIA TTS model: {self.model_name}")
            
            # Charger le modèle et le tokenizer
            loop = asyncio.get_event_loop()
            
            # Chargement asynchrone du modèle
            self.model = await loop.run_in_executor(
                None,
                lambda: ParlerTTSForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            )
            
            # Chargement asynchrone du tokenizer
            self.tokenizer = await loop.run_in_executor(
                None,
                lambda: AutoTokenizer.from_pretrained(self.model_name)
            )
            
            # Mettre à jour la configuration
            self.audio_config["sample_rate"] = self.model.config.sampling_rate
            # Configurer les tokens spéciaux
            self.generation_config["pad_token_id"] = self.tokenizer.pad_token_id
            if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                self.generation_config["eos_token_id"] = self.tokenizer.eos_token_id
            
            logger.info(f"✅ ADIA TTS model loaded successfully")
            logger.info(f"   Device: {self.device}")
            logger.info(f"   Sample rate: {self.audio_config['sample_rate']} Hz")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load ADIA TTS model: {str(e)}")
            return False
        finally:
            self._is_loading = False
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """
        Diviser le texte en chunks optimaux pour la synthèse vocale
        
        Args:
            text: Texte à diviser
            
        Returns:
            Liste de chunks de texte
        """
        if len(text) <= self.max_chunk_length:
            return [text.strip()]
        
        chunks = []
        
        # Diviser par phrases d'abord
        sentences = re.split(r'[.!?]+', text)
        
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Si ajouter cette phrase dépasserait la limite
            if len(current_chunk) + len(sentence) + 1 > self.max_chunk_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # La phrase est trop longue, la diviser par mots
                    words = sentence.split()
                    word_chunk = ""
                    
                    for word in words:
                        if len(word_chunk) + len(word) + 1 > self.max_chunk_length:
                            if word_chunk:
                                chunks.append(word_chunk.strip())
                                word_chunk = word
                            else:
                                # Mot unique très long, le garder tel quel
                                chunks.append(word)
                        else:
                            word_chunk += " " + word if word_chunk else word
                    
                    if word_chunk:
                        current_chunk = word_chunk
            else:
                current_chunk += ". " + sentence if current_chunk else sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        logger.info(f"📝 Text split into {len(chunks)} chunks")
        return chunks
    
    def _validate_audio_duration(self, audio_data: np.ndarray, text: str, chunk_index: int) -> bool:
        """
        Valider que la durée de l'audio est cohérente avec le texte
        
        Args:
            audio_data: Données audio générées
            text: Texte original
            chunk_index: Index du chunk
            
        Returns:
            True si la durée semble correcte
        """
        if len(audio_data) == 0:
            return False
        
        duration_seconds = len(audio_data) / self.audio_config["sample_rate"]
        char_count = len(text)
        
        # Estimation: ~10-15 caractères par seconde pour une lecture normale
        expected_min_duration = char_count / 15  # Lecture rapide
        expected_max_duration = char_count / 8   # Lecture lente
        
        is_valid = expected_min_duration <= duration_seconds <= expected_max_duration * 2
        
        if not is_valid:
            logger.warning(f"⚠️ Chunk {chunk_index}: Duration {duration_seconds:.2f}s seems incorrect for {char_count} chars")
            logger.warning(f"   Expected: {expected_min_duration:.2f}s - {expected_max_duration:.2f}s")
        else:
            logger.debug(f"✅ Chunk {chunk_index}: Duration {duration_seconds:.2f}s OK for {char_count} chars")
        
        return is_valid

    def _generate_chunk_audio(self, chunk_text: str, chunk_index: int, voice_description: str, context: str = "") -> Tuple[int, np.ndarray]:
        """
        Générer l'audio pour un chunk spécifique avec une durée adaptée au texte
        Le paramètre `context` permet de fournir les derniers mots du chunk précédent
        comme contexte (sans les inclure dans le prompt principal) afin d'éviter
        que des mots soient coupés en fin de chunk.
        """
        start_ts = time.time()
        try:
            logger.debug(f"🎤 Generating audio for chunk {chunk_index}: '{chunk_text[:80]}...' (context='{context[:40]}')")

            # Estimer le nombre de tokens audio nécessaires basé sur la longueur du texte
            # Approximation plus généreuse: ~5-7 tokens audio par caractère de texte
            char_count = len(chunk_text)
            estimated_tokens = min(max(char_count * 6, 100), 1000)  # Plus généreux
            min_tokens = min(char_count * 3, estimated_tokens // 2)  # Minimum plus élevé

            # Préparer l'input de contexte (voix + derniers mots du chunk précédent)
            if context:
                input_text = f"{voice_description} {context}"
            else:
                input_text = voice_description

            # Tokenization
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            prompt_ids = self.tokenizer(chunk_text, return_tensors="pt").input_ids.to(self.device)

            # Configuration de génération adaptée pour ce chunk
            generation_config = self.generation_config.copy()
            generation_config["max_new_tokens"] = estimated_tokens
            generation_config["min_new_tokens"] = max(min_tokens, 50)  # Minimum garanti
            generation_config["early_stopping"] = False  # Désactiver l'arrêt précoce

            logger.debug(f"📊 Chunk {chunk_index}: {char_count} chars → {generation_config['min_new_tokens']}-{generation_config['max_new_tokens']} tokens")

            # Génération audio
            with torch.no_grad():
                audio = self.model.generate(
                    input_ids=input_ids,
                    prompt_input_ids=prompt_ids,
                    **generation_config
                )

            # Convertir en numpy array
            audio_np = audio.cpu().numpy().squeeze()

            # Valider que la durée semble correcte
            self._validate_audio_duration(audio_np, chunk_text, chunk_index)

            elapsed = time.time() - start_ts
            logger.info(f"✅ Chunk {chunk_index} generated: {len(audio_np)} samples ({len(audio_np)/self.audio_config['sample_rate']:.2f}s) in {elapsed:.2f}s")
            return chunk_index, audio_np

        except Exception as e:
            logger.error(f"❌ Failed to generate audio for chunk {chunk_index}: {str(e)}")
            raise
    
    async def _generate_chunks_parallel(self, chunks: List[str], voice_description: str, max_workers: int = 3) -> List[np.ndarray]:
        """
        Générer l'audio pour tous les chunks en parallèle
        
        Args:
            chunks: Liste des chunks de texte
            voice_description: Description de la voix
            max_workers: Nombre maximum de workers parallèles
            
        Returns:
            Liste des arrays audio dans l'ordre
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info(f"🔄 Generating audio for {len(chunks)} chunks in parallel (max_workers={max_workers})")
        
        # Limiter le nombre de workers selon le dispositif et les ressources
        if self.device.startswith("cuda"):
            max_workers = min(max_workers, 2)  # Éviter l'overload GPU
        
        loop = asyncio.get_event_loop()
        audio_results = {}
        total = len(chunks)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Soumettre toutes les tâches
            future_to_index = {}
            for i, chunk in enumerate(chunks):
                # Calculer le contexte (derniers mots du chunk précédent)
                context = ""
                if i > 0 and self.overlap_words > 0:
                    prev_words = chunks[i-1].split()
                    if prev_words:
                        context = " ".join(prev_words[-self.overlap_words:])
                fut = loop.run_in_executor(
                    executor,
                    self._generate_chunk_audio,
                    chunk,
                    i,
                    voice_description,
                    context
                )
                future_to_index[fut] = i

            # Collecter les résultats
            completed = 0
            for future in asyncio.as_completed(list(future_to_index.keys())):
                try:
                    chunk_index, audio_data = await future
                    audio_results[chunk_index] = audio_data
                    completed += 1
                    logger.info(f"🔸 Progress: {completed}/{total} chunks completed ({completed/total*100:.1f}%)")
                except Exception as e:
                    chunk_index = future_to_index.get(future, None)
                    logger.error(f"❌ Chunk {chunk_index} failed: {str(e)}")
                    raise
        
        # Réordonner les résultats
        ordered_audio = [audio_results[i] for i in range(len(chunks))]
        logger.info(f"✅ All chunks generated successfully")
        
        return ordered_audio
    
    def _trim_silence(self, audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """
        Supprimer les silences en début et fin d'audio
        
        Args:
            audio: Array audio
            threshold: Seuil de silence (amplitude relative)
            
        Returns:
            Audio sans silences aux extrémités
        """
        if len(audio) == 0:
            return audio
            
        # Calculer l'amplitude absolue
        abs_audio = np.abs(audio)
        
        # Trouver le premier échantillon non-silence
        start_idx = 0
        for i, sample in enumerate(abs_audio):
            if sample > threshold:
                start_idx = i
                break
        
        # Trouver le dernier échantillon non-silence
        end_idx = len(audio)
        for i in range(len(abs_audio) - 1, -1, -1):
            if abs_audio[i] > threshold:
                end_idx = i + 1
                break
        
        return audio[start_idx:end_idx]

    def _concatenate_audio_chunks(self, audio_chunks: List[np.ndarray]) -> np.ndarray:
        """
        Concaténer les chunks audio en supprimant les silences et sans pause
        
        Args:
            audio_chunks: Liste des arrays audio
            
        Returns:
            Array audio concaténé sans bruits
        """
        if not audio_chunks:
            return np.array([])
        
        if len(audio_chunks) == 1:
            return self._trim_silence(audio_chunks[0])
        
        logger.info(f"🔗 Concatenating {len(audio_chunks)} audio chunks")
        
        # Nettoyer le premier chunk
        result = self._trim_silence(audio_chunks[0])
        
        for i, chunk in enumerate(audio_chunks[1:], 1):
            # Nettoyer chaque chunk individuellement
            clean_chunk = self._trim_silence(chunk)
            
            if len(clean_chunk) == 0:
                logger.warning(f"⚠️ Chunk {i} is empty after trimming")
                continue
            
            # Concaténation directe sans fondu pour éviter tout bruit
            result = np.concatenate([result, clean_chunk])
            
            logger.debug(f"✅ Chunk {i} concatenated (trimmed)")
        
        logger.info(f"✅ Audio concatenation completed: {len(result)} samples")
        return result
    
    def _save_audio_file(self, audio_data: np.ndarray, output_path: Path, format: str = "wav") -> bool:
        """
        Sauvegarder le fichier audio avec normalisation
        
        Args:
            audio_data: Données audio
            output_path: Chemin de sortie
            format: Format de sortie (wav, mp3)
            
        Returns:
            True si succès
        """
        try:
            # Normaliser l'audio si demandé
            if self.audio_config["normalize_audio"]:
                # Éviter la saturation
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:
                    audio_data = audio_data / max_val * 0.95
            
            # Sauvegarder en WAV d'abord
            wav_path = output_path.with_suffix('.wav')
            sf.write(
                wav_path, 
                audio_data, 
                self.audio_config["sample_rate"],
                subtype='PCM_16'
            )
            
            logger.info(f"💾 Audio saved: {wav_path}")
            
            # Convertir en MP3 si demandé
            if format.lower() == "mp3":
                mp3_path = output_path.with_suffix('.mp3')
                audio_segment = AudioSegment.from_wav(str(wav_path))
                audio_segment = normalize(audio_segment)
                audio_segment.export(str(mp3_path), format="mp3", bitrate="128k")
                
                logger.info(f"🎵 MP3 version created: {mp3_path}")
                
                # Supprimer le WAV temporaire si on ne veut que le MP3
                if output_path.suffix.lower() == '.mp3':
                    wav_path.unlink()
                    return True
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save audio file: {str(e)}")
            return False
    
    def _get_cache_key(self, text: str, voice_description: str) -> str:
        """Générer une clé de cache pour le texte et la voix"""
        content = f"{text}_{voice_description}_{self.max_chunk_length}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def generate_wolof_audio(self, 
                                 wolof_text: str, 
                                 output_filename: Optional[str] = None,
                                 voice_description: Optional[str] = None,
                                 format: str = "mp3",
                                 use_cache: bool = True) -> Dict[str, Any]:
        """
        Générer l'audio pour un texte wolof
        
        Args:
            wolof_text: Texte en wolof à synthétiser
            output_filename: Nom du fichier de sortie (optionnel)
            voice_description: Description de la voix (optionnel)
            format: Format de sortie (wav, mp3)
            use_cache: Utiliser le cache si disponible
            
        Returns:
            Dictionnaire avec les informations du fichier audio généré
        """
        if not wolof_text or not wolof_text.strip():
            raise ValueError("Le texte wolof ne peut pas être vide")
        
        # Charger le modèle si nécessaire
        if not await self.load_model():
            raise RuntimeError("Impossible de charger le modèle ADIA TTS")
        
        voice_description = voice_description or self.default_voice_description
        wolof_text = wolof_text.strip()
        
        # Vérifier le cache
        cache_key = self._get_cache_key(wolof_text, voice_description)
        
        # Déduplication: si une génération pour la même clé est en cours, attendre et réutiliser
        loop = asyncio.get_event_loop()
        with self._in_progress_lock:
            inprog = self._in_progress.get(cache_key)
            if inprog is not None:
                logger.info(f"⏳ Waiting for in-progress generation for key {cache_key}")
        if inprog is not None:
            # attendre que l'autre coroutine termine et retourner le résultat cache
            try:
                await inprog
            except Exception:
                # l'opération précédente a échoué, continuer et lancer une nouvelle génération
                logger.warning("⚠️ Previous in-progress generation failed; retrying")
            with self._cache_lock:
                if cache_key in self._audio_cache:
                    logger.info(f"📱 Using cached audio after waiting: {self._audio_cache[cache_key]['file_path']}")
                    return self._audio_cache[cache_key]
        
        # Créer un Future pour signaler aux requêtes concurrentes que la génération est en cours
        generation_future = loop.create_future()
        with self._in_progress_lock:
            self._in_progress[cache_key] = generation_future

        start_time = time.time()
        
        try:
            # Générer le nom de fichier si non fourni
            if not output_filename:
                timestamp = int(time.time())
                safe_text = re.sub(r'[^a-zA-Z0-9_-]', '', wolof_text[:30])
                output_filename = f"wolof_audio_{safe_text}_{timestamp}.{format}"
            
            output_path = self.audio_output_dir / output_filename
            
            logger.info(f"🎤 Generating Wolof audio: '{wolof_text[:100]}...'")
            
            # 1. Diviser le texte en chunks
            chunks = self._split_text_into_chunks(wolof_text)
            
            # 2. Générer l'audio pour chaque chunk en parallèle
            audio_chunks = await self._generate_chunks_parallel(chunks, voice_description)
            
            # 3. Concaténer les chunks
            final_audio = self._concatenate_audio_chunks(audio_chunks)
            
            # 4. Sauvegarder le fichier
            if not self._save_audio_file(final_audio, output_path, format):
                raise RuntimeError("Échec de la sauvegarde du fichier audio")
            
            generation_time = time.time() - start_time
            duration_seconds = len(final_audio) / self.audio_config["sample_rate"]
            
            result = {
                "success": True,
                "file_path": str(output_path),
                "filename": output_filename,
                "format": format,
                "duration_seconds": duration_seconds,
                "generation_time": generation_time,
                "chunks_count": len(chunks),
                "sample_rate": self.audio_config["sample_rate"],
                "file_size_bytes": output_path.stat().st_size if output_path.exists() else 0,
                "text_length": len(wolof_text),
                "voice_description": voice_description
            }
            
            # Mettre en cache
            if use_cache:
                with self._cache_lock:
                    self._audio_cache[cache_key] = result

            # Résoudre la future pour les attendus concurrents
            if not generation_future.done():
                generation_future.set_result(result)

            logger.info(f"✅ Wolof audio generated successfully:")
            logger.info(f"   File: {output_path}")
            logger.info(f"   Duration: {duration_seconds:.2f}s")
            logger.info(f"   Chunks: {len(chunks)}")
            logger.info(f"   Generation time: {generation_time:.2f}s")
            
            return result
            
        except Exception as e:
            error_msg = f"Échec de la génération audio: {str(e)}"
            logger.error(f"❌ {error_msg}")
            if not generation_future.done():
                generation_future.set_exception(e)
            
            return {
                "success": False,
                "error": error_msg,
                "generation_time": time.time() - start_time,
                "text_length": len(wolof_text)
            }
        finally:
            # Nettoyage de l'état in_progress
            with self._in_progress_lock:
                if cache_key in self._in_progress:
                    try:
                        del self._in_progress[cache_key]
                    except KeyError:
                        pass
    
    def clear_cache(self):
        """Vider le cache audio"""
        with self._cache_lock:
            self._audio_cache.clear()
        logger.info("🧹 Audio cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Obtenir des informations sur le cache"""
        with self._cache_lock:
            return {
                "cached_items": len(self._audio_cache),
                "cache_keys": list(self._audio_cache.keys())
            }
    
    def cleanup_old_files(self, max_age_hours: int = 24):
        """Nettoyer les anciens fichiers audio"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        cleaned_count = 0
        
        for file_path in self.audio_output_dir.glob("*"):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete old file {file_path}: {e}")
        
        logger.info(f"🧹 Cleaned up {cleaned_count} old audio files")
        return cleaned_count

# Instance globale du service TTS
tts_service = AdiaWolofTTSService()
