"""
Optimisations Avancées pour le RAG Engine Jaari
===============================================
Techniques d'optimisation supplémentaires pour des performances maximales.
"""

import asyncio
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import gc
import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from contextlib import asynccontextmanager
import logging

# Optional imports with fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """Configuration des performances"""
    max_workers: int = multiprocessing.cpu_count()
    max_memory_gb: float = 8.0
    enable_gc_optimization: bool = True
    enable_memory_monitoring: bool = True
    batch_processing_size: int = 1000
    prefetch_enabled: bool = True


class MemoryOptimizer:
    """Optimiseur de mémoire pour le RAG Engine"""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.memory_usage = []
        self.gc_threshold = gc.get_threshold()

    def get_memory_usage(self) -> Dict[str, Any]:
        """Surveillance de l'utilisation mémoire"""
        if not PSUTIL_AVAILABLE:
            return {
                "rss_mb": 0.0,
                "vms_mb": 0.0,
                "percent": 0.0,
                "available_gb": 8.0  # Valeur par défaut
            }

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "available_gb": psutil.virtual_memory().available / 1024 / 1024 / 1024
        }

    def optimize_memory(self):
        """Optimisation de la mémoire"""
        if self.config.enable_gc_optimization:
            # Collecte des objets non référencés
            collected = gc.collect()

            # Ajustement du seuil GC si nécessaire
            if len(self.memory_usage) > 10:
                avg_memory = sum(m["rss_mb"] for m in self.memory_usage[-10:]) / 10
                if avg_memory > self.config.max_memory_gb * 1024 * 0.8:  # 80% du max
                    gc.set_threshold(self.gc_threshold[0] * 2, self.gc_threshold[1], self.gc_threshold[2])

            logger.info(f"🧹 GC: {collected} objets collectés")

    def monitor_memory(self):
        """Monitoring continu de la mémoire"""
        if self.config.enable_memory_monitoring:
            memory_info = self.get_memory_usage()
            self.memory_usage.append(memory_info)

            # Garder seulement les 100 dernières mesures
            if len(self.memory_usage) > 100:
                self.memory_usage = self.memory_usage[-100:]

            # Alerte si utilisation élevée
            if memory_info["percent"] > 85:
                logger.warning(f"⚠️ Utilisation mémoire élevée: {memory_info['percent']:.1f}%")


class ParallelProcessor:
    """Processeur parallèle pour les tâches intensives"""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.thread_executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=max(1, config.max_workers // 2))
        self.memory_optimizer = MemoryOptimizer(config)

    async def process_batch_async(self, items: List[Any], processor_func, batch_size: Optional[int] = None) -> List[Any]:
        """Traitement par batch asynchrone"""
        batch_size = batch_size or self.config.batch_processing_size
        results = []

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]

            # Traitement parallèle du batch
            batch_tasks = [
                asyncio.get_event_loop().run_in_executor(
                    self.thread_executor,
                    processor_func,
                    item
                ) for item in batch
            ]

            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)

            # Optimisation mémoire après chaque batch
            self.memory_optimizer.optimize_memory()

        return results

    def process_embeddings_batch(self, texts: List[str], embedding_func) -> List[List[float]]:
        """Traitement optimisé des embeddings par batch"""
        # Diviser en batches pour éviter la surcharge mémoire
        batches = [
            texts[i:i + self.config.batch_processing_size]
            for i in range(0, len(texts), self.config.batch_processing_size)
        ]

        all_embeddings = []
        for batch in batches:
            # Traitement parallèle des embeddings
            futures = [
                self.thread_executor.submit(embedding_func, text)
                for text in batch
            ]

            batch_embeddings = []
            for future in futures:
                try:
                    embedding = future.result(timeout=30)  # Timeout de 30s
                    batch_embeddings.append(embedding)
                except Exception as e:
                    logger.warning(f"Erreur embedding: {e}")
                    batch_embeddings.append([])  # Embedding vide en cas d'erreur

            all_embeddings.extend(batch_embeddings)

            # Monitoring mémoire
            self.memory_optimizer.monitor_memory()

        return all_embeddings

    def parallel_search(self, queries: List[str], search_func) -> List[Dict[str, Any]]:
        """Recherche parallèle pour multiple queries"""
        futures = [
            self.thread_executor.submit(search_func, query)
            for query in queries
        ]

        results = []
        for future in futures:
            try:
                result = future.result(timeout=60)  # Timeout de 60s
                results.append(result)
            except Exception as e:
                logger.error(f"Erreur recherche parallèle: {e}")
                results.append({"error": str(e)})

        return results


class QueryOptimizer:
    """Optimiseur de requêtes pour améliorer la pertinence"""

    def __init__(self):
        self.query_cache = {}
        self.similarity_threshold = 0.85

    def optimize_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimisation de la requête pour de meilleurs résultats"""
        # Nettoyage et normalisation
        optimized_query = self._normalize_query(query)

        # Expansion de requête si nécessaire
        expanded_queries = self._expand_query(optimized_query, context)

        # Détection de langue et adaptation
        language_info = self._detect_language(optimized_query)

        return {
            "original_query": query,
            "optimized_query": optimized_query,
            "expanded_queries": expanded_queries,
            "language": language_info,
            "optimization_applied": True
        }

    def _normalize_query(self, query: str) -> str:
        """Normalisation de la requête"""
        # Suppression des caractères spéciaux
        query = re.sub(r'[^\w\s]', ' ', query)

        # Normalisation des espaces
        query = ' '.join(query.split())

        # Correction des fautes courantes agricoles
        corrections = {
            'patate': 'pomme de terre',
            'legume': 'légume',
            'tomate': 'tomates',
            'semence': 'semences'
        }

        for wrong, correct in corrections.items():
            query = re.sub(r'\b' + wrong + r'\b', correct, query, flags=re.IGNORECASE)

        return query.lower()

    def _expand_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Expansion de la requête avec des termes similaires"""
        expansions = [query]

        # Expansion basée sur les synonymes agricoles
        agricultural_synonyms = {
            'pomme de terre': ['tubercule', 'patate douce', 'semence pomme de terre'],
            'tomate': ['solanacée', 'fruit rouge', 'légume fruit'],
            'irrigation': ['arrosage', 'eau agricole', 'système irrigation'],
            'fertilisant': ['engrais', 'nutrition plante', 'amendement']
        }

        for term, synonyms in agricultural_synonyms.items():
            if term in query:
                expansions.extend([query.replace(term, synonym) for synonym in synonyms])

        return list(set(expansions))  # Supprimer les doublons

    def _detect_language(self, query: str) -> Dict[str, Any]:
        """Détection de langue pour adaptation"""
        # Comptage des mots français vs anglais
        french_words = {'le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'dans', 'sur'}
        english_words = {'the', 'and', 'or', 'in', 'on', 'with', 'for', 'to', 'of', 'at'}

        words = set(query.lower().split())
        french_count = len(words.intersection(french_words))
        english_count = len(words.intersection(english_words))

        if french_count > english_count:
            return {"language": "fr", "confidence": french_count / max(1, len(words))}
        elif english_count > french_count:
            return {"language": "en", "confidence": english_count / max(1, len(words))}
        else:
            return {"language": "mixed", "confidence": 0.5}


class AsyncContextManager:
    """Gestionnaire de contexte asynchrone pour les ressources"""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.resources = {}
        self.locks = {}

    @asynccontextmanager
    async def managed_resource(self, resource_name: str, resource_factory):
        """Gestionnaire de ressources avec cache et nettoyage automatique"""
        if resource_name not in self.resources:
            self.resources[resource_name] = await resource_factory()
            self.locks[resource_name] = asyncio.Lock()

        async with self.locks[resource_name]:
            try:
                yield self.resources[resource_name]
            finally:
                # Nettoyage si nécessaire
                if hasattr(self.resources[resource_name], 'cleanup'):
                    await self.resources[resource_name].cleanup()

    async def cleanup_resources(self):
        """Nettoyage de toutes les ressources"""
        for resource_name, resource in self.resources.items():
            if hasattr(resource, 'close'):
                await resource.close()
            elif hasattr(resource, 'cleanup'):
                await resource.cleanup()

        self.resources.clear()
        self.locks.clear()


# Configuration globale des performances
performance_config = PerformanceConfig()

# Instances des optimiseurs
memory_optimizer = MemoryOptimizer(performance_config)
parallel_processor = ParallelProcessor(performance_config)
query_optimizer = QueryOptimizer()
async_manager = AsyncContextManager(performance_config)


def optimize_rag_pipeline(rag_engine):
    """
    Application des optimisations au RAG Engine
    """
    # Injection des optimiseurs
    rag_engine.memory_optimizer = memory_optimizer
    rag_engine.parallel_processor = parallel_processor
    rag_engine.query_optimizer = query_optimizer
    rag_engine.async_manager = async_manager

    # Optimisation des méthodes principales
    original_query = rag_engine.query

    async def optimized_query(question: str, **kwargs):
        """Version optimisée de la méthode query"""
        # Optimisation de la requête
        optimized = query_optimizer.optimize_query(question)

        # Utilisation du contexte asynchrone
        async with async_manager.managed_resource("query_context", lambda: asyncio.create_task(asyncio.sleep(0))):
            # Traitement parallèle si multiple queries
            if isinstance(question, list):
                results = await parallel_processor.process_batch_async(
                    question,
                    lambda q: original_query(q, **kwargs)
                )
                return results
            else:
                result = await original_query(optimized["optimized_query"], **kwargs)

                # Optimisation mémoire après traitement
                memory_optimizer.optimize_memory()

                return result

    # Remplacement de la méthode
    rag_engine.query = optimized_query

    logger.info("🚀 Optimisations avancées appliquées au RAG Engine")
    return rag_engine


# Fonction utilitaire pour monitoring continu
def start_performance_monitoring(interval: int = 60):
    """Démarrage du monitoring des performances en arrière-plan"""

    def monitor_loop():
        while True:
            try:
                memory_info = memory_optimizer.get_memory_usage()
                logger.info(f"📊 Memory: {memory_info['rss_mb']:.1f}MB "
                          f"({memory_info['percent']:.1f}%)")

                # Optimisation mémoire si nécessaire
                if memory_info['percent'] > 90:
                    memory_optimizer.optimize_memory()

                threading.Event().wait(interval)

            except Exception as e:
                logger.error(f"Erreur monitoring: {e}")
                break

    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()

    logger.info("📈 Monitoring des performances démarré")


if __name__ == "__main__":
    # Démarrage du monitoring
    start_performance_monitoring()

    print("🎯 Optimisations avancées chargées")
    print("📊 Monitoring démarré")
    print("🚀 Prêt pour des performances optimales!")
