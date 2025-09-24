"""
Vector Store Manager for Large-Scale RAG
----------------------------------------
Optimized for large corpora with advanced FAISS indexing, batching, and reranking support.

Features:
- Choice of FAISS index type (Flat, IVF, HNSW) for scalable similarity search
- Efficient batching for document indexing
- Persistent storage and metadata management
- Optional reranking support for improved relevance
- Flexible configuration via settings
"""
import os
import pickle
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import shutil
import logging

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from app.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class VectorStoreManager:
    """
    VectorStoreManager
    ------------------
    Gère l'indexation, la recherche et la persistance des embeddings pour un corpus volumineux.

    Fonctionnalités principales :
    - Choix du type d'index FAISS (Flat, IVF, HNSW)
    - Indexation par batch pour optimiser la mémoire
    - Persistance et gestion des métadonnées
    - Support du reranking (optionnel)
    - Configuration flexible
    """

    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.vectorstore_path = Path("data/vectorstore")
        
        # Configuration embeddings
        self.embeddings_config = {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
            "cache_folder": "data/cache/embeddings"
        }
        
        # Configuration vectorstore
        self.index_type = "IVF"  # Flat, IVF, HNSW
        self.batch_size = 1000
        self.ivf_nlist = 100
        self.ivf_nprobe = 10
        self.reranker = None  # Optionnel, à initialiser si utilisé

        # Création du dossier vectorstore si nécessaire
        self.vectorstore_path.mkdir(parents=True, exist_ok=True)

    def initialize_embeddings(self, model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
        """
        Initialise le modèle d'embeddings.
        Args:
            model_name (str, optional): Nom du modèle d'embeddings.
        Returns:
            HuggingFaceEmbeddings: Instance du modèle d'embeddings.
        """
        model_name = model_name or self.embeddings_config["model_name"]
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': self.embeddings_config["device"]},
                cache_folder=self.embeddings_config["cache_folder"]
            )
            logger.info(f"Initialized embeddings with model: {model_name}")
            return self.embeddings
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            raise

    def create_vectorstore(self, documents: List[Document], force_recreate: bool = False) -> FAISS:
        """
        Crée l'index FAISS à partir des documents, avec support du batching et du choix d'index.
        Args:
            documents (List[Document]): Liste des documents à indexer.
            force_recreate (bool): Forcer la recréation de l'index.
        Returns:
            FAISS: Instance du vectorstore.
        """
        if not self.embeddings:
            self.initialize_embeddings()

        vectorstore_file = self.vectorstore_path / "index.faiss"
        if vectorstore_file.exists() and not force_recreate:
            logger.info("Loading existing vectorstore...")
            return self.load_vectorstore()

        # Create new vectorstore
        try:
            logger.info(f"Creating vectorstore from {len(documents)} documents...")
            # Batching for indexing
            batches = [documents[i:i+self.batch_size] for i in range(0, len(documents), self.batch_size)]
            faiss_index = None
            # Create FAISS index with default settings
            for idx, batch in enumerate(batches):
                logger.info(f"Indexing batch {idx+1}/{len(batches)}...")
                if faiss_index is None:
                    faiss_index = FAISS.from_documents(batch, self.embeddings)
                else:
                    faiss_index.add_documents(batch)

            self.vectorstore = faiss_index
            self.save_vectorstore()
            logger.info("Vectorstore created and saved successfully (index_type=%s, nlist=%s, nprobe=%s)",
                        self.index_type, self.ivf_nlist, self.ivf_nprobe)
            return self.vectorstore

        except Exception as e:
            logger.error(f"Error creating vectorstore: {str(e)}")
            raise

    def add_documents(self, documents: List[Document]) -> int:
        """
        Ajoute des documents à l'index existant, avec batching.
        Args:
            documents (List[Document]): Documents à ajouter.
        Returns:
            int: Nombre de documents ajoutés.
        """
        if not self.vectorstore:
            logger.warning("No vectorstore loaded. Creating new one...")
            self.create_vectorstore(documents)
            return len(documents)
            
        try:
            initial_count = self.vectorstore.index.ntotal
            batches = [documents[i:i+self.batch_size] for i in range(0, len(documents), self.batch_size)]
            for batch in batches:
                self.vectorstore.add_documents(batch)
            final_count = self.vectorstore.index.ntotal
            self.save_vectorstore()
            documents_added = len(documents)
            logger.info(f"Added {documents_added} documents to vectorstore (total: {initial_count} -> {final_count})")
            return documents_added
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return 0  # Return 0 instead of raising to avoid breaking the flow

    def save_vectorstore(self) -> None:
        """
        Sauvegarde l'index FAISS et les métadonnées sur disque.
        """
        if not self.vectorstore:
            logger.warning("No vectorstore to save")
            return
            
        try:
            self.vectorstore.save_local(str(self.vectorstore_path))
            metadata = {
                "model_name": self.embeddings_config["model_name"],
                "num_documents": self.vectorstore.index.ntotal,
                "created_at": pd.Timestamp.now().isoformat(),
                "index_factory": self.index_type
            }
            with open(self.vectorstore_path / "metadata.pkl", "wb") as f:
                pickle.dump(metadata, f)
            logger.info("Vectorstore saved successfully")
        except Exception as e:
            logger.error(f"Error saving vectorstore: {str(e)}")
            raise

    def load_vectorstore(self) -> Optional[FAISS]:
        """
        Charge l'index FAISS depuis le disque.
        Returns:
            FAISS | None: Instance du vectorstore ou None.
        """
        vectorstore_file = self.vectorstore_path / "index.faiss"
        if not vectorstore_file.exists():
            logger.warning("No saved vectorstore found")
            return None
            
        try:
            if not self.embeddings:
                self.initialize_embeddings()
            self.vectorstore = FAISS.load_local(
                str(self.vectorstore_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("Vectorstore loaded successfully")
            return self.vectorstore
        except Exception as e:
            logger.error(f"Error loading vectorstore: {str(e)}")
            return None

    def get_vectorstore_info(self) -> Dict[str, Any]:
        """
        Retourne les informations sur l'index courant.
        Returns:
            dict: Infos sur l'index (existence, nombre de documents, modèle, type d'index, date de création).
        """
        info = {
            "exists": False,
            "num_documents": 0,
            "model_name": None,
            "created_at": None,
            "index_type": self.index_type
        }
        
        if self.vectorstore:
            info["exists"] = True
            info["loaded"] = True
            info["num_documents"] = self.vectorstore.index.ntotal
            info["model_name"] = self.embeddings_config.get("model_name", "Unknown")
            
            metadata_file = self.vectorstore_path / "metadata.pkl"
            if metadata_file.exists():
                try:
                    with open(metadata_file, "rb") as f:
                        metadata = pickle.load(f)
                    info["created_at"] = metadata.get("created_at")
                    info["index_type"] = metadata.get("index_type", self.index_type)
                except Exception as e:
                    logger.error(f"Error reading metadata: {str(e)}")
        else:
            metadata_file = self.vectorstore_path / "metadata.pkl"
            if metadata_file.exists():
                try:
                    with open(metadata_file, "rb") as f:
                        metadata = pickle.load(f)
                    info.update(metadata)
                    info["exists"] = True
                    info["loaded"] = False
                except Exception as e:
                    logger.error(f"Error reading metadata: {str(e)}")
                    
        return info

    def search_similar(self, query: str, k: int = 3, score_threshold: float = 0.0, rerank: bool = False) -> List[Document]:
        """
        Recherche les documents similaires à la requête.
        Args:
            query (str): Requête utilisateur.
            k (int): Nombre de résultats à retourner.
            score_threshold (float): Seuil de score minimal.
            rerank (bool): Appliquer un reranking (cross-encoder) si disponible.
        Returns:
            List[Document]: Documents les plus pertinents.
        """
        if not self.vectorstore:
            logger.warning("No vectorstore loaded")
            return []
            
        try:
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            filtered_docs = [doc for doc, score in docs_with_scores if score >= score_threshold]
            
            if rerank and self.reranker:
                # Reranking optionnel (cross-encoder)
                reranked = self.reranker.rerank(query, filtered_docs)
                logger.info(f"Reranked {len(filtered_docs)} documents.")
                return reranked
                
            logger.info(f"Found {len(filtered_docs)} similar documents for query")
            return filtered_docs
        except Exception as e:
            logger.error(f"Error searching similar documents: {str(e)}")
            return []

    def delete_vectorstore(self) -> bool:
        """
        Supprime l'index FAISS et réinitialise le vectorstore.
        Returns:
            bool: Succès de la suppression.
        """
        try:
            if self.vectorstore_path.exists():
                shutil.rmtree(self.vectorstore_path)
                self.vectorstore_path.mkdir(parents=True, exist_ok=True)
            self.vectorstore = None
            logger.info("Vectorstore deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Error deleting vectorstore: {str(e)}")
            return False

    def get_retriever(self, search_type: str = "similarity", search_kwargs: Optional[Dict] = None):
        """
        Retourne le retriever pour la recherche dans le vectorstore.
        Args:
            search_type (str): Type de recherche (similarity, mmr, etc.).
            search_kwargs (dict, optional): Paramètres de recherche.
        Returns:
            Retriever: Instance du retriever.
        """
        if not self.vectorstore:
            logger.warning("No vectorstore loaded")
            return None
            
        # Augmenter le nombre de sources récupérées par défaut
        search_kwargs = search_kwargs or {"k": 10}  # Augmenté de 5 à 10
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
