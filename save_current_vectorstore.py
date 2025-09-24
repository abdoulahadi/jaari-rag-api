#!/usr/bin/env python3
"""
Script pour sauvegarder le vectorstore intelligent actuel
"""
import os
import sys
import logging
from pathlib import Path

# Ajouter le répertoire racine au Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.document_loader import DocumentLoader, DocumentSplitter
from app.core.vectorstore import VectorStoreManager
from app.config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
settings = get_settings()

def save_intelligent_vectorstore():
    """Sauvegarde le vectorstore avec chunking intelligent"""
    logger.info("🔄 Création et sauvegarde du vectorstore intelligent...")
    
    try:
        # Initialize components
        document_loader = DocumentLoader()
        document_splitter = DocumentSplitter()
        vectorstore_manager = VectorStoreManager()
        
        # Load documents from corpus
        corpus_path = Path("data/corpus")
        if not corpus_path.exists():
            logger.error(f"❌ Corpus path not found: {corpus_path}")
            return False
            
        logger.info(f"📂 Loading documents from: {corpus_path}")
        documents = document_loader.load_from_directory(str(corpus_path))
        
        if not documents:
            logger.error("❌ No documents found in corpus")
            return False
            
        logger.info(f"📄 Loaded {len(documents)} documents")
        
        # Split documents with intelligent chunking
        logger.info("🧠 Applying intelligent chunking...")
        chunks = document_splitter.split_documents(documents)
        logger.info(f"✂️ Created {len(chunks)} intelligent chunks")
        
        # Create and save vectorstore
        logger.info("🔄 Creating vectorstore with intelligent chunks...")
        vectorstore = vectorstore_manager.create_vectorstore(chunks)
        
        if vectorstore:
            # Ensure vectorstore directory exists
            vectorstore_path = Path("data/vectorstore")
            vectorstore_path.mkdir(parents=True, exist_ok=True)
            
            # Save vectorstore
            logger.info("💾 Saving vectorstore...")
            vectorstore_manager.save_vectorstore()
            
            # Verify save
            info = vectorstore_manager.get_vectorstore_info()
            logger.info(f"✅ Vectorstore saved successfully!")
            logger.info(f"   📊 Documents: {info.get('doc_count', 'Unknown')}")
            logger.info(f"   🏷️  Model: {info.get('embedding_model', 'Unknown')}")
            logger.info(f"   📍 Path: {vectorstore_path}")
            
            return True
        else:
            logger.error("❌ Failed to create vectorstore")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error saving vectorstore: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 SAUVEGARDE DU VECTORSTORE INTELLIGENT")
    print("=" * 50)
    
    success = save_intelligent_vectorstore()
    
    if success:
        print("\n✅ Vectorstore intelligent sauvegardé avec succès!")
        print("   Le système pourra maintenant le charger automatiquement au démarrage.")
    else:
        print("\n❌ Échec de la sauvegarde du vectorstore")
        sys.exit(1)
