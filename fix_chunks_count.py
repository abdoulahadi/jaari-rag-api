#!/usr/bin/env python3
"""
Script pour corriger les documents avec chunks_count NULL
"""
import sys
import os

# Ajouter le chemin de l'application au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from app.config.settings import settings
import logging

logger = logging.getLogger(__name__)

def fix_chunks_count():
    """Corriger les documents avec chunks_count NULL"""
    try:
        # Créer la connexion à la base de données
        engine = create_engine(settings.DATABASE_URL)
        
        with engine.connect() as conn:
            # Vérifier combien de documents ont chunks_count NULL
            result = conn.execute(text("SELECT COUNT(*) FROM documents WHERE chunks_count IS NULL"))
            null_count = result.scalar()
            
            print(f"Trouvé {null_count} documents avec chunks_count NULL")
            
            if null_count > 0:
                # Mettre à jour les documents avec chunks_count NULL vers 0
                update_result = conn.execute(text(
                    "UPDATE documents SET chunks_count = 0 WHERE chunks_count IS NULL"
                ))
                
                conn.commit()
                
                print(f"Mis à jour {update_result.rowcount} documents")
            else:
                print("Aucun document à corriger")
                
        print("✅ Migration terminée avec succès")
        
    except Exception as e:
        print(f"❌ Erreur lors de la migration: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    print("🔧 Correction des chunks_count NULL...")
    success = fix_chunks_count()
    sys.exit(0 if success else 1)
