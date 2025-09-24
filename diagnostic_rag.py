#!/usr/bin/env python3
"""
Script de diagnostic pour analyser le problème de récupération de sources
"""

import asyncio
import os
from pathlib import Path
from app.core.rag_engine import RAGEngine
from app.config.settings import settings

async def diagnostic_complet():
    print("🔍 DIAGNOSTIC COMPLET DU SYSTÈME RAG")
    print("=" * 50)
    
    # Initialiser le RAG engine
    rag = RAGEngine()
    
    print("\n1. INITIALISATION DU RAG ENGINE")
    success = await rag.initialize()
    print(f"   ✅ Initialisé: {success}")
    
    # Vérifier le statut
    print("\n2. STATUT DU SYSTÈME")
    status = rag.get_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Lister les fichiers du corpus
    print("\n3. FICHIERS DU CORPUS")
    corpus_dir = Path(settings.CORPUS_DIR)
    if corpus_dir.exists():
        files = list(corpus_dir.glob("*.pdf"))
        print(f"   📁 Dossier: {corpus_dir}")
        print(f"   📊 Fichiers PDF: {len(files)}")
        for file in files:
            print(f"      - {file.name}")
            if "POMME" in file.name.upper():
                print(f"        🥔 FICHIER POMME DE TERRE DÉTECTÉ!")
    
    # Test de recherche spécifique
    print("\n4. TESTS DE RECHERCHE")
    
    # Forcer le rechargement des documents
    print("   🔄 Rechargement forcé des documents...")
    try:
        rag.vectorstore_manager.delete_vectorstore()
        success = rag.load_documents_from_directory(settings.CORPUS_DIR, force_recreate=True)
        print(f"   ✅ Rechargement: {success}")
        
        # Réinitialiser la chaîne QA
        if success and rag.llm:
            rag.setup_rag_chain()
            print("   ✅ Chaîne QA reconfigurée")
    except Exception as e:
        print(f"   ❌ Erreur rechargement: {e}")
    
    # Test de recherche avec différents termes
    search_terms = [
        "pomme de terre",
        "pomme terre Sénégal",
        "potato cultivation",
        "tubercule",
        "TPS POMME",
        "culture tubercule",
        "plantain pomme"
    ]
    
    for term in search_terms:
        print(f"\n   🔍 Recherche: '{term}'")
        try:
            results = rag.search_documents(term, k=10)
            print(f"      📊 {len(results)} résultats trouvés")
            
            # Chercher spécifiquement les documents pomme de terre
            pomme_results = []
            for r in results:
                source = r['metadata'].get('source', '').upper()
                if 'POMME' in source or 'POTATO' in source:
                    pomme_results.append(r)
            
            if pomme_results:
                print(f"      🥔 {len(pomme_results)} résultats du fichier pomme de terre!")
                for r in pomme_results[:3]:
                    print(f"         Score: {r['similarity_score']:.4f}")
                    print(f"         Page: {r['metadata'].get('page', 'N/A')}")
                    print(f"         Contenu: {r['content'][:100]}...")
            else:
                print("      ❌ Aucun résultat du fichier pomme de terre")
                # Afficher les 3 premiers résultats pour debug
                for i, r in enumerate(results[:3]):
                    print(f"         {i+1}. {r['metadata'].get('source', 'N/A')} (Score: {r['similarity_score']:.4f})")
                    
        except Exception as e:
            print(f"      ❌ Erreur: {e}")
    
    # Test de requête complète
    print("\n5. TEST DE REQUÊTE COMPLÈTE")
    try:
        response = rag.query("Parle moi de la culture de pomme de terre au Sénégal", return_sources=True)
        print(f"   📝 Réponse générée: {len(response.get('answer', ''))} caractères")
        print(f"   📊 Sources: {response.get('num_sources', 0)}")
        
        if 'sources' in response:
            print("   📄 Sources trouvées:")
            for i, source in enumerate(response['sources'][:5]):
                source_name = source['metadata'].get('source', 'N/A')
                page = source['metadata'].get('page', 'N/A')
                print(f"      {i+1}. {source_name} - Page {page}")
                if 'POMME' in source_name.upper():
                    print(f"         🥔 POMME DE TERRE TROUVÉE!")
                    
    except Exception as e:
        print(f"   ❌ Erreur requête: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("✅ DIAGNOSTIC TERMINÉ")

if __name__ == "__main__":
    asyncio.run(diagnostic_complet())
