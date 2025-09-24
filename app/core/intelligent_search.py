"""
Module de recherche hybride intelligente pour améliorer la pertinence
"""
import re
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class IntelligentQueryProcessor:
    """
    Processeur de requêtes intelligent pour améliorer la recherche
    """
    
    def __init__(self):
        # Vocabulaire agricole spécialisé
        self.agricultural_keywords = {
            'pomme_de_terre': ['pomme de terre', 'pommes de terre', 'tubercule', 'patate', 'potato'],
            'haricot': ['haricot', 'haricots', 'légumineuse', 'bean'],
            'laitue': ['laitue', 'salade', 'lettuce'],
            'manioc': ['manioc', 'cassava', 'tapioca'],
            'pasteque': ['pastèque', 'pastéque', 'watermelon'],
            'oignon': ['oignon', 'onion'],
            'tomate': ['tomate', 'tomato'],
            'chou': ['chou', 'cabbage'],
        }
        
        # Termes techniques agricoles
        self.technical_terms = {
            'production': ['production', 'produire', 'cultiver', 'culture'],
            'semences': ['semence', 'semences', 'graine', 'graines', 'seed'],
            'plantation': ['plantation', 'planter', 'semer', 'plant'],
            'rendement': ['rendement', 'yield', 'productivité'],
            'technique': ['technique', 'méthode', 'procédé', 'itinéraire'],
            'sols': ['sol', 'sols', 'terre', 'terroir', 'soil'],
            'irrigation': ['irrigation', 'arrosage', 'eau', 'water'],
            'fertilisation': ['fertilisant', 'engrais', 'nutriment', 'fertilizer'],
        }
        
        # Contexte géographique
        self.geographic_terms = {
            'senegal': ['sénégal', 'senegal', 'sénégalais'],
            'niayes': ['niayes', 'niaye'],
            'afrique': ['afrique', 'africain', 'africa'],
        }
    
    def extract_key_concepts(self, question: str) -> Dict[str, List[str]]:
        """
        Extrait les concepts clés d'une question
        """
        question_lower = question.lower()
        concepts = {
            'crops': [],
            'techniques': [],
            'geography': [],
            'keywords': []
        }
        
        # Détecter les cultures
        for crop, synonyms in self.agricultural_keywords.items():
            for synonym in synonyms:
                if synonym.lower() in question_lower:
                    concepts['crops'].append(crop)
                    concepts['keywords'].append(synonym)
                    break
        
        # Détecter les termes techniques
        for tech, synonyms in self.technical_terms.items():
            for synonym in synonyms:
                if synonym.lower() in question_lower:
                    concepts['techniques'].append(tech)
                    concepts['keywords'].append(synonym)
                    break
        
        # Détecter la géographie
        for geo, synonyms in self.geographic_terms.items():
            for synonym in synonyms:
                if synonym.lower() in question_lower:
                    concepts['geography'].append(geo)
                    concepts['keywords'].append(synonym)
                    break
        
        return concepts
    
    def generate_search_queries(self, question: str) -> List[str]:
        """
        Génère plusieurs variantes de recherche pour une question
        """
        concepts = self.extract_key_concepts(question)
        queries = [question]  # Requête originale
        
        # Générer des requêtes simples par mots-clés
        if concepts['crops']:
            for crop in concepts['crops']:
                queries.append(crop.replace('_', ' '))
                
                # Combinaisons avec techniques
                if concepts['techniques']:
                    for tech in concepts['techniques']:
                        queries.append(f"{crop.replace('_', ' ')} {tech}")
        
        # Requêtes techniques pures
        if concepts['techniques']:
            for tech in concepts['techniques']:
                queries.append(tech)
        
        # Nettoyer et dédupliquer
        unique_queries = []
        for q in queries:
            q_clean = q.strip()
            if q_clean and q_clean not in unique_queries:
                unique_queries.append(q_clean)
        
        return unique_queries
    
    def prioritize_documents_by_source(self, documents: List[Dict], concepts: Dict) -> List[Dict]:
        """
        Priorise les documents en fonction des concepts détectés
        """
        scored_docs = []
        
        for doc in documents:
            score = 0
            source = doc.get('source', '').lower()
            content = doc.get('page_content', '').lower()
            
            # Bonus pour les documents TPS spécialisés
            if concepts['crops']:
                for crop in concepts['crops']:
                    crop_name = crop.replace('_', ' ')
                    if f"tps {crop_name}" in source or crop_name in source:
                        score += 100  # Bonus élevé pour TPS spécialisé
                    if crop_name in content:
                        score += 50   # Bonus pour contenu pertinent
            
            # Bonus pour les termes techniques
            if concepts['techniques']:
                for tech in concepts['techniques']:
                    if tech in content:
                        score += 30
            
            # Bonus pour la géographie
            if concepts['geography']:
                for geo in concepts['geography']:
                    if geo in content:
                        score += 20
            
            scored_docs.append({
                **doc,
                'relevance_score': score
            })
        
        # Trier par score de pertinence décroissant
        return sorted(scored_docs, key=lambda x: x.get('relevance_score', 0), reverse=True)
    
    def enhance_search_results(self, question: str, raw_results: List[Dict]) -> List[Dict]:
        """
        Améliore les résultats de recherche avec le scoring intelligent
        """
        concepts = self.extract_key_concepts(question)
        
        logger.info(f"🧠 Concepts détectés: {concepts}")
        
        # Prioriser les documents
        enhanced_results = self.prioritize_documents_by_source(raw_results, concepts)
        
        # Log des améliorations
        tps_count = sum(1 for doc in enhanced_results[:5] 
                       if any(crop.replace('_', ' ') in doc.get('source', '').lower() 
                             for crop in concepts.get('crops', [])))
        
        logger.info(f"📊 Top 5 résultats: {tps_count}/5 documents TPS pertinents")
        
        return enhanced_results
