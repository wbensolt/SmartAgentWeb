from langchain_core.runnables import Runnable
from core.llm_providers import LLMManager
from utils.search_cv import parse_user_query, search_cv_local
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import json
import logging

logger = logging.getLogger(__name__)

class ProfilCandidat(BaseModel):
    """Modèle Pydantic pour standardiser la structure des profils candidats"""
    id: str = Field(..., description="Identifiant unique du candidat")
    nom: str = Field(..., description="Initiale du nom (anonymisé)", max_length=5)
    competences: List[str] = Field(..., description="Liste des compétences clés", max_items=15)
    localisation: str = Field(..., description="Ville ou région du candidat", max_length=50)
    experience_niveau: str = Field("", description="Niveau d'expérience", max_length=20)
    statut: str = Field("", description="Statut actuel", max_length=20)
    disponibilite: str = Field("", description="Délai de disponibilité", max_length=20)

class RecruiterAgent(Runnable):
    """
    Agent spécialisé dans le sourcing de candidats avec gestion robuste des erreurs.
    
    Args:
        input (Dict): Doit contenir une clé 'query' avec la description du poste
    
    Returns:
        Dict: {
            "status": "success|error",
            "profils": List[ProfilCandidat],
            "error": Optional[str]
        }
    """
    def __init__(self, llm=None):
        self.llm = llm or LLMManager().get_llm()
        self.logger = logging.getLogger(__name__)
        
    def invoke(self, input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Recherche sécurisée de candidats avec format de sortie standardisé.
        
        Example de succès:
            {
                "status": "success",
                "profils": [
                    {
                        "id": "123",
                        "nom": "D.",
                        "competences": ["Python"],
                        ...
                    }
                ]
            }
            
        Example d'erreur:
            {
                "status": "error",
                "profils": [],
                "error": "Message d'erreur"
            }
        """
        try:
            # Validation de l'input
            query = str(input.get("query", "")).strip()
            if not query or len(query) > 500:
                raise ValueError("Requête invalide (vide ou >500 caractères)")
            
            # Recherche des profils
            criteria = parse_user_query(query[:500])  # Limite la longueur
            raw_profils = search_cv_local(criteria)
            
            # Formatage et validation des résultats
            profils = []
            for p in raw_profils[:10]:  # Limite à 10 résultats
                try:
                    profil = ProfilCandidat(
                        id=str(p.get("id", ""))[:20],
                        nom=str(p.get("nom", "X"))[:1] + ".",
                        competences=[str(c)[:50] for c in p.get("competences", [])][:15],
                        localisation=str(p.get("localisation", ""))[:50],
                        experience_niveau=str(p.get("experience_niveau", ""))[:20],
                        statut=str(p.get("statut", ""))[:20],
                        disponibilite=str(p.get("disponibilite", ""))[:20]
                    )
                    profils.append(profil.dict())
                except Exception as e:
                    logger.warning(f"Profil invalide ignoré: {str(e)}")
                    continue
            
            return {
                "status": "success",
                "profils": profils,
                "count": len(profils)
            }
            
        except Exception as e:
            logger.error(f"Erreur RecruiterAgent: {str(e)}")
            return {
                "status": "error",
                "profils": [],
                "error": str(e)[:200]  # Limite la longueur du message
            }

    async def ainvoke(self, input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Version asynchrone avec la même logique"""
        return self.invoke(input, **kwargs)

    def _validate_output(self, data: Dict) -> bool:
        """Validation supplémentaire du format de sortie"""
        required_keys = {"status", "profils"}
        return all(key in data for key in required_keys)