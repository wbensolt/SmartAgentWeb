from langchain_core.runnables import Runnable
from core.llm_providers import LLMManager
from utils.search_cv import parse_user_query, search_cv_local
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import json
import logging

logger = logging.getLogger(__name__)

class ProfilCandidat(BaseModel):
    id: str = Field(..., description="Identifiant unique du candidat")
    nom: str = Field(..., description="Initiale du nom (anonymisé)", max_length=5)
    competences: List[str] = Field(..., description="Liste des compétences clés", max_items=15)
    localisation: str = Field(..., description="Ville ou région du candidat", max_length=50)
    experience_niveau: str = Field("", description="Niveau d'expérience", max_length=20)
    statut: str = Field("", description="Statut actuel", max_length=20)
    disponibilite: str = Field("", description="Délai de disponibilité", max_length=20)

class RecruiterAgent(Runnable):
    def __init__(self, llm=None):
        self.llm = llm or LLMManager().get_llm()
        self.logger = logging.getLogger(__name__)
        
    def invoke(self, input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        try:
            query = str(input.get("query", "")).strip()
            state: Dict[str, Any] = input.get("state", {})

            if not query or len(query) > 500:
                raise ValueError("Requête invalide (vide ou >500 caractères)")
            
            # 🔍 Extraction des capacités internes déjà disponibles
            data_analytics = state.get("data_analytics", {})
            capacites_disponibles = data_analytics.get("capacites_disponibles", [])
            competences_exclues = {c["competence"].lower() for c in capacites_disponibles}

            # 🔍 Extraction des besoins à partir de la requête
            criteria = parse_user_query(query[:500])
            competences_recherchees = criteria.get("competences", [])
            competences_retenues = [c for c in competences_recherchees if c.lower() not in competences_exclues]

            if not competences_retenues:
                return {
                    "status": "success",
                    "profils": [],
                    "message": "Aucune compétence à rechercher : déjà disponibles en interne.",
                    "competences_evitees": list(competences_exclues)
                }

            # 🔎 Recherche ciblée
            criteria["competences"] = competences_retenues
            raw_profils = search_cv_local(criteria)

            profils = []
            for p in raw_profils[:10]:
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
                "count": len(profils),
                "competences_evitees": list(competences_exclues),
                "competences_recherchees_finales": competences_retenues
            }

        except Exception as e:
            logger.error(f"[RecruiterAgent] Erreur : {str(e)}")
            return {
                "status": "error",
                "profils": [],
                "error": str(e)[:200]
            }

    async def ainvoke(self, input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self.invoke(input, **kwargs)
