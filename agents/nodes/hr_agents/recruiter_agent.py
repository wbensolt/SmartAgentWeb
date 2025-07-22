# agents/nodes/hr_agents/recruiter_agent.py

import json
from langchain_core.tools import tool
from utils.search_cv import parse_user_query, search_cv_local
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class ProfilCandidat(BaseModel):
    id: str
    nom: str
    competences: List[str]
    localisation: str
    experience_niveau: str
    statut: str
    disponibilite: str

@tool
def recruit_candidates(query: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Recherche et filtre des profils candidats adaptés à un besoin de recrutement donné.

    La fonction analyse une requête textuelle décrivant le besoin en recrutement,
    enrichie éventuellement par des données analytiques (ex. issues d'un agent Data Analyst),
    pour extraire les compétences clés, la localisation, le budget, et autres critères.

    Elle exclut les compétences internes largement disponibles (pour éviter de recruter des profils
    déjà présents), recherche localement des CV correspondant aux critères restants, 
    puis classe et filtre les profils selon un score basé sur compétences, disponibilité et niveau.

    Args:
        query (str): Description textuelle du besoin en recrutement (max 500 caractères).
        data (Optional[Dict[str, Any]]): Données analytiques optionnelles
            (ex. localisation, budget, compétences disponibles/manquantes).

    Returns:
        Dict[str, Any]: Résultat contenant :
            - "status": "success" ou "error"
            - "profils": liste des profils candidats (anonymisés, avec compétences, localisation, etc.)
            - "count": nombre de profils retournés
            - "competences_evitees": compétences internes exclues de la recherche
            - "competences_recherchees_initiales": compétences extraites initialement
            - "competences_recherchees_finales": compétences retenues après exclusion
            - "competences_non_trouvees": compétences recherchées non couvertes par les profils trouvés
            - "localisation_projet": localisation ciblée pour la recherche
            - "budget_utilisateur": budget disponible estimé
            - "delai_utilisateur_jours": délai en jours estimé
            - "distance_max_km": distance maximale pour localisation des candidats
            - "resume_global": résumé synthétique du résultat
            - "message": message utilisateur contextualisé (ex. absence de profils)

    Raises:
        ValueError: Si la requête est vide ou dépasse 500 caractères.

    Notes:
        - Les profils sont triés selon un score prenant en compte les compétences,
          la disponibilité et le niveau d'expérience.
        - En cas d'absence de données analytiques, la fonction appelle l'agent Data Analyst pour les obtenir.
    """
    logger.info("🔎 Début de la recherche de candidats")
    logger.info(f"Requête: {query}")
    if data:
        logger.info(f"Données analytiques reçues: {json.dumps(data, indent=2)}")
    try:
        data = data or {}
        capacites_disponibles = data.get("capacites_disponibles", [])
        localisation = data.get("localisation", "Non spécifiée")
        budget = data.get("budget_utilisateur") or 0
        delai = data.get("delai_utilisateur_jours") or 0

        competences_exclues = {c["competence"].lower() for c in capacites_disponibles if c["effectif"] >= 5}
        criteria = parse_user_query(query)
        competences = [c for c in criteria.get("competences", []) if c.lower() not in competences_exclues]

        criteria["competences"] = competences
        criteria["localisation"] = localisation

        raw_profils = search_cv_local(criteria)
        profils = [ProfilCandidat(**{
            "id": str(p["id"])[:20],
            "nom": p.get("nom", "X")[:1] + ".",
            "competences": p.get("competences", []),
            "localisation": p.get("localisation", ""),
            "experience_niveau": p.get("experience_niveau", ""),
            "statut": p.get("statut", ""),
            "disponibilite": p.get("disponibilite", "")
        }).dict() for p in raw_profils]

        return {
            "status": "success",
            "profils": profils,
            "budget_utilisateur": budget,
            "delai_utilisateur_jours": delai,
            "competences_exclues": list(competences_exclues)
        }
    except Exception as e:
        logger.error(str(e))
        return {"status": "error", "error": str(e)}


def recruit_candidates_wrapper(json_input: str) -> str:
    payload = json.loads(json_input)
    query = payload.get("input", "")
    data = payload.get("data", {})
    result_dict = recruit_candidates(query, data)
    return json.dumps(result_dict, ensure_ascii=False)
