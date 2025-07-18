from langchain_core.tools import tool
from core.llm_providers import LLMManager
from utils.search_cv import parse_user_query, search_cv_local
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
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

@tool
def recruit_candidates(query: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Recherche des profils candidats en fonction de la requête et des données analytiques optionnelles.
    Args:
        query (str): Description du besoin en recrutement.
        data (Optional[Dict]): Données analytiques (issue par exemple d'un autre agent Data Analyst).
    Returns:
        Dict[str, Any]: Résultat contenant les profils candidats et métadonnées.
    """
    try:
        data_analytics = data or {}

        if not query or len(query) > 500:
            raise ValueError("Requête invalide (vide ou >500 caractères)")

        localisation_projet = data_analytics.get("localisation", "Non spécifiée")
        capacites_disponibles = data_analytics.get("capacites_disponibles", [])
        budget_utilisateur = data_analytics.get("budget_utilisateur") or 0
        delai_utilisateur = data_analytics.get("delai_utilisateur_jours") or 0

        seuil_min_effectif = 5
        distance_max_km = 50

        competences_exclues = set()
        for c in capacites_disponibles:
            competence = c.get("competence", "").lower()
            effectif = c.get("effectif", 0)
            if effectif >= seuil_min_effectif:
                competences_exclues.add(competence)

        criteria = parse_user_query(query[:500])
        competences_recherchees = criteria.get("competences", [])

        if not competences_recherchees:
            logger.warning("Aucune compétence extraite via la requête. On récupère depuis data_analytics.")
            competences_recherchees = data_analytics.get("competences_manquantes", [])

        competences_retenues = [c for c in competences_recherchees if c.lower() not in competences_exclues]

        if not competences_retenues and competences_recherchees:
            return {
                "status": "success",
                "profils": [],
                "message": "❌ Aucune compétence à rechercher — mais aucune compétence interne détectée non plus. Vérifiez les données internes fournies.",
                "competences_evitees": list(competences_exclues),
                "competences_recherchees_initiales": competences_recherchees
            }

        criteria["competences"] = competences_retenues
        criteria["localisation"] = localisation_projet

        raw_profils = search_cv_local(criteria)

        profils = []
        competences_trouvees_glob = set()

        for p in raw_profils:
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
                competences_trouvees_glob.update({c.lower() for c in profil.competences})
            except Exception as e:
                logger.warning(f"Profil invalide ignoré: {str(e)}")
                continue

        def score_profil(profil):
            profil_comp = {c.lower() for c in profil.get("competences", [])}
            score = 0
            for c in competences_retenues:
                poids = 1 if c.lower() in {"python", "excel", "sql", "java"} else 2
                if c.lower() in profil_comp:
                    score += poids

            dispo = profil.get("disponibilite", "").lower()
            if "immédiate" in dispo or "1 semaine" in dispo:
                score += 3
            elif "mois" in dispo:
                score -= 2

            niveau = profil.get("experience_niveau", "").lower()
            if budget_utilisateur:
                if "senior" in niveau:
                    score -= 3
                elif "junior" in niveau:
                    score += 1

            return score

        profils.sort(key=score_profil, reverse=True)
        profils = profils[:10]

        competences_non_couvertes = [
            c for c in competences_retenues if c.lower() not in competences_trouvees_glob
        ]

        resume_global = f"🔍 Recrutement pour {len(competences_retenues)} compétences non couvertes.\n"
        resume_global += f"📍 Localisation cible : {localisation_projet}\n"
        if budget_utilisateur:
            resume_global += f"💰 Budget disponible : {budget_utilisateur} €\n"
        if delai_utilisateur:
            resume_global += f"⏳ Délai disponible : {delai_utilisateur} jours\n"
        if not profils:
            resume_global += "❌ Aucun profil trouvé dans les critères.\n"
        elif len(profils) < len(competences_retenues):
            resume_global += f"⚠️ Profils partiellement disponibles. Il manque {len(competences_non_couvertes)} compétences clés.\n"
        else:
            resume_global += "✅ Profils potentiels trouvés pour toutes les compétences requises.\n"
        resume_global += "📈 À valider par les agents onboarding/talent."

        if not profils:
            message = "❌ Aucun profil trouvé correspondant aux compétences recherchées."
        elif competences_non_couvertes:
            message = f"⚠️ Profils partiellement disponibles. Il manque {len(competences_non_couvertes)} compétences clés."
        else:
            message = "✅ Profils trouvés pour toutes les compétences requises."

        return {
            "status": "success",
            "profils": profils,
            "count": len(profils),
            "competences_evitees": list(competences_exclues),
            "competences_recherchees_initiales": competences_recherchees,
            "competences_recherchees_finales": competences_retenues,
            "competences_non_trouvees": competences_non_couvertes,
            "localisation_projet": localisation_projet,
            "budget_utilisateur": budget_utilisateur,
            "delai_utilisateur_jours": delai_utilisateur,
            "distance_max_km": distance_max_km,
            "resume_global": resume_global,
            "message": message
        }

    except Exception as e:
        logger.error(f"[recruit_candidates] Erreur : {str(e)}", exc_info=True)
        return {
            "status": "error",
            "profils": [],
            "error": str(e)[:200]
        }
