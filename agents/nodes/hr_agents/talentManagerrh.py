from langchain_core.runnables import Runnable
from core.llm_providers import LLMManager
from typing import Dict, Any, Literal
import json
import logging

class TalentManagerAgent(Runnable):
    def __init__(self, llm=None):
        self.llm = LLMManager().get_llm()
        self.logger = logging.getLogger(__name__)
        
    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retourne un dictionnaire validé contenant le plan de gestion des talents.
        Garantit toujours le même format de sortie même en cas d'erreur.
        """
        query = str(input.get("query", ""))[:2000]  # Limite la longueur
        
        try:
            # Prompt structuré avec validation stricte
            prompt = f"""
            [SYSTEM]
            Tu es un expert en gestion des compétences et formation interne.

            Tu dois évaluer si l’entreprise peut faire monter en compétence ses salariés pour un nouveau projet.

            Réponds uniquement en JSON VALIDE :
            {
            "upskill_possible": true|false,
            "duree_formation": "Durée estimée ou 'non estimable'",
            "referentiel_competences": "Résumé du référentiel utilisé ou à créer",
            "plan_global": "Résumé du plan de montée en compétence"
            }
            - Interdiction d’utiliser du texte hors JSON ou des apostrophes typographiques.
            """

            # Appel LLM
            response = self.llm.invoke(prompt)
            
            # Parsing et validation
            return self._validate_response(response)
            
        except Exception as e:
            self.logger.error(f"Erreur TalentManager: {str(e)}")
            return self._default_response(str(e))

    def _validate_response(self, response: Any) -> Dict[str, Any]:
        """Valide et transforme la réponse du LLM"""
        # Conversion si string JSON
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                self.logger.warning("Réponse JSON invalide")
                return self._default_response("Format JSON invalide")

        # Vérification du type
        if not isinstance(response, dict):
            return self._default_response("Type de réponse invalide")

        # Validation des champs
        validated = {
            "upskill_possible": bool(response.get("upskill_possible", False)),
            "duree_formation": str(response.get("duree_formation", "Non estimable"))[:50],
            "referentiel_competences": "oui" if str(response.get("referentiel_competences", "")).lower() == "oui" else "non",
            "plan_global": str(response.get("plan_global", ""))[:500]
        }
        
        return validated

    def _default_response(self, error_msg: str = "") -> Dict[str, Any]:
        """Réponse par défaut en cas d'erreur"""
        return {
            "upskill_possible": False,
            "duree_formation": "Non estimable",
            "referentiel_competences": "oui",
            "plan_global": f"Erreur: {error_msg[:200]}" if error_msg else "Analyse indisponible"
        }

    async def ainvoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return self.invoke(input)