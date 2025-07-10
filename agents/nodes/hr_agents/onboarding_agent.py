from langchain_core.runnables import Runnable
from core.llm_providers import LLMManager
from typing import Dict, Any, List
import json
from datetime import timedelta
from dateutil.parser import parse
import logging

class OnboardingAgent(Runnable):
    def __init__(self, llm=None):
        self.llm = LLMManager().get_llm()
        self.logger = logging.getLogger(__name__)
        self.default_time_estimate = timedelta(days=30)

    def _default_response(self, team_size: int) -> Dict[str, Any]:
        """Réponse par défaut en cas d'erreur"""
        return {
            "accueil": {
                "resources": {
                    "rh": 1,
                    "tutors": max(1, team_size // 5),
                    "tools": ["Manuel d'accueil", "Kit d'intégration"]
                },
                "timeline": {
                    "estimated_duration": "30 jours",
                    "per_person_days": 2
                },
                "risks": ["Charge des tuteurs", "Retard documentation"],
                "checklist": [
                    "Préparer les postes de travail",
                    "Planifier les sessions de formation"
                ],
                "error": "Fallback to default plan"
            }
        }

    def _validate_response(self, response: Dict) -> bool:
        """Valide la structure de la réponse"""
        required_keys = {
            "resources": ["rh", "tutors", "tools"],
            "timeline": ["estimated_duration", "per_person_days"],
            "risks": None,
            "checklist": None
        }
        try:
            for section, subkeys in required_keys.items():
                if section not in response:
                    return False
                if subkeys:
                    for key in subkeys:
                        if key not in response[section]:
                            return False
            return True
        except Exception:
            return False

    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Version robuste avec :
        - Validation JSON stricte
        - Gestion d'erreurs complète
        - Limitation des entrées/sorties
        """
        try:
            # 1. Nettoyage des entrées
            query = str(input.get("query", ""))[:1000]  # Limite à 1000 caractères
            team_size = min(int(input.get("team_size", 15)), 50)  # Max 50 personnes
            start_date = input.get("start_date")

            # 2. Prompt structuré avec exemple de sortie
            prompt = f"""
                    [SYSTEM]
                    Tu es expert en intégration RH.

                    Tu dois estimer les besoins pour accueillir des collaborateurs dans de bonnes conditions (RH, matériel, tuteurs, planning).

                    Réponds uniquement avec un JSON VALIDE :
                    {
                    "resources": {
                        "rh": nombre_rh,
                        "tutors": nombre_tuteurs,
                        "tools": ["outil 1", "outil 2"]
                    },
                    "timeline": {
                        "estimated_duration": "ex: 30 jours",
                        "per_person_days": nombre
                    },
                    "risks": ["risque 1", "risque 2"],
                    "checklist": ["étape 1", "étape 2"]
                    }
                    - Aucune balise markdown.
                    """

            # 3. Appel LLM sécurisé
            raw_response = self.llm.invoke(prompt)
            
            # 4. Parsing et validation
            if isinstance(raw_response, str):
                try:
                    response = json.loads(raw_response)
                except json.JSONDecodeError:
                    self.logger.warning("Réponse JSON invalide")
                    return self._default_response(team_size)
            else:
                response = raw_response

            if not self._validate_response(response):
                self.logger.warning("Structure de réponse invalide")
                return self._default_response(team_size)

            # 5. Post-traitement
            plan = {"accueil": response}
            
            # Calcul date de fin si start_date fourni
            if start_date:
                try:
                    days = int(response["timeline"]["estimated_duration"].split()[0])
                    end_date = parse(start_date) + timedelta(days=days)
                    plan["accueil"]["timeline"]["completion_date"] = end_date.isoformat()
                except Exception as e:
                    self.logger.error(f"Erreur calcul date: {str(e)}")
                    plan["accueil"]["error"] = "Invalid date calculation"

            return plan

        except Exception as e:
            self.logger.error(f"Erreur majeure: {str(e)}")
            return self._default_response(team_size)

    async def ainvoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return self.invoke(input)