from langchain_core.runnables import Runnable
from agents.nodes.hr_agents.schema import OnboardingPlan
from core.llm_providers import LLMManager
from typing import Dict, Any, Optional
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from datetime import timedelta
from dateutil.parser import parse
import logging
import json

class OnboardingAgent(Runnable):
    def __init__(self, llm=None):
        self.llm = llm or LLMManager().get_llm()
        self.logger = logging.getLogger(__name__)
        self.parser = PydanticOutputParser(pydantic_object=OnboardingPlan)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
        self.default_time_estimate = timedelta(days=30)

    def _default_response(self, team_size: int) -> Dict[str, Any]:
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

    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        team_size = min(int(input.get("team_size", 15)), 50)
        start_date = input.get("start_date", None)
        query = str(input.get("query", ""))[:1000]
        state: Dict[str, Any] = input.get("state", {})

        try:
            # 🔍 Récupération des capacités internes
            data_analytics = state.get("data_analytics", {})
            capacites_disponibles = data_analytics.get("capacites_disponibles", [])
            capacites_json = json.dumps(capacites_disponibles, ensure_ascii=False, indent=2)

            prompt = f"""
Tu es expert en intégration RH.

Tu dois estimer les besoins pour accueillir des collaborateurs dans de bonnes conditions (RH, matériel, tuteurs, planning).

Contexte du projet :
{query}

Capacités internes disponibles (employés formateurs ou RH, outils, etc.) :
{capacites_json}

Estime les besoins supplémentaires pour l'onboarding, en tenant compte de ce qui existe déjà.

Réponds uniquement avec un JSON VALIDE (sans balises markdown) correspondant au schéma suivant :

{{
  "resources": {{
    "rh": int,
    "tutors": int,
    "tools": [str]
  }},
  "timeline": {{
    "estimated_duration": "ex: 30 jours",
    "per_person_days": int
  }},
  "risks": [str],
  "checklist": [str]
}}
            """

            raw_response = self.llm.invoke(prompt).strip()
            if raw_response.startswith("```json"):
                raw_response = raw_response[7:]
            if raw_response.endswith("```"):
                raw_response = raw_response[:-3]
            cleaned_response = raw_response.strip()

            parsed = self.fixing_parser.parse(cleaned_response)
            response = parsed.dict()

            plan = {"accueil": response}

            # Ajout date de fin si date de début précisée
            if start_date:
                try:
                    days = int(response["timeline"]["estimated_duration"].split()[0])
                    end_date = parse(start_date) + timedelta(days=days)
                    plan["accueil"]["timeline"]["completion_date"] = end_date.isoformat()
                except Exception as e:
                    self.logger.error(f"Erreur calcul date: {e}")
                    plan["accueil"]["error"] = "Invalid date calculation"

            return plan

        except Exception as e:
            self.logger.error(f"Erreur majeure OnboardingAgent: {e}", exc_info=True)
            return self._default_response(team_size)

    async def ainvoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return self.invoke(input)
