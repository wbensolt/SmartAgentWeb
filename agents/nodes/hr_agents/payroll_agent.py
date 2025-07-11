from langchain_core.runnables import Runnable
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from agents.nodes.hr_agents.schema import PayrollPlan
from core.llm_providers import LLMManager
from typing import Dict, Any
from pydantic import BaseModel
import logging
import json

class PayrollAgent(Runnable):
    def __init__(self, llm=None):
        self.llm = llm or LLMManager().get_llm()
        self.parser = PydanticOutputParser(pydantic_object=PayrollPlan)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
        self.logger = logging.getLogger(__name__)
        self._default_response = PayrollPlan(
            cout_mensuel="erreur",
            budget_suffisant="non",
            analyse_cout="Erreur d'analyse",
            recommandations=["Contacter le service RH"]
        )

    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        query = str(input.get("query", ""))[:500]
        state: Dict[str, Any] = input.get("state", {})

        # 🔍 Récupérer le contexte de l’entreprise si disponible
        data_analytics = state.get("data_analytics", {})
        budget_moyen = data_analytics.get("budget_moyen", "inconnu")
        capacites_disponibles = data_analytics.get("capacites_disponibles", [])
        capacites_json = json.dumps(capacites_disponibles, ensure_ascii=False, indent=2)

        # 🔧 Prompt enrichi avec contexte interne
        prompt = f"""
[SYSTEM]
Tu es un expert paie et budget.

Ton rôle est d'estimer le coût mensuel approximatif du projet, en analysant si le budget est suffisant, tout en tenant compte des données internes de l'entreprise.

Contexte projet :
{query}

Contexte entreprise :
- Budget moyen des projets réalisés : {budget_moyen} €
- Compétences déjà disponibles (à ne pas recruter de nouveau) :
{capacites_json}

Réponds STRICTEMENT avec un JSON VALIDE au format suivant :
{{
  "cout_mensuel": "ex: 133k€",
  "budget_suffisant": "oui|non",
  "analyse_cout": "Explication brève de ton évaluation budgétaire",
  "recommandations": ["Suggestion 1", "Suggestion 2"]
}}

- N'inclus aucune phrase ou balise de code en dehors du JSON.
- Sois cohérent avec les profils envisagés (junior, alternants…).
- Si des compétences internes existent, priorise leur usage pour réduire les coûts.
"""

        try:
            response = self.llm.invoke(prompt)
            self.logger.debug(f"Réponse brute LLM : {response}")

            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            parsed = self.fixing_parser.parse(cleaned_response)
            return parsed.dict()

        except Exception as e:
            self.logger.error(f"PayrollAgent error: {str(e)}", exc_info=True)
            return self._default_response.dict()

    async def ainvoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return self.invoke(input)
