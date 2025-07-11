from langchain_core.runnables import Runnable
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from agents.nodes.hr_agents.schema import PayrollPlan
from core.llm_providers import LLMManager
from typing import Dict, Any, List
from pydantic import BaseModel
import logging

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

        prompt = f"""
[SYSTEM]
Tu es un expert paie et budget.

Calcule le coût mensuel approximatif et analyse la faisabilité du budget projet.

Réponds STRICTEMENT en JSON VALIDE au format suivant :
{{
  "cout_mensuel": "ex: 133k€",
  "budget_suffisant": "oui|non",
  "analyse_cout": "Explication brève de ton évaluation budgétaire",
  "recommandations": ["Suggestion 1", "Suggestion 2"]
}}

- Aucune phrase ou bloc de code en dehors du JSON.
- Sois cohérent avec les profils envisagés (junior, alternants…).

Contexte :
{query}
"""

        try:
            response = self.llm.invoke(prompt)
            self.logger.debug(f"Réponse brute LLM : {response}")

            # Nettoyage simple si nécessaire
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            # Parsing + correction automatique
            parsed = self.fixing_parser.parse(cleaned_response)
            return parsed.dict()

        except Exception as e:
            self.logger.error(f"PayrollAgent error: {str(e)}", exc_info=True)
            return self._default_response.dict()

    async def ainvoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return self.invoke(input)
