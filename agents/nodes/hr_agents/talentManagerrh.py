from langchain_core.runnables import Runnable
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from core.llm_providers import LLMManager
from agents.nodes.hr_agents.schema import TalentPlan
from typing import Dict, Any
import logging

class TalentManagerAgent(Runnable):
    def __init__(self, llm=None):
        self.llm = llm or LLMManager().get_llm()
        self.parser = PydanticOutputParser(pydantic_object=TalentPlan)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
        self.logger = logging.getLogger(__name__)

    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        query = str(input.get("query", ""))[:2000]
        state = input.get("state", {})
        data_analytics = state.get("data_analytics", {})

        budget_moyen = data_analytics.get("budget_moyen", "inconnu")
        delai_moyen = data_analytics.get("delai_moyen_lancement_projet", "inconnu")
        competences = data_analytics.get("capacites_disponibles") or []
        competences_str = ", ".join(c.get("competence", "") for c in competences)

        contexte_entreprise = (
            f"Budget moyen des projets: {budget_moyen} €, "
            f"Délai moyen lancement: {delai_moyen} jours, "
            f"Compétences internes disponibles: {competences_str}"
        )

        try:
            prompt = f"""
            Tu es un expert en gestion des compétences et formation interne.
            Tu dois évaluer si l'entreprise peut faire monter en compétence ses salariés pour un nouveau projet.

            Contexte projet + marché:
            {query}

            Contexte interne de l'entreprise:
            {contexte_entreprise}

            Réponds avec un JSON strictement conforme au schéma suivant:
            - upskill_possible: bool (true/false si la montée en compétence est possible)
            - duree_formation: str (durée estimée de formation)
            - referentiel_competences: str ("oui" ou "non" selon l'existence d'un référentiel)
            - plan_global: str (description concise du plan de formation)

            Exemple de réponse valide:
            {{
                "upskill_possible": true,
                "duree_formation": "6 semaines",
                "referentiel_competences": "oui",
                "plan_global": "Formation intensive sur les énergies vertes avec certification"
            }}

            Ta réponse (UNIQUEMENT le JSON, sans commentaires):
            """.strip()

            response = self.llm.invoke(prompt)
            self.logger.debug(f"Raw LLM response: {response}")

            cleaned_response = response
            if isinstance(cleaned_response, dict):
                cleaned_response = cleaned_response.get("response", "")
            cleaned_response = cleaned_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            parsed = self.fixing_parser.parse(cleaned_response)
            return parsed.dict()

        except Exception as e:
            self.logger.error(f"TalentManagerAgent error: {str(e)}", exc_info=True)
            return {
                "upskill_possible": False,
                "duree_formation": "Non estimable",
                "referentiel_competences": "non",
                "plan_global": f"Erreur lors de l'analyse: {str(e)[:200]}"
            }

    async def ainvoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return self.invoke(input)
