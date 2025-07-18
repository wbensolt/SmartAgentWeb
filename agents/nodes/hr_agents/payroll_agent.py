from langchain_core.tools import tool
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from agents.nodes.hr_agents.schema import PayrollPlan
from core.llm_providers import LLMManager
from typing import Dict, Any
import logging
import json

logger = logging.getLogger(__name__)

@tool("PayrollTool")
def payroll_agent_tool(input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Estime le coût mensuel d'un projet RH et analyse si le budget est suffisant.

    Cette fonction utilise un LLM pour évaluer le coût mensuel approximatif du projet
    à partir d'une description utilisateur et du contexte interne de l'entreprise,
    notamment le budget moyen des projets et les compétences disponibles.

    Le LLM doit retourner un JSON strict comprenant :
    - cout_mensuel : estimation du coût mensuel (ex : "133k€")
    - budget_suffisant : "oui" ou "non" selon l'adéquation du budget
    - analyse_cout : explication brève de l'évaluation budgétaire
    - recommandations : liste de suggestions pour optimiser le budget ou le projet

    Args:
        input (Dict[str, Any]): Dictionnaire contenant :
            - "query" (str) : description textuelle du projet (limité à 500 caractères)
            - "state" (dict, optionnel) : contexte interne avec données analytiques 
              (ex : budget moyen, compétences disponibles)

    Returns:
        Dict[str, Any]: Résultat analysé conforme au schéma PayrollPlan,
        ou une réponse par défaut en cas d'erreur.
    """
    logger.info("[PayrollAgent] Analyse du prompt projet lancé.")
    llm = LLMManager().get_llm()
    parser = PydanticOutputParser(pydantic_object=PayrollPlan)
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

    default_response = PayrollPlan(
        cout_mensuel="erreur",
        budget_suffisant="non",
        analyse_cout="Erreur d'analyse",
        recommandations=["Contacter le service RH"]
    )

    query = str(input.get("query", ""))[:500]
    state: Dict[str, Any] = input.get("state", {})

    data_analytics = state.get("data_analytics", {})
    budget_moyen = data_analytics.get("budget_moyen", "inconnu")
    capacites_disponibles = data_analytics.get("capacites_disponibles", [])
    capacites_json = json.dumps(capacites_disponibles, ensure_ascii=False, indent=2)

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
        response = llm.invoke(prompt)
        logger.debug(f"Réponse brute LLM : {response}")

        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()

        parsed = fixing_parser.parse(cleaned_response)
        return parsed.dict()

    except Exception as e:
        logger.error(f"PayrollTool error: {str(e)}", exc_info=True)
        return default_response.dict()
