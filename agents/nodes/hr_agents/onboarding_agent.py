from langchain_core.tools import tool
from core.llm_providers import LLMManager
from agents.nodes.hr_agents.schema import OnboardingPlan
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from dateutil.parser import parse
from datetime import timedelta
import json
import logging
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)

@tool("OnboardingAgent")
def onboarding_agent_tool(input: str) -> Dict[str, Any]:
    """
    Estime les besoins d'onboarding RH, matériel, tuteurs et planning pour un projet donné.

    Analyse un projet RH à partir d'une requête utilisateur (input JSON) incluant :
    - "query" (str) : description du projet
    - "team_size" (int, optionnel) : taille de l'équipe, maximale 50
    - "start_date" (str, optionnel) : date de début prévue au format ISO
    - "state" (dict, optionnel) : état/contextes additionnels, notamment données analytiques

    Utilise un LLM pour générer un plan d'onboarding conforme au schéma OnboardingPlan, 
    contenant notamment les ressources nécessaires (RH, tuteurs, outils), la timeline estimée, 
    les risques identifiés et une checklist.

    Si la date de début est fournie, calcule la date de fin estimée en fonction de la durée.

    En cas d'erreur, renvoie un plan par défaut avec une estimation raisonnable.

    Args:
        input (str): JSON sérialisé avec les paramètres décrits ci-dessus.

    Returns:
        Dict[str, Any]: Dictionnaire avec une clé "accueil" contenant le plan d'onboarding 
        au format OnboardingPlan, éventuellement enrichi de la date de fin et/ou d'un message d'erreur.
    """
    
    try:
        params = json.loads(input)
        query = params.get("query", "")
        team_size = min(int(params.get("team_size", 15)), 50)
        start_date = params.get("start_date", None)
        state = params.get("state", {})

        llm = LLMManager().get_llm()
        parser = PydanticOutputParser(pydantic_object=OnboardingPlan)
        fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

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

        raw_response = llm.invoke(prompt).strip()
        match = re.search(r"```json(.*?)```", raw_response, re.DOTALL | re.IGNORECASE)
        cleaned_response = match.group(1).strip() if match else raw_response

        parsed = fixing_parser.parse(cleaned_response)
        response = parsed.dict()

        plan = {"accueil": response}

        if start_date:
            try:
                days = int(response["timeline"]["estimated_duration"].split()[0])
                end_date = parse(start_date) + timedelta(days=days)
                plan["accueil"]["timeline"]["completion_date"] = end_date.isoformat()
            except Exception as e:
                logger.error(f"Erreur calcul date: {e}")
                plan["accueil"]["error"] = "Invalid date calculation"

        return plan

    except Exception as e:
        logger.error(f"Erreur majeure OnboardingAgent: {e}", exc_info=True)
        team_size = params.get("team_size", 15) if "params" in locals() else 15
        return {
            "accueil": {
                "resources": {
                    "rh": 1,
                    "tutors": max(1, int(team_size)//5),
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
