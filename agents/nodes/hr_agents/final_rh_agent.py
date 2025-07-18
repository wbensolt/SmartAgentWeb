from langchain_core.tools import tool
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from core.llm_providers import LLMManager
from agents.nodes.hr_agents.schema import FinalRHPlan
from typing import Dict, Any
import logging
import json

logger = logging.getLogger(__name__)

@tool("FinalRHTool")
def final_agent_rh_tool(input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synthétise les résultats de plusieurs agents RH spécialisés pour produire une 
    décision finale claire et structurée sur la faisabilité d’un projet RH.

    Utilise un LLM pour analyser les réponses, critiques et validations fournies,
    et retourne un JSON validé contenant :
      - faisabilite (str) : "Oui", "Non" ou "Partiel"
      - conditions_reussite (List[str]) : conditions nécessaires à la réussite
      - score_confiance (float) : score de confiance entre 0.0 et 1.0
      - recommandation (str) : synthèse concise et exploitable
      - risques_principaux (List[str]) : principaux risques identifiés

    Args:
        input (Dict[str, Any]): Dictionnaire avec les clés suivantes :
            - "answers": réponses des agents spécialisés (dict)
            - "critiques": critiques issues des analyses (dict)
            - "validations": résultats des validations (dict)

    Returns:
        Dict[str, Any]: Dictionnaire conforme au modèle FinalRHPlan décrivant la synthèse finale.

    En cas d'erreur, retourne une réponse par défaut indiquant une erreur technique.
    """
    llm = LLMManager().get_llm()
    parser = PydanticOutputParser(pydantic_object=FinalRHPlan)
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

    default_response = FinalRHPlan(
        faisabilite="Non",
        conditions_reussite=["Vérifier les logs système"],
        score_confiance=0.0,
        recommandation="Erreur technique dans l'analyse",
        risques_principaux=["Erreur technique dans l'analyse"]
    )

    try:
        answers = input.get("answers", {})
        critiques = input.get("critiques", {})
        validations = input.get("validations", {})

        answers_str = json.dumps(answers, ensure_ascii=False)[:1500]
        critiques_str = json.dumps(critiques, ensure_ascii=False)[:1000]
        validations_str = json.dumps(validations, ensure_ascii=False)[:1000]

        prompt = f"""
[SYSTEM]
Tu es un expert en synthèse RH stratégique.

À partir des réponses de plusieurs agents spécialisés, tu dois donner une décision métier claire sur la faisabilité d’un projet RH.

Réponds uniquement avec un JSON VALIDE au format strict :
{{
  "faisabilite": "Oui|Non|Partiel",
  "conditions_reussite": ["condition 1", "condition 2"],
  "score_confiance": 0.0 à 1.0,
  "recommandation": "Phrase synthétique et exploitable",
  "risques_principaux": ["risque 1", "risque 2"]
}}

Réponses des agents spécialisés :
- Answers: {answers_str}
- Critiques: {critiques_str}
- Validations: {validations_str}

Ne jamais inclure de texte hors JSON.
Corrige les erreurs de format si besoin.
"""

        raw_response = llm.invoke(prompt)
        logger.debug(f"[FinalRHTool] Réponse brute LLM : {repr(raw_response)}")

        cleaned_response = raw_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()

        parsed = fixing_parser.parse(cleaned_response)
        return parsed.dict()

    except Exception as e:
        logger.error(f"[FinalRHTool] Erreur invoke : {str(e)}", exc_info=True)
        return default_response.dict()
