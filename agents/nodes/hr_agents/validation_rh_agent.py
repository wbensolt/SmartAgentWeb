from langchain_core.tools import tool
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from core.llm_providers import LLMManager
from agents.nodes.hr_agents.schema import ValidationRHPlan
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

@tool("ValidationRHTool")
def validation_agent_rh_tool(input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valide une critique RH donnée en analysant sa pertinence et sa qualité.

    Cette fonction reçoit un dictionnaire contenant une critique RH sous forme de texte,
    puis interroge un LLM pour déterminer si cette critique est valide ou non, 
    avec une justification courte. La réponse est formatée en JSON conforme au modèle ValidationRHPlan.

    Args:
        input (Dict[str, Any]): Dictionnaire contenant la clé :
            - "critique" (str) : texte de la critique à valider (max 5000 caractères).

    Returns:
        Dict[str, Any]: Résultat JSON avec les champs :
            - "validation" (str) : "valide" ou "non valide".
            - "justification" (str) : brève justification (3-5 mots).

    Notes:
        - La fonction répond uniquement avec un JSON strict.
        - En cas d'absence de critique ou d'erreur, une réponse par défaut est retournée.
    """

    llm = LLMManager().get_llm()
    parser = PydanticOutputParser(pydantic_object=ValidationRHPlan)
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

    default_response = ValidationRHPlan(
        validation="non valide",
        justification="Erreur de traitement"
    )

    critique = str(input.get("critique", "")).strip()[:5000]
    if not critique:
        return {
            "validation": "non valide",
            "justification": "Critique vide"
        }

    prompt = f"""
Validation RH - Format JSON strict

Critique à évaluer:
{critique[:2000]}

Réponds UNIQUEMENT avec ce format JSON STRICT :
{{
    "validation": "valide|non valide",
    "justification": "3-5 mots maximum"
}}
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
        logger.error(f"Erreur validation_rh_tool: {str(e)}", exc_info=True)
        return default_response.dict()
