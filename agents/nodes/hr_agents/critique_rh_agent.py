from langchain_core.tools import tool
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from core.llm_providers import LLMManager
from agents.nodes.hr_agents.schema import CritiqueRHPlan
from typing import Dict, Any
import logging
import re

logger = logging.getLogger(__name__)

def _extract_json_block(text: str) -> str | None:
    """
    Tente d'extraire un bloc JSON valide à partir d'une réponse textuelle (éventuellement encadrée par ```json ... ```).
    
    Args:
        text (str): Texte brut retourné par le LLM.

    Returns:
        str | None: Bloc JSON sous forme de chaîne, ou None si aucun bloc trouvé.
    """
    match = re.search(r"```json(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"```(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    json_start = text.find("{")
    if json_start != -1:
        return text[json_start:].strip()
    return None


@tool("CritiqueRHTool")
def critique_agent_rh_tool(input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyse un contenu RH (plan, stratégie, politique, etc.) et fournit une critique constructive
    sous forme de JSON structuré. Utilise un LLM pour extraire les points forts, points faibles,
    suggestions, une note de cohérence, un commentaire global et un indicateur d'erreur.

    Args:
        input (Dict[str, Any]): Dictionnaire contenant le champ "content" avec le texte à analyser.

    Returns:
        Dict[str, Any]: Résultat structuré conforme au schéma CritiqueRHPlan, contenant :
            - points_forts (List[str])
            - points_faibles (List[str])
            - suggestions (List[str])
            - note_coherence (int)
            - commentaire_global (str)
            - erreur (bool)
    
    Exemple de réponse attendue :
    {
        "points_forts": ["Bonne clarté du plan", "Structure logique"],
        "points_faibles": ["Peu de données quantitatives"],
        "suggestions": ["Ajouter des indicateurs de performance"],
        "note_coherence": 4,
        "commentaire_global": "Un bon plan globalement cohérent mais améliorable.",
        "erreur": false
    }
    """
    logger.info("[CritiqueRH] Analyse du contenu RH lancé.")
    llm = LLMManager().get_llm()
    parser = PydanticOutputParser(pydantic_object=CritiqueRHPlan)
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

    default_response = CritiqueRHPlan(
        points_forts=["Aucun contenu à analyser"],
        points_faibles=["Contenu vide"],
        suggestions=["Fournir du contenu à analyser"],
        note_coherence=0,
        commentaire_global="Aucun contenu fourni",
        erreur=True
    )

    content = input.get("content", "")
    if not content:
        return default_response.dict()

    prompt = f"""
Tu es un expert RH qui fait des critiques constructives. Analyse ce contenu et réponds UNIQUEMENT avec un JSON valide :

Contenu à analyser :
{content[:3000]}

Réponds STRICTEMENT avec ce format JSON (sans texte avant ou après) :
{{
    "points_forts": ["point1", "point2"],
    "points_faibles": ["point1", "point2"],
    "suggestions": ["suggestion1", "suggestion2"],
    "note_coherence": 4,
    "commentaire_global": "commentaire synthétique",
    "erreur": false
}}
"""

    try:
        response = llm.invoke(prompt)
        logger.debug(f"Réponse brute LLM : {response}")

        json_text = _extract_json_block(response)
        if not json_text:
            logger.error("Aucun bloc JSON détecté dans la réponse du LLM")
            return default_response.dict()

        # Nettoyage guillemets typographiques
        json_text = json_text.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'").strip()

        parsed = fixing_parser.parse(json_text)
        return parsed.dict()

    except Exception as e:
        logger.error(f"Erreur critique_rh_tool: {str(e)}", exc_info=True)
        return default_response.dict()
