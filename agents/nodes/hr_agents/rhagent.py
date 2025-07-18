from langchain_core.tools import tool
from core.llm_providers import LLMManager
from agents.nodes.hr_agents.schema import RHResponse
from utils.code_travail_loader import CodeTravailLoader
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser

from typing import Dict, Any
import logging
import json
import re

logger = logging.getLogger(__name__)

@tool
def check_labor_law(input: str) -> Dict[str, Any]:
    """
    Vérifie la conformité d'un projet RH avec le droit du travail français.

    Cette fonction prend en entrée une chaîne JSON contenant une requête textuelle décrivant un projet RH
    ainsi que des données contextuelles internes (budget, délais, compétences, etc.). Elle utilise un modèle
    de langage pour analyser la conformité du projet aux règles du Code du travail et retourne un JSON
    structuré avec une réponse claire, les références légales pertinentes et un statut de conformité.

    Args:
        input (str): Chaîne JSON contenant au moins les clés suivantes :
            - "query" (str) : description du projet RH à analyser.
            - "data" (dict, optionnel) : données internes sur le projet (budget, délais, compétences).

    Returns:
        Dict[str, Any]: Dictionnaire JSON conforme au schéma RHResponse avec les champs :
            - "response" (str) : synthèse actionnable sur la conformité.
            - "legal_references" (dict) : références aux articles du Code du travail et conventions collectives.
            - "compliance_status" (str) : statut ("conforme", "à_verifier", "non_conforme").
            - Eventuellement champs d'erreur si problème d'analyse.

    Raises:
        ValueError: si la requête est vide.

    Notes:
        - La réponse est strictement un bloc JSON sans texte additionnel.
        - Les références légales sont limitées aux extraits pertinents extraits via mots-clés.
        - En cas d'erreur, un objet avec statut "non_conforme" et message d'erreur est retourné.
    """
    logger.info("[check_labor_law] Analyse du prompt projet lancé.")
    try:
        params = json.loads(input)
        query = params.get("query", "")
        data = params.get("data", {})

        if not query:
            raise ValueError("Requête vide.")

        llm = LLMManager().get_llm()
        parser = PydanticOutputParser(pydantic_object=RHResponse)
        fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
        max_input_length = 2000
        max_output_length = 5000
        code_loader = CodeTravailLoader("data/code_du_travail.json")

        system_prompt = """[SYSTEM]
Tu es un expert en droit du travail français. Ta tâche est de vérifier si le projet décrit respecte le droit du travail.
Tu DOIS répondre uniquement avec un **bloc JSON** VALIDE (aucun texte autour), comme ceci :
```json
{
"response": "Réponse concise et actionnable indiquant si le projet RH est conforme. Mentionne les risques légaux si budget ou délai posent problème.",
"legal_references": {
    "code_du_travail": ["L.6321-1", "L.4121-1"],
    "convention_collective": "IDCC 2098"
},
"compliance_status": "conforme|à_verifier|non_conforme"
}
Utilise uniquement les articles fournis dans la section [EXTRAITS CODE DU TRAVAIL].
Si aucun article n’est pertinent, indique "code_du_travail": [] explicitement.
Prends en compte le budget, les délais, les effectifs, et les compétences internes.
Ne rajoute aucune explication ni texte en dehors du bloc markdown JSON."""

        question = query.strip()[:max_input_length]
        keywords = question.lower().split()
        articles = code_loader.search_by_keywords(keywords, max_results=3)

        extrait_code = (
            "\n\n".join(f"Article {a['id']} - {a['title']}:\n{a['content'][:500]}..." for a in articles)
            if articles else "Aucun article pertinent trouvé."
        )
        articles_ids = [a["id"] for a in articles] if articles else []

        budget_moyen = data.get("budget_moyen", "inconnu")
        delai_moyen = data.get("delai_moyen_lancement_projet", "inconnu")
        competences = data.get("capacites_disponibles", [])
        competences_str = ", ".join(c.get("competence", "") for c in competences)

        contexte = {
            "entreprise": {
                "budget_moyen": budget_moyen,
                "delai_moyen_lancement_projet": delai_moyen,
                "competences_internes": competences_str
            },
            "contexte_externe": data.get("context", {})
        }
        contexte_json = json.dumps(contexte, ensure_ascii=False)[:1000]

        prompt = (
            f"{system_prompt}\n\n"
            f"[QUESTION]\n{question}\n\n"
            f"[CONTEXTE INTERNE]\n{contexte_json}\n\n"
            f"[EXTRAITS CODE DU TRAVAIL]\n{extrait_code}\n\n"
            f"[EXIGENCES]\n"
            f"- Vérifie si le projet respecte les obligations légales.\n"
            f"- Signale si le budget ou le temps est insuffisant pour respecter la loi.\n"
            f"- Donne une réponse claire et juridiquement exploitable.\n"
            f"- Maximum {max_output_length} caractères."
        )

        raw_response = llm.invoke(prompt)
        # Correction ici : extraire le texte si raw_response est un AIMessage
        raw_response = getattr(raw_response, "content", raw_response)

        def clean_json_response(text: str) -> str:
            match = re.search(r"```json(.*?)```", text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
            return text.strip()

        cleaned = clean_json_response(raw_response)
        json_obj = json.loads(cleaned)

        # Correction références légales
        legal_refs = json_obj.get("legal_references", {})
        if not isinstance(legal_refs, dict):
            json_obj["legal_references"] = {"code_du_travail": [], "convention_collective": ""}
        if legal_refs.get("code_du_travail") is None:
            json_obj["legal_references"]["code_du_travail"] = []
        if legal_refs.get("convention_collective") is None:
            json_obj["legal_references"]["convention_collective"] = ""

        result = fixing_parser.parse(json.dumps(json_obj)).dict()
        result["legal_references"]["code_du_travail"] = articles_ids

        return result

    except Exception as e:
        logger.error(f"[check_labor_law] Erreur RHAgent : {str(e)}", exc_info=True)
        fallback = RHResponse(
            response=f"Erreur: {str(e)[:200]}",
            legal_references={},
            compliance_status="non_conforme",
            error=True,
            error_details=str(e)
        )
        return fallback.dict()
