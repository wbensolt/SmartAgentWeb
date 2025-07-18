from langchain_core.tools import tool
from core.llm_providers import LLMManager
from agents.nodes.hr_agents.schema import TalentPlan
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from typing import Dict, Any
import logging
import json
import re
from datetime import datetime

logger = logging.getLogger(__name__)

@tool
def talent_manager_tool(input: str) -> Dict[str, Any]:
    """
    Analyse la faisabilité d’un plan de montée en compétences dans un contexte RH.

    Cette fonction reçoit une chaîne JSON contenant une requête décrivant un projet RH
    ainsi qu’un état incluant des données analytiques internes (budget, délais, compétences couvertes/manquantes).
    Elle interroge un LLM pour générer un plan de formation structuré sous forme JSON conforme au modèle TalentPlan.

    Args:
        input (str): JSON string avec au moins les clés :
            - "query" (str) : description textuelle du projet RH.
            - "state" (dict, optionnel) : état incluant "data_analytics" avec contexte interne.

    Returns:
        Dict[str, Any]: Dictionnaire JSON contenant :
            - upskill_possible (bool) : indique si la montée en compétences est réalisable.
            - duree_formation (str) : estimation de la durée de la formation.
            - referentiel_competences (str) : indicateur sur la disponibilité d’un référentiel.
            - plan_global (str) : description synthétique du plan de formation.
            - metadata (dict) : informations additionnelles (timestamp, budget, délai, compétences).

    Raises:
        ValueError: si la requête est vide.

    Notes:
        - La réponse est strictement un JSON conforme au schéma attendu.
        - Le contenu du prompt est tronqué à 2000 caractères maximum.
        - En cas d’erreur, un fallback minimal est retourné avec message d’erreur.
    """
    logger.info("[talent_manager_tool] Analyse du prompt projet lancé.")
    try:
        params = json.loads(input)
        query = params.get("query", "")
        state = params.get("state", {})
        data = state.get("data_analytics", {})

        if not query:
            raise ValueError("Requête vide.")

        llm = LLMManager().get_llm()
        parser = PydanticOutputParser(pydantic_object=TalentPlan)
        fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

        # Limitations d'entrée/sortie si besoin
        max_input_length = 2000

        # Construire contexte résumé
        budget = data.get("budget_moyen", "inconnu")
        delai = data.get("delai_moyen_lancement_projet", "inconnu")
        comp_couv = data.get("competences_couvertes", [])
        comp_manq = data.get("competences_manquantes", [])

        contexte_entreprise = (
            f"Budget: {budget} €, "
            f"Délai: {delai} jours, "
            f"Compétences couvertes: {', '.join([c.get('competence', '') for c in comp_couv]) or 'Aucune'}, "
            f"Compétences manquantes: {', '.join(comp_manq) or 'Aucune'}"
        )

        exemple = json.dumps({
            "upskill_possible": True,
            "duree_formation": "X semaines",
            "referentiel_competences": "oui",
            "plan_global": "Plan de formation détaillé ici"
        }, indent=2, ensure_ascii=False)

        prompt = f"""
Tu es un expert RH spécialisé dans la montée en compétences.

Contexte du projet:
{query[:max_input_length]}

Contexte entreprise:
{contexte_entreprise}

Analyse:
Fournis un JSON strictement conforme au modèle suivant :
{exemple}

Réponse:
""".strip()

        raw_response = llm.invoke(prompt)
        raw_response = getattr(raw_response, "content", raw_response)

        # Nettoyage JSON dans réponse (extrait bloc ```json``` s'il existe)
        def clean_json_response(text: str) -> str:
            match = re.search(r"```json(.*?)```", text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
            return text.strip()

        cleaned = clean_json_response(raw_response)

        # Parsing avec fixing parser
        parsed = fixing_parser.parse(cleaned)
        result = parsed.dict()

        # Ajout metadata (optionnel)
        result["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "budget": budget,
            "delai": delai,
            "competences_manquantes": comp_manq,
            "competences_couvertes": comp_couv,
        }

        return result

    except Exception as e:
        logger.error(f"[talent_manager_tool] Erreur TalentManagerAgent : {str(e)}", exc_info=True)
        # Fallback minimal
        return {
            "upskill_possible": False,
            "duree_formation": "Non estimable",
            "referentiel_competences": "non",
            "plan_global": f"Erreur: {str(e)}",
            "metadata": {"error": str(e)}
        }
