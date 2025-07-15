from langchain_core.runnables import Runnable
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from core.llm_providers import LLMManager
from agents.nodes.hr_agents.schema import CritiqueRHPlan
from typing import Dict, Any, Optional
import logging
import re
import json


class CritiqueRHAgent(Runnable):
    def __init__(self, llm=None):
        self.llm = llm or LLMManager().get_llm()
        self.parser = PydanticOutputParser(pydantic_object=CritiqueRHPlan)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
        self.logger = logging.getLogger(__name__)
        self._default_response = CritiqueRHPlan(
            points_forts=["Aucun contenu à analyser"],
            points_faibles=["Contenu vide"],
            suggestions=["Fournir du contenu à analyser"],
            note_coherence=0,
            commentaire_global="Aucun contenu fourni",
            erreur=True
        )

    def _extract_json_block(self, text: str) -> Optional[str]:
        """
        Tente d'extraire un bloc JSON valide dans la réponse brute du LLM.
        Supporte la présence de ```json ... ``` ou juste un JSON brut.
        """
        # Extraction d'un bloc ```json ... ```
        match = re.search(r"```json(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Extraction d'un bloc ``` ... ```
        match = re.search(r"```(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Sinon, tenter d'extraire un JSON brut en cherchant l'accolade ouvrante
        json_start = text.find("{")
        if json_start != -1:
            json_text = text[json_start:]
            return json_text.strip()

        # Pas de JSON détecté
        return None

    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        content = input.get("content", "")
        if not content:
            return self._default_response.dict()

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
            response = self.llm.invoke(prompt)
            self.logger.debug(f"Réponse brute LLM : {response}")

            # Extraction propre du JSON
            json_text = self._extract_json_block(response)
            if not json_text:
                self.logger.error("Aucun bloc JSON détecté dans la réponse du LLM")
                return self._default_response.dict()

            # Nettoyage simple des guillemets typographiques
            json_text = json_text.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'").strip()

            # Parsing avec correction automatique
            parsed = self.fixing_parser.parse(json_text)
            return parsed.dict()

        except Exception as e:
            self.logger.error(f"Erreur CritiqueRHAgent: {str(e)}", exc_info=True)
            return self._default_response.dict()

    async def ainvoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        # version async (non modifiée ici pour simplicité)
        return self.invoke(input)
