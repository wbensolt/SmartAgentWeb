from langchain_core.runnables import Runnable
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from core.llm_providers import LLMManager
from agents.nodes.hr_agents.schema import CritiqueRHPlan 
from typing import Dict, Any
import logging

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
            commentaire_global="Aucun contenu fourni"
        )

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
    "commentaire_global": "commentaire synthétique"
}}
"""

        try:
            response = self.llm.invoke(prompt)
            self.logger.debug(f"Réponse brute LLM : {response}")

            # Nettoyage simple
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            # Parsing avec correction automatique
            parsed = self.fixing_parser.parse(cleaned_response)
            return parsed.dict()

        except Exception as e:
            self.logger.error(f"Erreur CritiqueRHAgent: {str(e)}", exc_info=True)
            return self._default_response.dict()

    async def ainvoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return self.invoke(input)
