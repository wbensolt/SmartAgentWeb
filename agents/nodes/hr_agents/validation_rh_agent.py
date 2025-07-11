from typing import Dict, Any
from langchain_core.runnables import Runnable
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from core.llm_providers import LLMManager
from agents.nodes.hr_agents.schema import ValidationRHPlan
import logging

class ValidationRHAgent(Runnable):
    def __init__(self, llm=None):
        self.llm = llm or LLMManager().get_llm()
        self.parser = PydanticOutputParser(pydantic_object=ValidationRHPlan)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
        self.logger = logging.getLogger(__name__)
        self._default_response = ValidationRHPlan(
            validation="non valide",
            justification="Erreur de traitement"
        )

    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
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
            response = self.llm.invoke(prompt)
            self.logger.debug(f"Réponse brute LLM : {response}")

            # Nettoyage simple
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            # Parsing et correction automatique avec Pydantic
            parsed = self.fixing_parser.parse(cleaned_response)
            return parsed.dict()

        except Exception as e:
            self.logger.error(f"Erreur ValidationRHAgent: {str(e)}", exc_info=True)
            return self._default_response.dict()

    async def ainvoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return self.invoke(input)
