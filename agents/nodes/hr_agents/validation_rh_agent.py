from typing import Dict, Any, TypedDict
from langchain_core.runnables import Runnable
from core.llm_providers import LLMManager
import json
import logging

class ValidationResponse(TypedDict):
    validation: str
    justification: str

class ValidationRHAgent(Runnable):
    def __init__(self, llm=None):
        self.llm = LLMManager().get_llm()
        self.logger = logging.getLogger(__name__)
        self._default_response = {
            "validation": "non valide",
            "justification": "Erreur de traitement",
            "erreur": True
        }

    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        try:
            critique = str(input.get("critique", ""))[:5000]  # Limite la taille
            if not critique.strip():
                return self._error_response("Critique vide")

            prompt = f"""Validation RH - Format JSON strict

Critique à évaluer:
{critique}

Répondre UNIQUEMENT avec ce format:
{{
    "validation": "valide|non valide",
    "justification": "3-5 mots maximum"
}}""".format(critique=critique[:2000])

            response = self.llm.invoke(prompt)
            return self._parse_response(response)
            
        except Exception as e:
            self.logger.error(f"Erreur ValidationRHAgent: {str(e)}")
            return self._error_response(str(e))

    def _parse_response(self, response: Any) -> Dict[str, Any]:
        """Parse et valide la réponse du LLM"""
        try:
            if isinstance(response, dict):
                data = response
            else:
                data = json.loads(str(response).replace('```json', '').replace('```', '').strip())

            # Validation des valeurs
            validation = str(data.get("validation", "")).lower()
            if validation not in ("valide", "non valide"):
                validation = "non valide"

            return {
                "validation": validation,
                "justification": str(data.get("justification", ""))[:50],
                "erreur": False
            }
        except Exception as e:
            self.logger.warning(f"Réponse invalide: {str(response)[:200]}")
            return self._error_response(f"Format invalide: {str(e)}")

    def _error_response(self, error_msg: str) -> Dict[str, Any]:
        """Réponse d'erreur standardisée"""
        return {
            "validation": "erreur",
            "justification": error_msg[:100],
            "erreur": True
        }

    async def ainvoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return self.invoke(input)