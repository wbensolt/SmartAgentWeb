from typing import Any, Dict, Optional
from langchain_core.runnables import Runnable
from core.llm_providers import LLMManager
from pydantic import BaseModel
import json
import logging
import re

logger = logging.getLogger(__name__)

class RHResponse(BaseModel):
    """Modèle de réponse standardisé pour l'agent RH"""
    response: str
    legal_references: Dict[str, str]
    compliance_status: str
    error: Optional[bool] = False
    error_details: Optional[str] = None

class RHAgent(Runnable):
    def __init__(self, llm=None):
        self.llm = llm or LLMManager().get_llm()
        self.max_input_length = 2000
        self.max_output_length = 5000

        self.system_prompt = """[SYSTEM]
        Expert en droit du travail français - Réponses structurées
        Exige le format JSON suivant :
        {
            "response": "texte concis",
            "legal_references": {
                "code_du_travail": "article",
                "convention_collective": "IDCC"
            },
            "compliance_status": "conforme|à_verifier|non_conforme"
        }"""

    def _truncate_input(self, text: str) -> str:
        return text[:self.max_input_length]

    def _parse_json_recursive(self, content: str) -> Dict[str, Any]:
        """Essaye de parser du JSON, même s’il est encodé en string ou incomplet"""
        try:
            # Nettoyage des balises éventuelles (markdown, etc.)
            cleaned = re.sub(r"```json|```", "", content, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"'", '"', cleaned)

            # Si la chaîne contient un JSON, l'extraire
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                raise ValueError("Aucun bloc JSON détecté")

            data = json.loads(match.group(0))

            # Récursivement, parser les chaînes qui contiennent du JSON
            for key, val in list(data.items()):
                if isinstance(val, str) and val.strip().startswith("{") and val.strip().endswith("}"):
                    try:
                        data[key] = json.loads(val)
                    except:
                        pass
            return data

        except Exception as e:
            raise ValueError(f"Erreur parsing JSON brut : {str(e)}")

    def _validate_output(self, data: Dict) -> Dict:
        return {
            "response": str(data.get("response", ""))[:self.max_output_length],
            "legal_references": {
                "code_du_travail": str(data.get("legal_references", {}).get("code_du_travail", "À compléter")),
                "convention_collective": str(data.get("legal_references", {}).get("convention_collective", "À compléter"))
            },
            "compliance_status": str(data.get("compliance_status", "à_verifier")).lower()
        }

    def _save_to_file(self, data: Dict[str, Any], filename: str = "result_rh_agent.json"):
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"[RHAgent] Réponse sauvegardée dans {filename}")
        except Exception as e:
            logger.error(f"[RHAgent] Erreur lors de la sauvegarde : {str(e)}")

    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        try:
            question = self._truncate_input(str(input.get("query", "")))
            if not question.strip():
                raise ValueError("Question vide")

            prompt = f"""{self.system_prompt}

            [QUESTION]
            {question}

            [CONTEXTE]
            {json.dumps(input.get('context', {}), ensure_ascii=False)[:1000]}

            [EXIGENCES]
            - Réponse en JSON VALIDE
            - Maximum {self.max_output_length} caractères
            - Citer au moins 1 article de loi"""

            raw_response = self.llm.invoke(prompt)

            if isinstance(raw_response, str):
                response_data = self._parse_json_recursive(raw_response)
            else:
                response_data = raw_response

            validated = self._validate_output(response_data)

            final = RHResponse(**validated).dict()
            self._save_to_file(final)

            return final

        except Exception as e:
            logger.error(f"Erreur RHAgent: {str(e)}")
            fallback = RHResponse(
                response=f"Erreur: {str(e)[:200]}",
                legal_references={},
                compliance_status="non_conforme",
                error=True,
                error_details=str(e)
            ).dict()
            self._save_to_file(fallback)
            return fallback

    async def ainvoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return self.invoke(input)
