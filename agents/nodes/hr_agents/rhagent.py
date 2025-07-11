from typing import Any, Dict, Optional
from langchain_core.runnables import Runnable
from agents.nodes.hr_agents.schema import RHResponse
from core.llm_providers import LLMManager
from pydantic import BaseModel
import json
import logging
import re
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser

logger = logging.getLogger(__name__)

class RHAgent(Runnable):
    def __init__(self, llm=None):
        self.llm = llm or LLMManager().get_llm()
        self.parser = PydanticOutputParser(pydantic_object=RHResponse)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
        self.max_input_length = 2000
        self.max_output_length = 5000

        # Prompt système renforcé pour exiger un bloc JSON strict dans un bloc markdown
        self.system_prompt = """[SYSTEM]
        Expert en droit du travail français - Réponses structurées
        Tu DOIS répondre uniquement avec un bloc markdown JSON, sans aucun texte additionnel :
        ```json
        {
            "response": "texte concis",
            "legal_references": {
                "code_du_travail": "article",
                "convention_collective": "IDCC"
            },
            "compliance_status": "conforme|à_verifier|non_conforme"
        }
        Ne rajoute aucune explication ni texte en dehors du bloc markdown JSON."""

    def _truncate_input(self, text: str) -> str:
        return text[:self.max_input_length]

    def _clean_response(self, response: str) -> str:
        # Extraction stricte du bloc JSON dans un markdown ```json ... ```
        match = re.search(r"```json(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Sinon nettoyage simple
        return response.strip().replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')

    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        try:
            question = self._truncate_input(str(input.get("query", "")))
            if not question.strip():
                raise ValueError("Question vide")

            context = json.dumps(input.get("context", {}), ensure_ascii=False)[:1000]

            prompt = (
                f"{self.system_prompt}\n\n[QUESTION]\n{question}\n\n[CONTEXTE]\n{context}\n\n"
                f"[EXIGENCES]\n- Réponse en JSON VALIDE\n- Maximum {self.max_output_length} caractères\n- Citer au moins 1 article de loi"
            )

            raw_response = self.llm.invoke(prompt)
            logger.info(f"[RHAgent] Réponse brute LLM : {raw_response}")

            cleaned_response = self._clean_response(raw_response)
            parsed = self.fixing_parser.parse(cleaned_response)
            result = parsed.dict()

            # Sauvegarde optionnelle
            # with open("result_rh_agent.json", "w", encoding="utf-8") as f:
            #     json.dump(result, f, ensure_ascii=False, indent=2)

            return result

        except Exception as e:
            logger.error(f"Erreur RHAgent: {str(e)}", exc_info=True)
            fallback = RHResponse(
                response=f"Erreur: {str(e)[:200]}",
                legal_references={},
                compliance_status="non_conforme",
                error=True,
                error_details=str(e)
            )
            return fallback.dict()

    async def ainvoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return self.invoke(input)
