from typing import Dict, Any
from langchain_core.runnables import Runnable
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from core.llm_providers import LLMManager
from agents.nodes.hr_agents.schema import FinalRHPlan
import logging
import json

class FinalRHAgent(Runnable):
    def __init__(self, llm=None):
        self.llm = llm or LLMManager().get_llm()
        self.parser = PydanticOutputParser(pydantic_object=FinalRHPlan)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
        self.logger = logging.getLogger(__name__)
        self._default_response = FinalRHPlan(
            faisabilite="Non",
            conditions_reussite=["Vérifier les logs système"],
            score_confiance=0.0,
            recommandation="Erreur technique dans l'analyse",
            risques_principaux=["Erreur technique dans l'analyse"]
        )

    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        try:
            answers = input.get("answers", {})
            critiques = input.get("critiques", {})
            validations = input.get("validations", {})

            # Limiter la taille pour éviter prompt trop long
            answers_str = json.dumps(answers, ensure_ascii=False)[:1500]
            critiques_str = json.dumps(critiques, ensure_ascii=False)[:1000]
            validations_str = json.dumps(validations, ensure_ascii=False)[:1000]

            prompt = f"""
[SYSTEM]
Tu es un expert en synthèse RH stratégique.

À partir des réponses de plusieurs agents spécialisés, tu dois donner une décision métier claire sur la faisabilité d’un projet RH.

Réponds uniquement avec un JSON VALIDE au format strict :
{{
  "faisabilite": "Oui|Non|Partiel",
  "conditions_reussite": ["condition 1", "condition 2"],
  "score_confiance": 0.0 à 1.0,
  "recommandation": "Phrase synthétique et exploitable",
  "risques_principaux": ["risque 1", "risque 2"]
}}

Réponses des agents spécialisés :
- Answers: {answers_str}
- Critiques: {critiques_str}
- Validations: {validations_str}

Ne jamais inclure de texte hors JSON.
Corrige les erreurs de format si besoin.
"""

            raw_response = self.llm.invoke(prompt)
            self.logger.debug(f"[FinalRHAgent] Réponse brute LLM : {repr(raw_response)}")

            # Nettoyage simple
            cleaned_response = raw_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            # Parsing + correction auto
            parsed = self.fixing_parser.parse(cleaned_response)

            # Sauvegarde optionnelle, à adapter selon besoin
            # self._save_to_json(parsed.dict(), filename="result_final_rh.json")

            return parsed.dict()

        except Exception as e:
            self.logger.error(f"[FinalRHAgent] Erreur invoke : {str(e)}", exc_info=True)
            return self._default_response.dict()

    # Si tu veux garder la méthode de sauvegarde
    # def _save_to_json(self, data: Dict[str, Any], filename: str):
    #     try:
    #         with open(filename, "w", encoding="utf-8") as f:
    #             json.dump(data, f, ensure_ascii=False, indent=2)
    #         self.logger.info(f"[FinalRHAgent] Résultat sauvegardé dans {filename}")
    #     except Exception as e:
    #         self.logger.error(f"[FinalRHAgent] Erreur de sauvegarde fichier : {str(e)}")

    async def ainvoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return self.invoke(input)
