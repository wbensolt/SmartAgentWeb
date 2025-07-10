from langchain_core.runnables import Runnable
from core.llm_providers import LLMManager
from typing import Dict, Any
import json
import logging
import re
import os

class FinalRHAgent(Runnable):
    def __init__(self, llm=None):
        self.llm = llm or LLMManager().get_llm()
        self.logger = logging.getLogger(__name__)

    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        try:
            answers = input.get("answers", {})
            critiques = input.get("critiques", {})
            validations = input.get("validations", {})

            answers_str = json.dumps(answers, ensure_ascii=False)[:1500]
            critiques_str = json.dumps(critiques, ensure_ascii=False)[:1000]
            validations_str = json.dumps(validations, ensure_ascii=False)[:1000]

            prompt = f"""
            [SYSTEM]
            Tu es un expert en synthèse RH stratégique.

            À partir des réponses de plusieurs agents spécialisés, tu dois donner une décision métier claire sur la faisabilité d’un projet RH.

            Réponds uniquement avec un JSON VALIDE :
            {
            "faisabilite": "Oui|Non|Partiel",
            "conditions_reussite": ["condition 1", "condition 2"],
            "score_confiance": 0.0 à 1.0,
            "recommandation": "Phrase synthétique et exploitable",
            "risques_principaux": ["risque 1", "risque 2"]
            }
            - Ne jamais inclure de texte hors JSON
            - Corriger les erreurs des autres agents si besoin (ex: parsing)
            """
            response = self.llm.invoke(prompt)

            self.logger.debug(f"[FinalRHAgent] Réponse brute LLM : {repr(response)}")
            print(f"[DEBUG] Réponse brute LLM : {repr(response)}")  # à retirer en prod

            result = self._format_final_response(response)

            # Sauvegarde du résultat dans un fichier
            self._save_to_json(result, filename="result_final_rh.json")

            return result

        except Exception as e:
            self.logger.error(f"[FinalRHAgent] Erreur invoke : {str(e)}")
            return self._error_response(str(e))

    def _format_final_response(self, response: str) -> Dict[str, Any]:
        try:
            cleaned = response.strip()
            cleaned = re.sub(r"```json|```", "", cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"'", '"', cleaned)

            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                raise ValueError("Aucun bloc JSON détecté dans la réponse")

            json_str = match.group(0)
            data = self._parse_json_recursive(json_str)

            faisabilite = data.get("faisabilite", "Indéterminé")
            if faisabilite not in ["Oui", "Non", "Partiel"]:
                faisabilite = "Indéterminé"

            return {
                "faisabilite": faisabilite,
                "conditions_reussite": [str(c)[:100] for c in data.get("conditions_reussite", [])][:5],
                "score_confiance": round(min(max(float(data.get("score_confiance", 0.0)), 0.0), 1.0), 2),
                "recommandation": str(data.get("recommandation", ""))[:500],
                "risques_principaux": [str(r)[:100] for r in data.get("risques_principaux", [])][:5],
            }

        except Exception as e:
            self.logger.warning(f"[FinalRHAgent] Erreur parsing JSON : {str(e)}\nRéponse brute : {repr(response)}")
            return self._error_response(f"Erreur parsing JSON : {str(e)}")

    def _parse_json_recursive(self, content: str) -> Dict[str, Any]:
        """
        Tente de parser un JSON potentiellement doublement encapsulé.
        """
        try:
            data = json.loads(content)
            # Vérifie récursivement les chaînes qui contiennent elles-mêmes du JSON
            for key, value in list(data.items()):
                if isinstance(value, str) and value.strip().startswith("{") and value.strip().endswith("}"):
                    try:
                        sub_data = json.loads(value)
                        data[key] = sub_data
                    except Exception:
                        pass  # Pas un JSON valide, on laisse tel quel
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Erreur JSON récursif : {str(e)}")

    def _save_to_json(self, data: Dict[str, Any], filename: str):
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"[FinalRHAgent] Résultat sauvegardé dans {filename}")
        except Exception as e:
            self.logger.error(f"[FinalRHAgent] Erreur de sauvegarde fichier : {str(e)}")

    def _error_response(self, error: str) -> Dict[str, Any]:
        return {
            "faisabilite": "Erreur",
            "conditions_reussite": ["Vérifier les logs système"],
            "score_confiance": 0.0,
            "recommandation": f"Erreur : {error[:200]}",
            "risques_principaux": ["Erreur technique dans l'analyse"]
        }
