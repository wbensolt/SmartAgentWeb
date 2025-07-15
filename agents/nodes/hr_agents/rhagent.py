from typing import Any, Dict
from langchain_core.runnables import Runnable
from agents.nodes.hr_agents.schema import RHResponse
from core.llm_providers import LLMManager
from pydantic import BaseModel
import json
import logging
import re
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from utils.code_travail_loader import CodeTravailLoader

logger = logging.getLogger(__name__)

class RHAgent(Runnable):
    def __init__(self, llm=None):
        self.llm = llm or LLMManager().get_llm()
        self.parser = PydanticOutputParser(pydantic_object=RHResponse)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
        self.max_input_length = 2000
        self.max_output_length = 5000
        self.code_loader = CodeTravailLoader("data/code_du_travail.json")

        self.system_prompt = """[SYSTEM]
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

    def _truncate_input(self, text: str) -> str:
        return text[:self.max_input_length]

    def _clean_response(self, response: str) -> str:
        match = re.search(r"```json(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return response.strip().replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')

    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        try:
            question = self._truncate_input(str(input.get("query", "")))
            print(f"🟡 Étape 1 : Requête utilisateur : {question}")
            if not question.strip():
                raise ValueError("Question vide")

            # Recherche d'articles
            keywords = question.lower().split()
            articles = self.code_loader.search_by_keywords(keywords, max_results=3)

            if articles:
                print(f"🟡 Étape 2 : Articles trouvés : {[a['id'] for a in articles]}")
                extrait_code = "\n\n".join(
                    f"Article {a['id']} - {a['title']}:\n{a['content'][:500]}..."
                    for a in articles
                )
                articles_ids = [a['id'] for a in articles]
            else:
                print("🟡 Étape 2 : Aucun article pertinent trouvé.")
                extrait_code = "Aucun article pertinent trouvé."
                articles_ids = []

            # Contexte entreprise
            state = input.get("state", {})
            data_analytics = state.get("data_analytics", {})
            budget_moyen = data_analytics.get("budget_moyen", "inconnu")
            delai_moyen = data_analytics.get("delai_moyen_lancement_projet", "inconnu")
            competences = data_analytics.get("capacites_disponibles", [])
            competences_str = ", ".join(c.get("competence", "") for c in competences)

            contexte = {
                "entreprise": {
                    "budget_moyen": budget_moyen,
                    "delai_moyen_lancement_projet": delai_moyen,
                    "competences_internes": competences_str
                },
                "contexte_externe": input.get("context", {})
            }
            contexte_json = json.dumps(contexte, ensure_ascii=False)[:1000]

            print("🟡 Étape 3 : Préparation du prompt pour le LLM...")
            prompt = (
                f"{self.system_prompt}\n\n"
                f"[QUESTION]\n{question}\n\n"
                f"[CONTEXTE INTERNE]\n{contexte_json}\n\n"
                f"[EXTRAITS CODE DU TRAVAIL]\n{extrait_code}\n\n"
                f"[EXIGENCES]\n"
                f"- Vérifie si le projet respecte les obligations légales.\n"
                f"- Signale si le budget ou le temps est insuffisant pour respecter la loi.\n"
                f"- Donne une réponse claire et juridiquement exploitable.\n"
                f"- Maximum {self.max_output_length} caractères."
            )

            raw_response = self.llm.invoke(prompt)
            print("🟢 Étape 4 : Réponse brute reçue du LLM")

            cleaned_response = self._clean_response(raw_response)
            print(f"\n🧩 JSON nettoyé avant parsing :\n{cleaned_response}")

            # 🔧 Correction JSON si nécessaire
            try:
                json_obj = json.loads(cleaned_response)

                # Si legal_references n'est pas un dictionnaire, on force un format correct
                if not isinstance(json_obj.get("legal_references"), dict):
                    json_obj["legal_references"] = {
                        "code_du_travail": [],
                        "convention_collective": ""
                    }

                if isinstance(json_obj["legal_references"].get("code_du_travail"), dict):
                    json_obj["legal_references"]["code_du_travail"] = list(json_obj["legal_references"]["code_du_travail"].keys())
                elif json_obj["legal_references"].get("code_du_travail") is None:
                    json_obj["legal_references"]["code_du_travail"] = []

                # Correction : convention_collective = None -> ""
                if json_obj["legal_references"].get("convention_collective") is None:
                    json_obj["legal_references"]["convention_collective"] = ""

                cleaned_response = json.dumps(json_obj)

            except Exception as fix_err:
                print(f"⚠️ Erreur correction JSON RHAgent : {fix_err}")


            parsed = self.fixing_parser.parse(cleaned_response)
            print("✅ Étape 5 : Résultat final prêt.")
            result = parsed.dict()

            if "legal_references" in result:
                result["legal_references"]["code_du_travail"] = articles_ids

            return result

        except Exception as e:
            print(f"❌ Erreur RHAgent: {str(e)}")
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
