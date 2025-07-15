import json
import logging
import re
import sqlite3
import unicodedata
import string
from typing import Any, Dict, List, Optional
from langchain_core.runnables import Runnable
from core.llm_providers import LLMManager
from agents.nodes.hr_agents.schema import DataAnalystInsight
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser


def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def extraire_contexte_societe(sqlite_path: str) -> Dict[str, Any]:
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()
    result = {}
    try:
        cursor.execute("SELECT AVG(budget_disponible), AVG(duree_prevue_jours) FROM projets")
        avg_budget, avg_duree = cursor.fetchone()
        result["budget_moyen"] = round(avg_budget or 0, 2)
        result["delai_moyen_lancement_projet"] = int(avg_duree or 0)

        cursor.execute("""
            SELECT c.nom, COUNT(ec.employe_id)
            FROM employes_competences ec
            JOIN competences c ON ec.competence_id = c.id
            GROUP BY c.nom
        """)
        result["capacites_disponibles"] = [
            {"competence": row[0], "effectif": row[1]} for row in cursor.fetchall()
        ]
    except Exception as e:
        print(f"[extraire_contexte_societe] Erreur SQLite : {str(e)}")
        result = {
            "budget_moyen": 0,
            "delai_moyen_lancement_projet": 0,
            "capacites_disponibles": [],
        }
    finally:
        conn.close()
    return result


class DataAnalystAgent(Runnable):
    def __init__(self, retriever, llm=None, db_path: str = "data/northwind_company.db", villes_connues: Optional[List[str]] = None):
        self.retriever = retriever
        self.llm = llm or LLMManager().get_llm()
        self.db_path = db_path
        self.villes_connues = villes_connues or ["toulouse", "paris", "lyon", "marseille", "lille", "bordeaux"]
        self.seuil_effectif_competence = 5
        self.parser = PydanticOutputParser(pydantic_object=DataAnalystInsight)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)

        # Variables pour mémoriser la dernière localisation, budget, délai
        self.last_localisation = None
        self.last_budget = None
        self.last_delai = None

    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        try:
            query = str(input.get("query", ""))
            print(f"\n[DataAnalystAgent] Requête reçue : {query}")

            docs = self.retriever.invoke(query)
            context = "\n".join(d.page_content for d in docs[:3])
            print(f"[DataAnalystAgent] Contexte externe extrait (trunc): {context[:200]}")

            societe_data = extraire_contexte_societe(self.db_path)
            print(f"[DataAnalystAgent] Données internes entreprise : budget moyen={societe_data['budget_moyen']}, délai moyen={societe_data['delai_moyen_lancement_projet']}")

            # Extraction localisation
            localisation = self.extraire_localisation(query)
            if localisation == "Non spécifiée" and self.last_localisation is not None:
                print(f"[DataAnalystAgent] Localisation non spécifiée dans la requête, utilisation de la dernière localisation mémorisée : {self.last_localisation}")
                localisation = self.last_localisation
            else:
                self.last_localisation = localisation
            print(f"[DataAnalystAgent] Localisation extraite : {localisation}")

            # Extraction budget et délai
            budget_delai = self.extraire_budget_delai(query)
            budget_user_val = budget_delai.get("budget")
            delai_user_val = budget_delai.get("delai")

            if budget_user_val is None and self.last_budget is not None:
                print(f"[DataAnalystAgent] Budget non spécifié dans la requête, utilisation du dernier budget mémorisé : {self.last_budget}")
                budget_user_val = self.last_budget
            else:
                self.last_budget = budget_user_val

            if delai_user_val is None and self.last_delai is not None:
                print(f"[DataAnalystAgent] Délai non spécifié dans la requête, utilisation du dernier délai mémorisé : {self.last_delai}")
                delai_user_val = self.last_delai
            else:
                self.last_delai = delai_user_val

            budget_user_val = budget_user_val or 0
            delai_user_val = delai_user_val or 0

            print(f"[DataAnalystAgent] Budget utilisateur extrait : {budget_user_val}")
            print(f"[DataAnalystAgent] Délai utilisateur extrait : {delai_user_val}")

            # Extraction compétences
            competences_visees = self.extraire_competences_via_llm(query)
            print(f"[DataAnalystAgent] Compétences visées détectées : {competences_visees}")

            competences_internes = {c["competence"].lower(): c["effectif"] for c in societe_data["capacites_disponibles"]}
            competences_couvertes, competences_manquantes = [], []
            for c in competences_visees:
                if competences_internes.get(c.lower(), 0) >= self.seuil_effectif_competence:
                    competences_couvertes.append({"competence": c, "effectif": competences_internes[c.lower()]})
                else:
                    competences_manquantes.append(c)

            warnings = []
            if budget_user_val and budget_user_val < societe_data["budget_moyen"] * 0.5:
                warnings.append(f"Budget très faible ({budget_user_val}€) comparé au budget moyen ({societe_data['budget_moyen']}€)")
            if delai_user_val and delai_user_val < societe_data["delai_moyen_lancement_projet"] * 0.5:
                warnings.append(f"Délai très court ({delai_user_val}j) comparé au délai moyen ({societe_data['delai_moyen_lancement_projet']}j)")
            if competences_manquantes:
                warnings.append(f"Compétences manquantes : {', '.join(competences_manquantes)}")

            prompt = f"""
Tu es un expert en stratégie RH et analyse interne.
Voici le contexte externe :
{context}
Et les données internes de l'entreprise :
- Budget moyen : {societe_data['budget_moyen']} €
- Délai moyen : {societe_data['delai_moyen_lancement_projet']} jours
- Compétences internes : {json.dumps(competences_couvertes, ensure_ascii=False, indent=2)}
Requête :
{query}
NE fournis rien d’autre qu’un JSON VALIDE avec EXACTEMENT les champs suivants :
{{
  "bassin_emploi": "",
  "disponibilite_profils": "",
  "tendances_marche": [""],
  "budget_moyen": 0,
  "delai_moyen_lancement_projet": 0,
  "capacites_disponibles": [{{"competence": "", "effectif": 0}}],
  "erreur": null
}}
"""
            raw_response = self.llm.invoke(prompt)
            print(f"[DataAnalystAgent] Réponse brute LLM (trunc): {raw_response[:300]}")

            cleaned = self._extract_json_block(raw_response)
            try:
                parsed = self.fixing_parser.parse(cleaned)
                result = parsed.dict()
            except Exception as e:
                print(f"[DataAnalystAgent] Erreur parsing JSON LLM : {e}")
                result = {
                    "bassin_emploi": "Erreur parsing",
                    "disponibilite_profils": "",
                    "tendances_marche": [],
                    "budget_moyen": societe_data.get("budget_moyen", 0),
                    "delai_moyen_lancement_projet": societe_data.get("delai_moyen_lancement_projet", 0),
                    "capacites_disponibles": societe_data.get("capacites_disponibles", []),
                    "erreur": str(e)
                }

            if warnings:
                result["warnings"] = warnings

            # Patch : extraire compétences manquantes des warnings si pas présent
            if not competences_manquantes and "warnings" in result:
                for w in result["warnings"]:
                    if "Compétences manquantes" in w:
                        texte = w.split(":")[-1]
                        competences_manquantes = [c.strip() for c in texte.split(",") if c.strip()]
                        result["competences_manquantes"] = competences_manquantes
                        break

            result["competences_vises"] = competences_visees
            result["competences_couvertes"] = competences_couvertes
            result["competences_manquantes"] = competences_manquantes
            result["localisation"] = localisation
            result["budget_utilisateur"] = budget_user_val
            result["delai_utilisateur_jours"] = delai_user_val

            print(f"[DataAnalystAgent] Résultat final : localisation={result['localisation']}, budget_utilisateur={result['budget_utilisateur']}, delai_utilisateur_jours={result['delai_utilisateur_jours']}")
            return result

        except Exception as e:
            print(f"[DataAnalystAgent] Exception générale : {str(e)}")
            return DataAnalystInsight(
                bassin_emploi="Erreur d'analyse",
                disponibilite_profils="",
                tendances_marche=[],
                budget_moyen=0,
                delai_moyen_lancement_projet=0,
                capacites_disponibles=[],
                erreur=str(e)
            ).dict()

    def _extract_json_block(self, text: str) -> str:
        try:
            start = text.index('{')
            end = text.rindex('}') + 1
            return text[start:end]
        except ValueError:
            return text.strip()

    def extraire_localisation(self, query: str) -> str:
        query_norm = normalize_text(query)
        print(f"[extraire_localisation] Texte normalisé : {query_norm}")
        for ville in self.villes_connues:
            ville_norm = normalize_text(ville)
            print(f"[extraire_localisation] Compare avec ville normalisée : {ville_norm}")
            if ville_norm in query_norm:
                print(f"[extraire_localisation] Localisation trouvée : {ville.capitalize()}")
                return ville.capitalize()
        print("[extraire_localisation] Localisation non spécifiée")
        return "Non spécifiée"

    def extraire_budget_delai(self, query: str) -> Dict[str, Any]:
        budget = None
        delai = None
        query_norm = normalize_text(query)
        print(f"[extraire_budget_delai] Texte normalisé : {query_norm}")

        match_budget = re.search(r"(\d+)[\s]*euros?", query_norm)
        if match_budget:
            print(f"[extraire_budget_delai] Budget capturé : {match_budget.group(0)}")
            try:
                budget_str = match_budget.group(1).replace(" ", "").replace(".", "").replace(",", ".")
                budget = float(budget_str)
                print(f"[extraire_budget_delai] Budget converti : {budget}")
            except Exception as e:
                print(f"[extraire_budget_delai] Erreur conversion budget: {e}")

        match_delai = re.search(r"(\d+)[\s]*(mois|jours?)", query_norm)
        if match_delai:
            val, unite = int(match_delai.group(1)), match_delai.group(2)
            print(f"[extraire_budget_delai] Délai capturé : {val} {unite}")
            delai = val * 30 if "mois" in unite else val
            print(f"[extraire_budget_delai] Délai converti en jours : {delai}")

        return {"budget": budget, "delai": delai}

    def extraire_competences_via_llm(self, query: str) -> List[str]:
        prompt = f"""
Tu es un expert en gestion de projet industriel. Pour le projet suivant :

"{query}"

Donne uniquement la liste des compétences nécessaires sous le format JSON suivant :
["Compétence 1", "Compétence 2", "Compétence 3", ...]

Ne donne rien d'autre que la liste JSON, sans introduction ni explication.
"""

        response = self.llm.invoke(prompt).strip()
        print(f"[extraire_competences_via_llm] Réponse LLM brute : {response[:200]}")

        if not response:
            print("[extraire_competences_via_llm] Réponse LLM vide.")
            return []

        try:
            start = response.find('[')
            end = response.rfind(']') + 1
            if start == -1 or end == -1:
                print(f"[extraire_competences_via_llm] Pas de liste JSON détectée dans la réponse.")
                return []
            competences = json.loads(response[start:end])
            if not isinstance(competences, list):
                print(f"[extraire_competences_via_llm] Format JSON inattendu.")
                return []
            return competences
        except Exception as e:
            print(f"[extraire_competences_via_llm] Erreur parsing JSON : {e}")
            return []
