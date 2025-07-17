import unicodedata
import json
import logging
import re
import sqlite3
import unicodedata
import string
from typing import Any, Dict, List, Optional
from langchain_core.runnables import Runnable
from agents.state.graph_state import GraphState
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

class DataAnalystAgentNode:
    def __init__(self, retriever, llm, db_path: str):
        self.retriever = retriever
        self.llm = llm
        self.db_path = db_path
        self.parser = PydanticOutputParser(pydantic_object=DataAnalystInsight)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=llm)
        self.villes_connues = [
    "Paris", "Lyon", "Marseille", "Toulouse", "Nice",
    "Bordeaux", "Lille", "Nantes", "Strasbourg", "Grenoble"
]

    def __call__(self, state: GraphState) -> GraphState:
        query = state.get("query", "")
        societe_data = extraire_contexte_societe(self.db_path)

        docs = self.retriever.invoke(query)
        context = "\n".join(d.page_content for d in docs[:3])

        localisation = self.extraire_localisation(query)
        budget_delai = self.extraire_budget_delai(query)
        budget_user_val = budget_delai.get("budget", 0)
        delai_user_val = budget_delai.get("delai", 0)

        competences_visees = self.extraire_competences_via_llm(query)
        competences_internes = {
            c["competence"].lower(): c["effectif"]
            for c in societe_data["capacites_disponibles"]
        }

        competences_couvertes = []
        competences_manquantes = []
        for c in competences_visees:
            if competences_internes.get(c.lower(), 0) >= 5:
                competences_couvertes.append({
                    "competence": c, "effectif": competences_internes[c.lower()]
                })
            else:
                competences_manquantes.append(c)

        warnings = []
        if budget_user_val < societe_data["budget_moyen"] * 0.5:
            warnings.append("Budget utilisateur très faible comparé à la moyenne.")
        if delai_user_val < societe_data["delai_moyen_lancement_projet"] * 0.5:
            warnings.append("Délai utilisateur très court comparé à la moyenne.")
        if competences_manquantes:
            warnings.append(f"Compétences manquantes : {', '.join(competences_manquantes)}")

        # MCP: Génération de l’insight
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
NE fournis qu’un JSON VALIDE avec exactement les champs suivants :
{{ "bassin_emploi": "", "disponibilite_profils": "", "tendances_marche": [""], "budget_moyen": 0, "delai_moyen_lancement_projet": 0, "capacites_disponibles": [{{"competence": "", "effectif": 0}}], "erreur": null }}
"""
        try:
            raw = self.llm.invoke(prompt)
            clean = self._extract_json_block(raw)
            parsed = self.fixing_parser.parse(clean)
            insight_data = parsed.dict()
        except Exception as e:
            insight_data = {
                "bassin_emploi": "",
                "disponibilite_profils": "",
                "tendances_marche": [],
                "budget_moyen": societe_data["budget_moyen"],
                "delai_moyen_lancement_projet": societe_data["delai_moyen_lancement_projet"],
                "capacites_disponibles": societe_data["capacites_disponibles"],
                "erreur": str(e)
            }

        return {
            **state,
            "localisation": localisation,
            "budget_utilisateur": budget_user_val,
            "delai_utilisateur_jours": delai_user_val,
            "project_info": {
                "competences_vises": competences_visees,
                "competences_couvertes": competences_couvertes,
                "competences_manquantes": competences_manquantes
            },
            "insight_data": insight_data,
            "warnings": warnings if warnings else [],
        }
    
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

