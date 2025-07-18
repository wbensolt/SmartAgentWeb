# tools/data_analyst.py

import json
import logging
import sqlite3
import unicodedata
import string
from typing import Any, Dict
import regex  # regex récursif pour JSON

from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

from core.llm_providers import LLMManager

# 📦 Mémoire
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

logging.basicConfig(level=logging.INFO)
CHROMA_PATH = "indexes/northwind_chroma"
logger = logging.getLogger(__name__)


def get_local_retriever():
    try:
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings).as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        logger.error(f"Erreur d'initialisation du retriever: {str(e)}")
        raise

retriever = get_local_retriever()

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
        result = {
            "budget_moyen": 0,
            "delai_moyen_lancement_projet": 0,
            "capacites_disponibles": [],
            "erreur": str(e)
        }
    finally:
        conn.close()
    return result

def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def extraire_budget_delai(query: str) -> Dict[str, Any]:
    query_norm = normalize_text(query)
    budget = None
    delai = None

    match_budget = regex.search(r"(\d+)[\s]*euros?", query_norm)
    if match_budget:
        try:
            budget = float(match_budget.group(1).replace(",", "."))
        except Exception:
            pass

    match_delai = regex.search(r"(\d+)[\s]*(mois|jours?)", query_norm)
    if match_delai:
        val, unite = int(match_delai.group(1)), match_delai.group(2)
        delai = val * 30 if "mois" in unite else val

    return {"budget": budget, "delai": delai}

def extraire_localisation(query: str, villes_connues=None) -> str:
    villes_connues = villes_connues or ["toulouse", "paris", "lyon", "marseille", "lille", "bordeaux"]
    query_norm = normalize_text(query)
    for ville in villes_connues:
        if normalize_text(ville) in query_norm:
            return ville.capitalize()
    return "Non spécifiée"

@tool
def analyse_data_analyst(query: str) -> dict:
    """
    Analyse une requête utilisateur décrivant un projet et fournit une évaluation RH de faisabilité.

    Cette fonction utilise :
    - un retriever vectoriel (Chroma) pour obtenir du contexte externe,
    - des données internes extraites d'une base SQLite (`northwind_company.db`) sur le budget moyen,
      le délai moyen de lancement de projets, et les compétences disponibles,
    - des fonctions NLP pour extraire la localisation, le budget et le délai depuis la requête utilisateur.

    Elle compile ces données dans un prompt destiné à un LLM pour obtenir une réponse en format JSON,
    qui inclut les éléments suivants :
    - bassin d'emploi,
    - disponibilité des profils,
    - tendances du marché,
    - budget/délai moyen vs utilisateur,
    - localisation,
    - et les compétences internes disponibles.

    Args:
        query (str): Requête textuelle de l'utilisateur décrivant un projet.

    Returns:
        dict: Résultat de l'analyse sous forme de dictionnaire JSON.
    """
    logger.info("[DataAnalyst] Analyse du prompt projet lancé.")
    logger.info("[DataAnalyst] Prompt reçu : %s", query)

    llm = LLMManager().get_llm(verbose=True)
    context_docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in context_docs[:3]])

    societe = extraire_contexte_societe("data/northwind_company.db")
    localisation = extraire_localisation(query)
    budget_delai = extraire_budget_delai(query)
    budget_user_val = budget_delai["budget"] or 0
    delai_user_val = budget_delai["delai"] or 0

    prompt = f"""
Tu es un expert en stratégie RH.
Voici le contexte externe :
{context}

Et les données internes :
- Budget moyen : {societe['budget_moyen']} €
- Délai moyen : {societe['delai_moyen_lancement_projet']} jours
- Compétences internes : {json.dumps(societe['capacites_disponibles'], ensure_ascii=False)}

Projet demandé :
{query}

Analyse le projet et retourne un JSON :
{{
  "bassin_emploi": "",
  "disponibilite_profils": "",
  "tendances_marche": [""],
  "budget_moyen": {societe['budget_moyen']},
  "delai_moyen_lancement_projet": {societe['delai_moyen_lancement_projet']},
  "capacites_disponibles": {json.dumps(societe['capacites_disponibles'], ensure_ascii=False)},
  "budget_utilisateur": {budget_user_val},
  "delai_utilisateur_jours": {delai_user_val},
  "localisation": "{localisation}",
  "erreur": null
}}
""".strip()

    logger.info("[DataAnalyst] Prompt complet envoyé au LLM :\n%s", prompt)

    try:
        response = llm.invoke(prompt)

        if hasattr(response, "content"):
            text_response = response.content
        else:
            text_response = str(response)

        logger.info("[DataAnalyst] Réponse brute du LLM : %s", text_response)

        json_match = regex.search(r"\{(?:[^{}]|(?R))*\}", text_response)
        if json_match:
            json_str = json_match.group(0)
            resultat = json.loads(json_str)
            logger.info("[DataAnalyst] JSON parsé : %s", resultat)
            return resultat
        else:
            raise ValueError("Aucun JSON détecté dans la réponse.")

    except Exception as e:
        logger.error("[DataAnalyst] Erreur parsing JSON ou LLM : %s", str(e), exc_info=True)
        return {
            "erreur": f"Erreur parsing ou LLM : {str(e)}",
            "budget_utilisateur": budget_user_val,
            "delai_utilisateur_jours": delai_user_val,
            "localisation": localisation
        }
