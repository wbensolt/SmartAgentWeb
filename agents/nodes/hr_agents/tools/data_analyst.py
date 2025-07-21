# tools/data_analyst.py

from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import sqlite3
import unicodedata
import string
import json
import regex
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)
CHROMA_PATH = "indexes/northwind_chroma"

def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def extraire_budget_delai(query: str) -> Dict[str, Any]:
    logger.info(f"💰 Extraction budget/délai depuis la requête: {query}")
    query_norm = normalize_text(query)
    budget, delai = None, None

    match_budget = regex.search(r"(\d+)[\s]*euros?", query_norm)
    if match_budget:
        budget = float(match_budget.group(1))

    match_delai = regex.search(r"(\d+)[\s]*(mois|jours?)", query_norm)
    if match_delai:
        val, unite = int(match_delai.group(1)), match_delai.group(2)
        delai = val * 30 if "mois" in unite else val

    return {"budget": budget, "delai": delai}

def extraire_localisation(query: str) -> str:
    logger.info(f"📍 Extraction de la localisation depuis la requête: {query}")
    villes = ["toulouse", "paris", "lyon", "marseille", "lille", "bordeaux"]
    query_norm = normalize_text(query)
    for ville in villes:
        if normalize_text(ville) in query_norm:
            return ville.capitalize()
    return "Non spécifiée"

def extraire_contexte_societe(db_path: str = "data/northwind_company.db") -> Dict[str, Any]:
    logger.info("🔍 Extraction du contexte société depuis la base de données")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT AVG(budget_disponible), AVG(duree_prevue_jours) FROM projets")
        avg_budget, avg_duree = cursor.fetchone()
        cursor.execute("""
            SELECT c.nom, COUNT(ec.employe_id)
            FROM employes_competences ec
            JOIN competences c ON ec.competence_id = c.id
            GROUP BY c.nom
        """)
        capacites = [{"competence": row[0], "effectif": row[1]} for row in cursor.fetchall()]
    except Exception as e:
        return {"erreur": str(e)}
    finally:
        conn.close()
    return {
        "budget_moyen": round(avg_budget or 0, 2),
        "delai_moyen_lancement_projet": int(avg_duree or 0),
        "capacites_disponibles": capacites
    }

@tool
def analyse_data_analyst(query: str) -> Dict[str, Any]:
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
    logger.info("🔎 Début de l'analyse des données")
    logger.info(f"Requête: {query}")

    try:
        societe = extraire_contexte_societe()
        localisation = extraire_localisation(query)
        budget_delai = extraire_budget_delai(query)

        return {
            "bassin_emploi": localisation,
            "budget_utilisateur": budget_delai["budget"],
            "delai_utilisateur_jours": budget_delai["delai"],
            "localisation": localisation,
            **societe,
            "erreur": None
        }
    except Exception as e:
        return {"erreur": str(e)}
