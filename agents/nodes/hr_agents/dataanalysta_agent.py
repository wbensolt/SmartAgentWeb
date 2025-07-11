import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional
from langchain_core.runnables import Runnable
from core.llm_providers import LLMManager
from agents.nodes.hr_agents.schema import DataAnalystInsight
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser


class DataAnalystAgent(Runnable):
    def __init__(self, retriever, llm=None, db_path: str = "data/northwind_company.db"):
        self.retriever = retriever
        self.llm = llm or LLMManager().get_llm()
        self.logger = logging.getLogger(__name__)
        self.parser = PydanticOutputParser(pydantic_object=DataAnalystInsight)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
        self.db_path = db_path

    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        try:
            query = str(input.get("query", ""))
            docs = self.retriever.invoke(query)
            context = "\n".join(d.page_content for d in docs[:3])
            societe_data = extraire_contexte_societe(self.db_path)

            prompt = f"""
            [SYSTEM]
            Tu es un expert en data RH, stratégie interne et évaluation de faisabilité.

            Contexte du marché :
            {context}

            Données internes de l'entreprise :
            - Budget moyen des projets : {societe_data['budget_moyen']} €
            - Délai moyen de lancement de projet : {societe_data['delai_moyen_lancement_projet']} jours
            - Compétences disponibles en interne :
            {json.dumps(societe_data['capacites_disponibles'], indent=2, ensure_ascii=False)}

            Question :
            {query}

            Réponds uniquement avec un JSON VALIDE du type :
            {{
            "bassin_emploi": "Description claire du marché local (ville/région)",
            "disponibilite_profils": "Résumé de la disponibilité des profils pour ce projet",
            "tendances_marche": ["Tendance 1", "Tendance 2"],
            "budget_moyen": 0,
            "delai_moyen_lancement_projet": 0,
            "capacites_disponibles": [
                {{"competence": "Nom", "effectif": 0}}
            ],
            "erreur": null
            }}

            - Sois synthétique, orienté décision.
            - N'utilise pas de guillemets simples ni de blocs de code.
                        """

            raw_response = self.llm.invoke(prompt)
            cleaned = raw_response.strip().replace("```json", "").replace("```", "").strip()
            parsed = self.fixing_parser.parse(cleaned)
            return parsed.dict()

        except Exception as e:
            self.logger.error(f"[DataAnalystAgent] Erreur : {str(e)}", exc_info=True)
            return DataAnalystInsight(
                bassin_emploi="Erreur d'analyse",
                disponibilite_profils="Données indisponibles",
                tendances_marche=["Erreur de traitement"],
                erreur=str(e)
            ).dict()

    async def ainvoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return self.invoke(input)


def extraire_contexte_societe(sqlite_path: str) -> Dict[str, Any]:
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()
    result = {}

    try:
        # Budget moyen et durée moyenne
        cursor.execute("SELECT AVG(budget_disponible), AVG(duree_prevue_jours) FROM projets")
        avg_budget, avg_duree = cursor.fetchone()
        result["budget_moyen"] = round(avg_budget or 0, 2)
        result["delai_moyen_lancement_projet"] = int(avg_duree or 0)

        # Calcul dynamique des capacités internes
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
        logging.error(f"[extraire_contexte_societe] Erreur SQLite : {str(e)}", exc_info=True)
        result = {
            "budget_moyen": 0,
            "delai_moyen_lancement_projet": 0,
            "capacites_disponibles": [],
        }
    finally:
        conn.close()

    return result
