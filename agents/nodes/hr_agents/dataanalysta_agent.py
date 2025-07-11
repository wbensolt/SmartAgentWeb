# agents/nodes/hr_agents/dataanalyst_agent.py

import json
import logging
from typing import Dict, Any
from langchain_core.runnables import Runnable
from core.llm_providers import LLMManager
from agents.nodes.hr_agents.schema import DataAnalystInsight
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser

class DataAnalystAgent(Runnable):
    def __init__(self, retriever, llm=None):
        self.retriever = retriever
        self.llm = llm or LLMManager().get_llm()
        self.logger = logging.getLogger(__name__)
        self.parser = PydanticOutputParser(pydantic_object=DataAnalystInsight)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)

    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        try:
            query = str(input.get("query", ""))
            docs = self.retriever.invoke(query)
            context = "\n".join(d.page_content for d in docs[:3])

            prompt = f"""
[SYSTEM]
Tu es un expert en data RH et analyse de marché pour les services en France.

Contexte :
{context}

Question :
{query}

Réponds uniquement avec un JSON VALIDE du type :
{{
  "bassin_emploi": "Description claire du marché local (ville/région)",
  "disponibilite_profils": "Résumé de la disponibilité des profils pour ce projet",
  "tendances_marche": ["Tendance 1", "Tendance 2"]
}}
- Sois synthétique et orienté décision.
- Interdiction d'utiliser des blocs de code ou des guillemets simples.
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
