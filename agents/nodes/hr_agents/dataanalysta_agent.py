from typing import Dict, Any
import json
import logging
from langchain_core.runnables import Runnable
from core.llm_providers import LLMManager

class DataAnalystAgent(Runnable):
    def __init__(self, retriever, llm=None):
        self.retriever = retriever
        self.llm = LLMManager().get_llm()
        self.logger = logging.getLogger(__name__)
        
    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        try:
            query = str(input.get("query", ""))
            docs = self.retriever.invoke(query)
            context = "\n".join(d.page_content for d in docs[:3])
            
            prompt = f"""
[SYSTEM]
Tu es un expert en data RH et analyse de marché pour les services en France.

Ta mission : évaluer le potentiel de recrutement et les tendances du marché local pour un projet donné.

Réponds uniquement avec un JSON VALIDE du type :
{
  "bassin_emploi": "Description claire du marché local (ville/région)",
  "disponibilite_profils": "Résumé de la disponibilité des profils pour ce projet",
  "tendances_marche": ["Tendance 1", "Tendance 2", ...]
}
- Sois synthétique et orienté décision.
- Interdiction d'utiliser des blocs de code ou des guillemets simples.
"""
            
            response = self.llm.invoke(prompt)
            return self._parse_response(response)
            
        except Exception as e:
            self.logger.error(f"Erreur DataAnalyst: {str(e)}")
            return {
                "bassin_emploi": "Erreur d'analyse",
                "disponibilite_profils": "Données indisponibles",
                "tendances_marche": ["Erreur de traitement"],
                "erreur": str(e)
            }

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse robuste des réponses JSON"""
        try:
            # Nettoyage de la réponse
            cleaned = response.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned.replace('```json', '').replace('```', '').strip()
            
            # Extraction du JSON
            start = cleaned.find('{')
            end = cleaned.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = cleaned[start:end]
                data = json.loads(json_str)
                
                # Validation et nettoyage des données
                return {
                    "bassin_emploi": str(data.get("bassin_emploi", "Non analysé"))[:200],
                    "disponibilite_profils": str(data.get("disponibilite_profils", "Non analysé"))[:200],
                    "tendances_marche": [str(t)[:100] for t in data.get("tendances_marche", [])][:5],
                    "erreur": data.get("erreur")
                }
            else:
                raise ValueError("Pas de JSON trouvé dans la réponse")
                
        except Exception as e:
            self.logger.warning(f"Réponse JSON invalide: {response[:200]}")
            return {
                "bassin_emploi": "Données indisponibles",
                "disponibilite_profils": "Données indisponibles",
                "tendances_marche": ["Analyse impossible"],
                "erreur": f"Erreur de parsing: {str(e)}"
            }