from langchain_core.runnables import Runnable
from core.llm_providers import LLMManager
from typing import Dict, Any
import json
import logging

class CritiqueRHAgent(Runnable):
    def __init__(self, llm=None):
        self.llm = LLMManager().get_llm()
        self.logger = logging.getLogger(__name__)
        
    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        try:
            content = input.get("content", "")
            if not content:
                return self._empty_response()
            
            # Échapper les accolades dans le contenu pour éviter erreurs dans f-string
            safe_content = content.replace("{", "{{").replace("}", "}}")
            
            prompt = f"""Tu es un expert RH qui fait des critiques constructives. Analyse ce contenu et réponds UNIQUEMENT avec un JSON valide :

Contenu à analyser :
{safe_content[:3000]}

Réponds STRICTEMENT avec ce format JSON (sans texte avant ou après) :
{{
    "points_forts": ["point1", "point2"],
    "points_faibles": ["point1", "point2"],
    "suggestions": ["suggestion1", "suggestion2"],
    "note_coherence": 4,
    "commentaire_global": "commentaire synthétique"
}}"""
            
            response = self.llm.invoke(prompt)
            return self._parse_critique(response)
            
        except Exception as e:
            self.logger.error(f"Erreur CritiqueRHAgent: {str(e)}")
            return self._error_response(str(e))

    def _parse_critique(self, response: str) -> Dict[str, Any]:
        """Parsing robuste des critiques"""
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
                
                return {
                    "points_forts": [str(p)[:100] for p in data.get("points_forts", [])][:5],
                    "points_faibles": [str(p)[:100] for p in data.get("points_faibles", [])][:5],
                    "suggestions": [str(s)[:100] for s in data.get("suggestions", [])][:5],
                    "note_coherence": min(max(int(data.get("note_coherence", 3)), 0), 5),
                    "commentaire_global": str(data.get("commentaire_global", ""))[:300]
                }
            else:
                raise ValueError("Pas de JSON trouvé")
                
        except Exception as e:
            self.logger.warning(f"Erreur parsing critique: {str(e)}")
            return self._error_response("Format de critique invalide")

    def _empty_response(self) -> Dict[str, Any]:
        return {
            "points_forts": ["Aucun contenu à analyser"],
            "points_faibles": ["Contenu vide"],
            "suggestions": ["Fournir du contenu à analyser"],
            "note_coherence": 0,
            "commentaire_global": "Aucun contenu fourni"
        }

    def _error_response(self, error: str) -> Dict[str, Any]:
        return {
            "points_forts": [],
            "points_faibles": ["Erreur de traitement"],
            "suggestions": ["Vérifier les logs"],
            "note_coherence": 0,
            "commentaire_global": f"Erreur: {error[:100]}"
        }
