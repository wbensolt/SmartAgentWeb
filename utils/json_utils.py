import json
import re
from typing import Any, Dict, Union

class JSONRepairer:
    @staticmethod
    def repair(json_str: Union[str, dict]) -> Dict[str, Any]:
        """Réparation agressive des JSON malformés"""
        # Si c'est déjà un dict, le retourner tel quel
        if isinstance(json_str, dict):
            return json_str
            
        try:
            # Conversion en string si nécessaire
            json_str = str(json_str).strip()
            
            # Suppression des blocs de code markdown
            json_str = re.sub(r'^```(json)?|```$', '', json_str, flags=re.IGNORECASE | re.MULTILINE)
            json_str = json_str.strip()
            
            # Extraction du contenu JSON principal
            match = re.search(r'\{.*\}', json_str, re.DOTALL)
            if not match:
                raise ValueError("Aucun JSON valide trouvé")
                
            json_content = match.group(0)
            
            # Correction des erreurs courantes
            json_content = re.sub(r',\s*([}\]])', r'\1', json_content)  # Virgules traînantes
            json_content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_content)  # Caractères non-printables
            json_content = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_content)  # Clés sans guillemets
            
            # Tentative de parsing
            return json.loads(json_content)
            
        except json.JSONDecodeError as e:
            # Fallback: extraction manuelle des paires clé-valeur
            try:
                pairs = re.findall(r'"([^"]+)"\s*:\s*("[^"]*"|\d+\.?\d*|true|false|null|\[[^\]]*\])', json_str, re.IGNORECASE)
                result = {}
                for k, v in pairs:
                    try:
                        if v.startswith('"') and v.endswith('"'):
                            result[k] = v[1:-1]  # Enlever les guillemets
                        elif v.startswith('[') and v.endswith(']'):
                            # Parsing simple des listes
                            items = re.findall(r'"([^"]*)"', v)
                            result[k] = items
                        elif v.lower() in ('true', 'false'):
                            result[k] = v.lower() == 'true'
                        elif v.lower() == 'null':
                            result[k] = None
                        else:
                            result[k] = float(v) if '.' in v else int(v)
                    except:
                        result[k] = str(v)
                
                if result:
                    return result
                else:
                    raise ValueError("Aucune donnée extraite")
                    
            except Exception as fallback_error:
                return {
                    "error": "Échec de réparation JSON",
                    "original_error": str(e),
                    "fallback_error": str(fallback_error),
                    "raw_content": json_str[:200]
                }

    @staticmethod
    def safe_parse(text: str):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                cleaned = text.strip()
                cleaned = re.sub(r"^```json\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
                cleaned = cleaned.replace("“", "\"").replace("”", "\"")
                cleaned = cleaned.replace("‘", "'").replace("’", "'")
                cleaned = re.sub(r"\n", " ", cleaned)
                cleaned = re.sub(r"\\'", "'", cleaned)
                # Répare les virgules manquantes entre éléments
                cleaned = re.sub(r'(":[^,}\]]+)(\s*")', r'\1,\2', cleaned)
                return json.loads(cleaned)
            except Exception as e:
                raise Exception(f"[JSONRepairer] Parsing échoué: {e}\nTexte brut: {text}")

    @staticmethod
    def validate_structure(data: Dict, required_fields: list) -> bool:
        """Valide qu'un dictionnaire contient les champs requis"""
        if not isinstance(data, dict):
            return False
        return all(field in data for field in required_fields)