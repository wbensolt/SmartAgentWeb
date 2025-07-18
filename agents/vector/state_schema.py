from pydantic import BaseModel
from typing import Optional, Dict, Any

class GraphState(BaseModel):
    query: str
    data_analytics: Optional[Dict[str, Any]] = None
    recruiter: Optional[Dict[str, Any]] = None
    rh: Optional[Dict[str, Any]] = None
    talent: Optional[Dict[str, Any]] = None
    onboarding: Optional[Dict[str, Any]] = None
    payroll: Optional[Dict[str, Any]] = None
    critique: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    final_answer: Optional[Dict[str, Any]] = None

    # Ajoutez cette méthode pour permettre l'accès dictionnaire
    def get(self, key, default=None):
        return getattr(self, key, default)