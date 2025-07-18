from typing import Optional, List, Dict, Any
from pydantic import BaseModel

class GraphState(BaseModel):
    query: Optional[str] = None
    data_analytics: Optional[Dict[str, Any]] = None
    recruiter: Optional[List[Dict[str, Any]]] = None
    rh: Optional[str] = None
    talent: Optional[Dict[str, Any]] = None
    onboarding: Optional[Dict[str, Any]] = None
    payroll: Optional[Dict[str, Any]] = None
    critique: Optional[str] = None
    validation: Optional[Dict[str, Any]] = None
    final_answer: Optional[Dict[str, Any]] = None
