# agents/nodes/hr_agents/schema.py

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class TalentPlan(BaseModel):
    upskill_possible: bool
    duree_formation: str
    referentiel_competences: str
    plan_global: str

class Resources(BaseModel):
    rh: int = Field(..., description="Nombre de personnes RH nécessaires")
    tutors: int = Field(..., description="Nombre de tuteurs nécessaires")
    tools: List[str] = Field(..., description="Liste des outils et documents fournis")

class Timeline(BaseModel):
    estimated_duration: str = Field(..., description="Durée estimée totale, ex: '30 jours'")
    per_person_days: int = Field(..., description="Nombre de jours par personne")
    completion_date: str | None = Field(None, description="Date de fin estimée (ISO)")

class OnboardingPlan(BaseModel):
    resources: Resources
    timeline: Timeline
    risks: List[str] = Field(..., description="Liste des risques identifiés")
    checklist: List[str] = Field(..., description="Liste des étapes à suivre")

# 1. Modèle Pydantic de sortie
class PayrollPlan(BaseModel):
    cout_mensuel: str  # ex: "133k€"
    budget_suffisant: str  # "oui" ou "non"
    analyse_cout: str
    recommandations: List[str]

class CritiqueRHPlan(BaseModel):
    points_forts: List[str]
    points_faibles: List[str]
    suggestions: List[str]
    note_coherence: int  # entre 0 et 5
    commentaire_global: str

class ValidationRHPlan(BaseModel):
    validation: str = Field(..., pattern="^(valide|non valide)$")
    justification: str = Field(..., max_length=50)

class FinalRHPlan(BaseModel):
    faisabilite: str = Field(..., pattern="^(Oui|Non|Partiel)$")
    conditions_reussite: List[str] = Field(default_factory=list)
    score_confiance: float = Field(..., ge=0.0, le=1.0)
    recommandation: str = Field(..., max_length=500)
    risques_principaux: List[str] = Field(default_factory=list)
    erreur: Optional[bool] = False

class RHResponse(BaseModel):
    response: str = Field(..., max_length=5000)
    legal_references: Dict[str, str] = Field(default_factory=dict)
    compliance_status: str = Field(..., pattern="^(conforme|à_verifier|non_conforme)$")
    error: Optional[bool] = False
    error_details: Optional[str] = None

class DataAnalystInsight(BaseModel):
    bassin_emploi: str = Field(..., max_length=200)
    disponibilite_profils: str = Field(..., max_length=1000)
    tendances_marche: List[str] = Field(..., max_items=5)
    budget_moyen: Optional[float] = None
    delai_moyen_lancement_projet: Optional[int] = None
    capacites_disponibles: Optional[List[Dict[str, Any]]] = None
    erreur: Optional[str] = None