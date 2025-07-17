from typing import TypedDict, Optional, List

class GraphState(TypedDict, total=False):
    query: str
    localisation: Optional[str]
    budget_utilisateur: Optional[float]
    delai_utilisateur_jours: Optional[int]
    project_info: Optional[dict]
    insight_data: Optional[dict]
    warnings: Optional[List[str]]
    error: Optional[str]
