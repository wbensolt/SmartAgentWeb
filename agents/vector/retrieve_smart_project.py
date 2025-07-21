from typing import Any, Dict, Optional
from langgraph.graph import StateGraph, END
from agents.vector.state_schema import GraphState
from core.llm_providers import LLMManager
from agents.nodes.hr_agents.tools.data_analyst import analyse_data_analyst
from agents.nodes.hr_agents.recruiter_agent import recruit_candidates  
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from utils.json_utils import JSONRepairer
from datetime import datetime
import logging
import os
import json
import re
from agents.nodes.hr_agents.tools_wrapper.recruiter_tool_wrapper import recruiter_tool_wrapper
from langchain.agents import initialize_agent, AgentType, Tool
from pydantic import BaseModel
from core.formation_database import FormationDatabase
from agents.nodes.hr_agents.schema import TalentPlan, OnboardingPlan

logger = logging.getLogger(__name__)
CHROMA_PATH = "indexes/northwind_chroma"

class EnhancedGraphState(BaseModel):
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

    def get(self, key, default=None):
        return getattr(self, key, default)

def get_local_retriever():
    try:
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings).as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        logger.error(f"Erreur d'initialisation du retriever: {e}")
        raise

def clean_json_response(text: str) -> str:
    cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    cleaned = cleaned.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
    return cleaned

formation_db = FormationDatabase()
llm = LLMManager().get_llm()

def analyze_formation_feasibility(competences_manquantes, budget_moyen, delai_moyen):
    formations_trouvees, cout_total, duree_max_semaines = [], 0, 0
    try:
        budget_num = float(str(budget_moyen).replace("€", "").replace(",", "").replace(" ", "")) if budget_moyen != "inconnu" else 50000
        delai_num = float(str(delai_moyen).replace("jours", "").replace(" ", "")) if delai_moyen != "inconnu" else 90
    except Exception:
        budget_num, delai_num = 50000, 90

    for comp in competences_manquantes:
        formation = formation_db.find_formation(comp)
        if formation:
            formations_trouvees.append({"competence": comp, "formation": formation})
            cout_total += formation.get("cout_moyen", 0)
            try:
                duree_max = int(formation.get("duree", "0").split("-")[-1].split()[0])
                duree_max_semaines = max(duree_max_semaines, duree_max)
            except Exception:
                duree_max_semaines = max(duree_max_semaines, 8)

    ratio_budget = (cout_total / budget_num * 100) if budget_num > 0 else 0
    duree_jours = duree_max_semaines * 7
    ratio_delai = (duree_jours / delai_num * 100) if delai_num > 0 else 0

    faisable = ratio_budget <= 25 and ratio_delai <= 60 and len(formations_trouvees) >= len(competences_manquantes) * 0.7
    return {
        "formations_trouvees": formations_trouvees,
        "cout_total": cout_total,
        "duree_max_semaines": duree_max_semaines,
        "ratio_budget": ratio_budget,
        "ratio_delai": ratio_delai,
        "faisable": faisable,
        "nb_competences_formables": len(formations_trouvees)
    }

def generate_formation_plan(analysis):
    if not analysis["formations_trouvees"]:
        return "Aucune formation spécifique identifiée"

    plan_parts = [
        f"• {item['competence']}: {item['formation']['duree']} via {', '.join(item['formation']['organismes'][:2])} "
        f"(certification: {item['formation']['certifications'][0]}, taux réussite: {item['formation']['taux_reussite']}%)"
        for item in analysis["formations_trouvees"]
    ]

    organismes = list({org for item in analysis["formations_trouvees"] for org in item["formation"]["organismes"][:2]})
    return (
        "Plan de formation personnalisé:\n"
        + "\n".join(plan_parts)
        + f"\n\nSynthèse:\n- Durée: {analysis['duree_max_semaines']} semaines\n"
        + f"- Coût: {analysis['cout_total']:,} € ({analysis['ratio_budget']:.1f}% budget)\n"
        + f"- Compétences couvertes: {analysis['nb_competences_formables']}\n"
        + f"- Organismes: {', '.join(organismes)}"
    )

def run_data_analyst(state: EnhancedGraphState):
    try:
        retriever = get_local_retriever()
        analysis_result = analyse_data_analyst(state.query, retriever)
        state.data_analytics = analysis_result
        return state
    except Exception as e:
        logger.error(f"Data analyst error: {e}")
        state.data_analytics = {"error": str(e)}
        return state

def run_recruiter(state: EnhancedGraphState):
    if not state.data_analytics:
        state.recruiter = {"error": "No data analytics input"}
        return state
    
    try:
        recruitment_result = recruiter_tool_wrapper(state.data_analytics)
        state.recruiter = recruitment_result
        return state
    except Exception as e:
        logger.error(f"Recruiter error: {e}")
        state.recruiter = {"error": str(e)}
        return state

def create_project_graph() -> StateGraph:
    workflow = StateGraph(EnhancedGraphState)

    # Définition des nœuds
    workflow.add_node("data_analyst", run_data_analyst)
    workflow.add_node("recruiter", run_recruiter)

    # Configuration du flux
    workflow.set_entry_point("data_analyst")
    workflow.add_edge("data_analyst", "recruiter")
    workflow.add_edge("recruiter", END)

    return workflow

if __name__ == "__main__":
    # Exemple d'utilisation
    initial_state = EnhancedGraphState(
        query="Analyse les compétences manquantes pour le projet X"
    )
    
    app = create_project_graph().compile()
    result = app.invoke(initial_state)
    
    print("✅ Résultat final:")
    print(json.dumps(result.final_answer, indent=2))