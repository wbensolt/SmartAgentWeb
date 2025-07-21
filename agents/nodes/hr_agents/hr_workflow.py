# hr_workflow.py
from langgraph.graph import StateGraph, END
from typing import Dict, Any, TypedDict
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import logging
import json

logger = logging.getLogger(__name__)

# Définition du state
class AgentState(TypedDict):
    input: str
    data_analysis: Dict[str, Any]
    recruiter_results: Dict[str, Any]

# Wrapper pour vos outils existants
def run_data_analyst(query: str) -> Dict[str, Any]:
    from agents.nodes.hr_agents.tools.data_analyst import analyse_data_analyst
    logger.info("🔍 Exécution de data_analyst")
    return analyse_data_analyst.run({"query": query})

def run_recruiter(query: str, data: Dict[str, Any]) -> Dict[str, Any]:
    from agents.nodes.hr_agents.recruiter_agent import recruit_candidates
    logger.info("👥 Exécution de recruiter")
    return recruit_candidates.run({"query": query, "data": data})

# Définition des nœuds
def data_analyst_node(state: AgentState) -> Dict[str, Any]:
    try:
        result = run_data_analyst(state["input"])
        return {"data_analysis": result}
    except Exception as e:
        logger.error(f"Erreur data_analyst: {str(e)}")
        return {"data_analysis": {"error": str(e)}}

def recruiter_node(state: AgentState) -> Dict[str, Any]:
    try:
        analysis = state.get("data_analysis", {})
        result = run_recruiter(state["input"], analysis)
        return {"recruiter_results": result}
    except Exception as e:
        logger.error(f"Erreur recruiter: {str(e)}")
        return {"recruiter_results": {"error": str(e)}}

def final_node(state: AgentState) -> Dict[str, Any]:
    analysis = state.get("data_analysis", {})
    recruitment = state.get("recruiter_results", {})
    
    response = {
        "analysis": analysis,
        "recruitment": recruitment,
        "recommendation": "Voici les résultats de l'analyse et du recrutement"
    }
    
    logger.info("✅ Workflow terminé")
    return {"result": response}

# Création du workflow
def create_hr_workflow():
    workflow = StateGraph(AgentState)
    
    # Ajout des nœuds
    workflow.add_node("data_analyst", data_analyst_node)
    workflow.add_node("recruiter", recruiter_node)
    workflow.add_node("final", final_node)
    
    # Définition du flux
    workflow.set_entry_point("data_analyst")
    workflow.add_edge("data_analyst", "recruiter")
    workflow.add_edge("recruiter", "final")
    workflow.add_edge("final", END)
    
    return workflow.compile()

# Exemple d'utilisation
if __name__ == "__main__":
    # Initialisation
    workflow = create_hr_workflow()
    
    # Exécution
    inputs = {"input": "Lancer un projet à Toulouse sur 3 mois avec un budget de 30000 euros en data"}
    
    for output in workflow.stream(inputs):
        for key, value in output.items():
            print(f"Étape: {key}")
            print(json.dumps(value, indent=2, ensure_ascii=False))