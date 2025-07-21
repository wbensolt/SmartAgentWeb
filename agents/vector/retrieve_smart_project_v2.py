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