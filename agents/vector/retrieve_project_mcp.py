from typing import Any, Dict
from langgraph.graph import StateGraph, END
from agents.MCP.OnboardingAgentNode import OnboardingAgentNode
from agents.vector.state_schema import GraphState
from core.llm_providers import LLMManager
from agents.MCP.DataAnalystAgentNode import DataAnalystAgentNode
from agents.MCP.RecruiterAgentNode import RecruiterAgentNode
from agents.MCP.RHAgentNode import RHAgentNode
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import logging

logger = logging.getLogger(__name__)

CHROMA_PATH = "indexes/northwind_chroma"

def get_local_retriever():
    try:
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings).as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        logger.error(f"Erreur d'initialisation du retriever: {str(e)}")
        raise

def safe_node_wrapper(agent, key: str):
    def node(state: GraphState) -> Dict[str, Any]:
        logger.info(f"[{key}] Etat reçu avec clés : {list(state.keys())}")
        try:
            result_state = agent(state)
            logger.info(f"[{key}] Résultat agent (trunc): {str(result_state)[:300]}")
            return {key: result_state}
        except Exception as e:
            logger.error(f"Erreur dans le nœud {key}: {str(e)}")
            return {key: {"error": str(e), "status": "failed"}}
    return node

def create_project_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    llm = LLMManager().get_llm()
    retriever = get_local_retriever()

    dataanalyst_node = DataAnalystAgentNode(
        retriever=retriever,
        llm=llm,
        db_path="data/northwind_company.db"
    )
    recruiter_node = RecruiterAgentNode(llm=llm)
    rh_node = RHAgentNode(llm=llm)
    onboarding_node = OnboardingAgentNode(llm=llm)
    #payroll_node = PayrollAgentNode(llm=llm)
    # Ajout des noeuds
    graph.add_node("dataanalyst", safe_node_wrapper(dataanalyst_node, "data_analytics"))
    graph.add_node("recruiter", safe_node_wrapper(recruiter_node, "recruiter"))
    graph.add_node("rh", safe_node_wrapper(rh_node, "rh"))
    graph.add_node("onboarding", safe_node_wrapper(onboarding_node, "onboarding"))
    #graph.add_node("payroll", safe_node_wrapper(payroll_node, "payroll"))

    # Point d'entrée
    graph.set_entry_point("dataanalyst")

    # Chaîne des étapes
    graph.add_edge("dataanalyst", "recruiter")
    graph.add_edge("recruiter", "rh")
    graph.add_edge("rh", "onboarding")
    graph.add_edge("onboarding", END)


    return graph
