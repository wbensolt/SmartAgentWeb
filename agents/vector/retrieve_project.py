from typing import Any, Dict
from langgraph.graph import StateGraph, END
from agents.vector.state_schema import GraphState
from core.llm_providers import LLMManager
from agents.nodes.hr_agents.dataanalysta_agent import DataAnalystAgent
from agents.nodes.hr_agents.recruiter_agent import RecruiterAgent
from agents.nodes.hr_agents.rhagent import RHAgent
from agents.nodes.hr_agents.talentManagerrh import TalentManagerAgent
from agents.nodes.hr_agents.onboarding_agent import OnboardingAgent
from agents.nodes.hr_agents.payroll_agent import PayrollAgent
from agents.nodes.hr_agents.critique_rh_agent import CritiqueRHAgent
from agents.nodes.hr_agents.validation_rh_agent import ValidationRHAgent
from agents.nodes.hr_agents.final_rh_agent import FinalRHAgent
from agents.nodes.hr_agents.meta_rh_agent import MetaAgent  # 🔁 Assure-toi que ce fichier existe
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from utils.json_utils import JSONRepairer
from datetime import datetime
import logging
import os
import json
import re

logger = logging.getLogger(__name__)
CHROMA_PATH = "indexes/northwind_chroma"


def get_local_retriever():
    try:
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings).as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        logger.error(f"Erreur d'initialisation du retriever: {str(e)}")
        raise


def clean_json_response(text: str) -> str:
    cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    cleaned = cleaned.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
    return cleaned


def create_project_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    try:
        llm = LLMManager().get_llm()
        retriever = get_local_retriever()
        dataanalyst = DataAnalystAgent(retriever=retriever, llm=llm)
        recruiter = RecruiterAgent(llm=llm)
        rh = RHAgent(llm=llm)
        talent = TalentManagerAgent(llm=llm)
        onboarding = OnboardingAgent(llm=llm)
        payroll = PayrollAgent(llm=llm)
        critique = CritiqueRHAgent(llm=llm)
        validation = ValidationRHAgent(llm=llm)
        final = FinalRHAgent(llm=llm)

        agents_map = {
            "data_analytics": dataanalyst,
            "recruiter": recruiter,
            "rh": rh,
            "talent": talent,
            "onboarding": onboarding,
            "payroll": payroll
        }
        meta_agent = MetaAgent(llm=llm, agents_map=agents_map)

    except Exception as e:
        logger.critical(f"Erreur d'initialisation des agents: {str(e)}")
        raise

    def safe_node_wrapper(agent, key: str, needs_data: bool = False):
        def node(state: GraphState) -> Dict[str, Any]:
            try:
                input_data = {"query": state["query"]}
                if needs_data and "data_analytics" in state:
                    input_data["data"] = state["data_analytics"]

                raw_result = agent.invoke(input_data)

                if isinstance(raw_result, str):
                    cleaned = clean_json_response(raw_result)
                    result = JSONRepairer.safe_parse(cleaned)
                else:
                    result = raw_result

                return {key: result}

            except Exception as e:
                logger.error(f"Erreur dans le nœud {key}: {str(e)}")
                new_state = dict(state)
                new_state[key] = {"error": str(e), "status": "failed"}
                return new_state

        return node

    def meta_agent_node(state: GraphState) -> Dict[str, Any]:
        try:
            print("[DEBUG] Execution de meta_agent_node")
            agent_responses = {key: state[key] for key in agents_map.keys() if key in state}
            updated_state = meta_agent.invoke({"state": agent_responses})
            new_state = dict(state)
            new_state.update(updated_state)

            if "meta_agent_trace" not in new_state:
                new_state["meta_agent_trace"] = []

            new_state["meta_agent_trace"].append({
                "timestamp": datetime.now().isoformat(),
                "iteration": len(new_state["meta_agent_trace"]) + 1,
                "updated_responses": updated_state
            })
            return new_state

        except Exception as e:
            logger.error(f"Erreur dans meta_agent_node: {str(e)}")
            new_state = dict(state)
            new_state["meta_agent_error"] = str(e)
            return new_state

    def critique_node(state: GraphState) -> Dict[str, Any]:
        try:
            parts = [f"{k.upper()}:\n{str(state[k])}" for k in ["recruiter", "rh", "talent", "onboarding", "payroll"] if k in state]
            if not parts:
                return {"critique": {"error": "Aucune donnée à analyser"}}
            critique_input = {"content": "\n\n".join(parts)[:8000]}
            return {"critique": critique.invoke(critique_input)}
        except Exception as e:
            logger.error(f"Erreur dans critique_node: {str(e)}")
            return {"critique": {"error": str(e)}}

    def validation_node(state: GraphState) -> Dict[str, Any]:
        try:
            critique_content = state.get("critique", {})
            if not critique_content or "error" in critique_content:
                return {"validation": {"validation": "non valide", "justification": "Critique invalide ou manquante"}}
            return {"validation": validation.invoke({"critique": critique_content})}
        except Exception as e:
            logger.error(f"Erreur dans validation_node: {str(e)}")
            return {"validation": {"validation": "erreur", "justification": str(e)}}

    def final_node(state: GraphState) -> Dict[str, Any]:
        try:
            agent_responses = {k: state[k] for k in agents_map if k in state}
            final_result = final.invoke({
                "answers": agent_responses,
                "critiques": state.get("critique", {}),
                "validations": state.get("validation", {})
            })

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            slug = re.sub(r"[^a-zA-Z0-9\-]+", "_", state.get("query", "no_query")).strip("_").lower()
            filename = f"smart_project_{timestamp}.json"
            filepath = os.path.join("outputs", filename)
            os.makedirs("outputs", exist_ok=True)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump({
                    "query": state.get("query", ""),
                    "agent_answers": agent_responses,
                    "meta_agent_trace": state.get("meta_agent_trace", []),
                    "critique": state.get("critique", {}),
                    "validation": state.get("validation", {}),
                    "final_answer": final_result
                }, f, indent=2, ensure_ascii=False)

            print(f"\n✅ Résultats enregistrés dans : {filepath}")
            return {"agent_answers": agent_responses, "final_answer": final_result}

        except Exception as e:
            logger.error(f"Erreur dans final_node: {str(e)}")
            return {
                "agent_answers": {},
                "final_answer": {
                    "faisabilite": "Erreur",
                    "conditions_reussite": [],
                    "score_confiance": 0.0,
                    "recommandation": f"Erreur de traitement: {str(e)}",
                    "risques_principaux": ["Erreur technique dans l'analyse"]
                }
            }

    nodes_config = [
        ("dataanalyst", dataanalyst, "data_analytics", False),
        ("recruiter", recruiter, "recruiter", True),
        ("rh", rh, "rh", True),
        ("talent", talent, "talent", False),
        ("onboarding", onboarding, "onboarding", False),
        ("payroll", payroll, "payroll", False),
    ]

    for name, agent, key, needs_data in nodes_config:
        graph.add_node(name, safe_node_wrapper(agent, key, needs_data))

    graph.add_node("meta_agent", meta_agent_node)
    graph.add_node("critique", critique_node)
    graph.add_node("validation", validation_node)
    graph.add_node("final", final_node)

    graph.set_entry_point("dataanalyst")

    main_nodes = ["recruiter", "rh", "talent", "onboarding", "payroll"]
    for node in main_nodes:
        graph.add_edge("dataanalyst", node)
        graph.add_edge(node, "meta_agent")

    graph.add_edge("meta_agent", "critique")
    graph.add_edge("critique", "validation")
    graph.add_edge("validation", "final")
    graph.add_edge("final", END)

    return graph