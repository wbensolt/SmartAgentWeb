# project_graph.py corrigé

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
from agents.nodes.hr_agents.meta_rh_agent import MetaAgent
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
            logger.info(f"[{key}] Etat reçu avec clés : {list(state.keys())}")
            try:
                input_data = {"query": state.get("query", "")}
                if needs_data:
                    if "data_analytics" in state:
                        data_analytics = state["data_analytics"]
                        # Préparer les champs nécessaires explicitement pour recruiter
                        input_data["data"] = {
                            "competences_manquantes": data_analytics.get("competences_manquantes", []),
                            "localisation": data_analytics.get("localisation", "Non spécifiée"),
                            "budget_utilisateur": data_analytics.get("budget_utilisateur", 0),
                            "delai_utilisateur_jours": data_analytics.get("delai_utilisateur_jours", 0),
                            "capacites_disponibles": data_analytics.get("capacites_disponibles", []),
                        }
                        logger.info(f"[{key}] Passage data_analytics à l'agent avec clés : {list(input_data['data'].keys())}")
                    else:
                        logger.warning(f"[{key}] data_analytics absent du state, passage sans data")

                raw_result = agent.invoke(input_data)

                if isinstance(raw_result, str):
                    cleaned = clean_json_response(raw_result)
                    result = JSONRepairer.safe_parse(cleaned)
                else:
                    result = raw_result

                logger.info(f"[{key}] Résultat agent (trunc): {str(result)[:300]}")
                return {key: result}
            except Exception as e:
                logger.error(f"Erreur dans le nœud {key}: {str(e)}")
                return {key: {"error": str(e), "status": "failed"}}

        return node

    def meta_agent_node(state: GraphState) -> Dict[str, Any]:
        try:
            logger.info("[meta_agent_node] Collecte réponses agents précédents")
            agent_responses = {k: state[k] for k in agents_map.keys() if k in state}
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
            return {"meta_agent_error": str(e)}

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

            raw_final_result = final.invoke({
                "answers": agent_responses,
                "critiques": state.get("critique", {}),
                "validations": state.get("validation", {})
            })

            # Si la réponse est une chaîne, on tente un parsing JSON sécurisé
            if isinstance(raw_final_result, str):
                try:
                    final_result = json.loads(raw_final_result)
                except Exception as e:
                    logger.warning(f"[final_node] Impossible de parser final_result JSON, on utilise la chaîne brute. Erreur: {e}")
                    final_result = {"recommandation": raw_final_result}
            else:
                final_result = raw_final_result

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
            logger.error(f"Erreur dans final_node: {str(e)}", exc_info=True)
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
        ("recruiter", recruiter, "recruiter", True),   # needs_data=True to get data_analytics in input
        ("rh", rh, "rh", True),
        ("talent", talent, "talent", False),
        ("onboarding", onboarding, "onboarding", False),
        ("payroll", payroll, "payroll", False),
    ]

    # Ajout des nœuds avec wrapper sécurisé
    for name, agent, key, needs_data in nodes_config:
        graph.add_node(name, safe_node_wrapper(agent, key, needs_data))

    graph.add_node("meta_agent", meta_agent_node)
    graph.add_node("critique", critique_node)
    graph.add_node("validation", validation_node)
    graph.add_node("final", final_node)

    # Point d'entrée
    graph.set_entry_point("dataanalyst")

    # Construction des arcs (edges)
    main_nodes = ["recruiter", "rh", "talent", "onboarding", "payroll"]
    for node in main_nodes:
        graph.add_edge("dataanalyst", node)  # tous partent de dataanalyst
        graph.add_edge(node, "meta_agent")   # convergent vers meta_agent

    graph.add_edge("meta_agent", "critique")
    graph.add_edge("critique", "validation")
    graph.add_edge("validation", "final")
    graph.add_edge("final", END)

    return graph
