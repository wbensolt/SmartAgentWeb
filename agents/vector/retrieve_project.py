from typing import Any, Dict
from langgraph.graph import StateGraph, END
from agents.vector.state_schema import GraphState
from core.llm_providers import LLMManager
from agents.nodes.hr_agents.tools.data_analyst import analyse_data_analyst
from agents.nodes.hr_agents.recruiter_agent import recruit_candidates  
from agents.nodes.hr_agents.rhagent import RHAgent
from agents.nodes.hr_agents.talentManagerrh import TalentManagerAgent
from agents.nodes.hr_agents.onboarding_agent import OnboardingAgent
from agents.nodes.hr_agents.payroll_agent import PayrollAgent
from agents.nodes.hr_agents.critique_rh_agent import CritiqueRHAgent
from agents.nodes.hr_agents.validation_rh_agent import ValidationRHAgent
from agents.nodes.hr_agents.final_rh_agent import FinalRHAgent
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from utils.json_utils import JSONRepairer
from datetime import datetime
import logging
import os
import json
import re

from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory

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

        # --- Tools ---
        tools = [
            Tool.from_function(
                func=analyse_data_analyst,
                name="AnalyseRH",
                description="Analyse les besoins RH de l'entreprise"
            ),
            Tool.from_function(
                func=recruit_candidates,
                name="Recruiter",
                description="Recherche des profils candidats selon la requête et données RH"
            )
        ]

        # --- Memoires ---
        memory_dataanalyst = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        memory_recruiter = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # --- Agents LangChain ---
        dataanalyst = initialize_agent(
            tools=[tools[0]],  # uniquement analyse_data_analyst
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            memory=memory_dataanalyst,
            verbose=True
        )

        recruiter_agent = initialize_agent(
            tools=[tools[1]],  # uniquement recruit_candidates
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            memory=memory_recruiter,
            verbose=True
        )

        rh = RHAgent(llm=llm)
        talent = TalentManagerAgent(llm=llm)
        onboarding = OnboardingAgent(llm=llm)
        payroll = PayrollAgent(llm=llm)
        critique = CritiqueRHAgent(llm=llm)
        validation = ValidationRHAgent(llm=llm)
        final = FinalRHAgent(llm=llm)

        agents_map = {
            "data_analytics": dataanalyst,
            "recruiter": recruiter_agent,
            "rh": rh,
            "talent": talent,
            "onboarding": onboarding,
            "payroll": payroll
        }

    except Exception as e:
        logger.critical(f"Erreur d'initialisation des agents: {str(e)}")
        raise

    # Wrapper générique pour la plupart des agents
    def safe_node_wrapper(agent, key: str, needs_data: bool = False):
        def node(state: GraphState) -> Dict[str, Any]:
            logger.info(f"[{key}] Etat reçu avec clés : {list(state.keys())}")
            try:
                input_data = {"input": state.get("query", "")}

                if needs_data and "data_analytics" in state:
                    input_data["input"] += f"\n\nInfos RH:\n{json.dumps(state['data_analytics'], ensure_ascii=False)}"

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

    # Wrapper spécifique pour recruiter qui appelle dataanalyst si data manquantes
    def safe_recruiter_node_wrapper(recruiter_agent, dataanalyst_agent, key: str):
        def node(state: GraphState) -> Dict[str, Any]:
            logger.info(f"[{key}] Etat reçu avec clés : {list(state.keys())}")
            try:
                query = state.get("query", "")
                data_analytics = state.get("data_analytics", {})

                # Si pas ou peu de data analytics, appel de dataanalyst
                if not data_analytics or len(data_analytics) == 0:
                    logger.info(f"[{key}] Pas de données analytiques, appel de dataanalyst...")
                    da_result = dataanalyst_agent.invoke({"input": query})
                    if isinstance(da_result, str):
                        da_result = JSONRepairer.safe_parse(clean_json_response(da_result))
                    # On récupère les données analytiques selon structure renvoyée
                    if da_result and "data_analytics" in da_result:
                        data_analytics = da_result["data_analytics"]
                    else:
                        data_analytics = da_result or {}

                # Appeler ensuite recruiter avec données mises à jour
                input_data = {
                    "input": query,
                    "data": data_analytics
                }

                raw_result = recruiter_agent.invoke(input_data)

                if isinstance(raw_result, str):
                    cleaned = clean_json_response(raw_result)
                    result = JSONRepairer.safe_parse(cleaned)
                else:
                    result = raw_result

                logger.info(f"[{key}] Résultat recruiter (trunc): {str(result)[:300]}")
                return {key: result}

            except Exception as e:
                logger.error(f"Erreur dans le nœud {key}: {str(e)}")
                return {key: {"error": str(e), "status": "failed"}}
        return node

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

    # Ajout des nœuds à la graph
    nodes_config = [
        ("dataanalyst", dataanalyst, "data_analytics", False),
        ("recruiter", recruiter_agent, "recruiter", True),  # ici, tool LangChain recruiter
        ("rh", rh, "rh", True),
        ("talent", talent, "talent", False),
        ("onboarding", onboarding, "onboarding", False),
        ("payroll", payroll, "payroll", False),
    ]

    for name, agent, key, needs_data in nodes_config:
        if name == "recruiter":
            graph.add_node(name, safe_recruiter_node_wrapper(agent, dataanalyst, key))
        else:
            graph.add_node(name, safe_node_wrapper(agent, key, needs_data))

    graph.add_node("critique", critique_node)
    graph.add_node("validation", validation_node)
    graph.add_node("final", final_node)

    graph.set_entry_point("dataanalyst")

    main_nodes = ["recruiter", "rh", "talent", "onboarding", "payroll"]
    for node in main_nodes:
        graph.add_edge("dataanalyst", node)
        graph.add_edge(node, "critique")

    graph.add_edge("critique", "validation")
    graph.add_edge("validation", "final")
    graph.add_edge("final", END)

    return graph
