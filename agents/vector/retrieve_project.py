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
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import logging
import re
import json
from utils.json_utils import JSONRepairer
from datetime import datetime
import os

logger = logging.getLogger(__name__)
CHROMA_PATH = "indexes/northwind_chroma"


def get_local_retriever():
    try:
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        return Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        ).as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        logger.error(f"Erreur d'initialisation du retriever: {str(e)}")
        raise


def clean_json_response(text: str) -> str:
    # Nettoie la chaîne JSON brute pour enlever markdown, guillemets typographiques, etc.
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

                # Affichage debug réponse brute
                print(f"\n[DEBUG] Réponse brute de l'agent {key} :\n{raw_result}")

                if isinstance(raw_result, str):
                    cleaned_result = clean_json_response(raw_result)
                    print(f"[DEBUG] Réponse nettoyée de l'agent {key} :\n{cleaned_result}")
                    result = JSONRepairer.safe_parse(cleaned_result)
                else:
                    result = raw_result

                print(f"[DEBUG] Réponse parsée JSON de l'agent {key} :\n{json.dumps(result, indent=2, ensure_ascii=False)}")

                return {key: result}

            except Exception as e:
                logger.error(f"Erreur dans le nœud {key}: {str(e)}")
                new_state = dict(state)
                new_state[key] = {"error": str(e), "status": "failed"}
                return new_state

        return node

    def critique_node(state: GraphState) -> Dict[str, Any]:
        try:
            content_parts = []
            required_nodes = ["recruiter", "rh", "talent", "onboarding", "payroll"]

            for node in required_nodes:
                if node in state and state[node]:
                    content = str(state[node])
                    if len(content) > 1500:
                        content = content[:750] + " [...] " + content[-750:]
                    content_parts.append(f"{node.upper()}:\n{content}")

            if not content_parts:
                new_state = dict(state)
                new_state["critique"] = {"error": "Aucune donnée à analyser"}
                return new_state

            critique_input = {"content": "\n\n".join(content_parts)[:8000]}
            critique_result = critique.invoke(critique_input)
            return {"critique": critique_result}

        except Exception as e:
            logger.error(f"Erreur dans critique_node: {str(e)}")
            new_state = dict(state)
            new_state["critique"] = {"error": str(e)}
            return new_state

    def validation_node(state: GraphState) -> Dict[str, Any]:
        try:
            critique_content = state.get("critique", {})

            if not critique_content or "error" in critique_content:
                new_state = dict(state)
                new_state["validation"] = {
                    "validation": "non valide",
                    "justification": "Critique invalide ou manquante"
                }
                return new_state

            validation_result = validation.invoke({"critique": critique_content})
            return {"validation": validation_result}

        except Exception as e:
            logger.error(f"Erreur dans validation_node: {str(e)}")
            new_state = dict(state)
            new_state["validation"] = {
                "validation": "erreur",
                "justification": str(e)[:200]
            }
            return new_state

    def final_node(state: GraphState) -> Dict[str, Any]:
        try:
            agent_responses = {}
            for key in ["data_analytics", "recruiter", "rh", "talent", "onboarding", "payroll"]:
                if key in state and state[key]:
                    agent_responses[key] = state[key]

            print("\n===== Réponses individuelles des agents =====")
            for k, v in agent_responses.items():
                print(f"\n--- {k.upper()} ---")
                if isinstance(v, dict):
                    print(json.dumps(v, indent=2, ensure_ascii=False))
                else:
                    print(v)

            inputs = {
                "answers": agent_responses,
                "critiques": state.get("critique", {}),
                "validations": state.get("validation", {})
            }

            final_result = final.invoke(inputs)

            print("\n===== Réponse finale =====")
            if isinstance(final_result, dict):
                print(json.dumps(final_result, indent=2, ensure_ascii=False))
            else:
                print(final_result)

            # === Génération du nom de fichier ===
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            slug = (
                re.sub(r"[^a-zA-Z0-9\-]+", "_", state.get("query", "no_query"))
                .strip("_")
                .lower()
            )
            filename = f"smart_project_{slug}_{timestamp}.json"
            save_dir = "outputs"
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, filename)

            # === Sauvegarde dans le fichier JSON ===
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump({
                    "query": state.get("query", ""),
                    "agent_answers": agent_responses,
                    "critique": state.get("critique", {}),
                    "validation": state.get("validation", {}),
                    "final_answer": final_result
                }, f, indent=2, ensure_ascii=False)

            print(f"\n✅ Résultats enregistrés dans : {filepath}")

            return {
                "agent_answers": agent_responses,
                "final_answer": final_result
            }

        except Exception as e:
            logger.error(f"Erreur dans final_node: {str(e)}")
            return {
                "agent_answers": {},
                "final_answer": {
                    "faisabilite": "Erreur",
                    "conditions_reussite": ["Vérifier les logs système"],
                    "score_confiance": 0.0,
                    "recommandation": f"Erreur de traitement: {str(e)[:200]}",
                    "risques_principaux": ["Erreur technique dans l'analyse"]
                }
            }

    # Configuration des noeuds
    nodes_config = [
        ("dataanalyst", dataanalyst, "data_analytics", False),
        ("recruiter", recruiter, "recruiter", True),
        ("rh", rh, "rh", True),
        ("talent", talent, "talent", False),
        ("onboarding", onboarding, "onboarding", False),
        ("payroll", payroll, "payroll", False)
    ]

    for name, agent, key, needs_data in nodes_config:
        graph.add_node(name, safe_node_wrapper(agent, key, needs_data))

    graph.add_node("critique", critique_node)
    graph.add_node("validation", validation_node)
    graph.add_node("final", final_node)

    graph.set_entry_point("dataanalyst")

    main_nodes = ["rh", "recruiter", "talent", "onboarding", "payroll"]
    for node in main_nodes:
        graph.add_edge("dataanalyst", node)
        graph.add_edge(node, "critique")

    graph.add_edge("critique", "validation")
    graph.add_edge("validation", "final")
    graph.add_edge("final", END)

    return graph
