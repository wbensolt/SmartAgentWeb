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

        class MetaAgentWithLoop(MetaAgent):
            def invoke(self, input: dict) -> dict:
                state = input.get("state", {})
                iteration = 0
                
                # Nouveau: nettoyage préalable des réponses
                for key in list(state.keys()):
                    if isinstance(state[key], str):
                        try:
                            state[key] = json.loads(JSONRepairer.safe_parse(state[key]))
                        except:
                            pass
                
                while iteration < self.max_iterations:
                    iteration += 1
                    self.logger.info(f"[MetaAgent] Itération {iteration} de détection incohérences")
                    
                    # Prompt amélioré avec détection explicite des types d'incohérences
                    prompt = f"""
        ## Rôle
        Tu es un méta-analyste expert en détection d'incohérences dans les réponses d'agents spécialisés.
        Ton objectif : identifier contradictions, incohérences logiques, données manquantes ou divergences entre les réponses.

        ## Types d'incohérences à détecter
        1. Contradictions entre agents (ex: Budget vs Coûts)
        2. Incohérences temporelles (ex: Délais impossibles)
        3. Données manquantes pour une décision
        4. Divergences sur les ressources nécessaires
        5. Conformité réglementaire vs propositions

        ## Réponses des agents (format JSON)
        {json.dumps(state, indent=2, ensure_ascii=False)}

        ## Consignes de réponse
        - FORMATE ta réponse EN JSON STRICTEMENT
        - Pour CHAQUE incohérence détectée :
        * Identifie l'agent responsable
        * Décris le problème avec précision
        * Formule 1-2 questions ciblées
        - Si aucune incohérence : renvoyer {{"incoherences": []}}

        ## Format de réponse OBLIGATOIRE
        {{
        "incoherences": [
            {{
            "agent": "nom_agent",
            "probleme": "Description précise de l'incohérence",
            "questions": ["Question 1", "Question 2"]
            }}
        ]
        }}

        ## Exemples d'incohérences à repérer
        - "recruiter" cherche des profils en 'énergie solaire' alors que "data_analytics" signale leur disponibilité interne
        - "payroll" indique un coût de 150k€ mais "data_analytics" montre un budget moyen de 440k€
        - Délai de "onboarding" incompatible avec le calendrier projet
        """
                    try:
                        raw_response = self.llm.invoke(prompt)
                        response_clean = raw_response.strip()
                        
                        # Nettoyage amélioré des réponses JSON
                        if response_clean.startswith("```json"):
                            response_clean = response_clean[7:]
                        if response_clean.endswith("```"):
                            response_clean = response_clean[:-3]
                        response_clean = response_clean.strip()
                        
                        # Réparation robuste du JSON
                        response_clean = JSONRepairer.safe_parse(response_clean)
                        meta_data = json.loads(response_clean)
                        
                        self.logger.info(f"[MetaAgent] Analyse incohérences : {json.dumps(meta_data, indent=2)}")
                        
                        incoherences = meta_data.get("incoherences", [])
                        if not incoherences:
                            self.logger.info("[MetaAgent] Aucune incohérence détectée, boucle terminée.")
                            break

                        # Traitement des incohérences détectées
                        for issue in incoherences:
                            agent_key = issue.get("agent")
                            questions = issue.get("questions", [])
                            
                            if not agent_key or not questions:
                                continue
                                
                            agent = self.agents_map.get(agent_key)
                            if not agent:
                                self.logger.warning(f"[MetaAgent] Agent '{agent_key}' introuvable")
                                continue

                            for question in questions:
                                self.logger.info(f"[MetaAgent] Relance à '{agent_key}': {question}")
                                
                                agent_input = {
                                    "query": f"RELANCE MÉTA-AGENT: {question}",
                                    "state": state,
                                    "initial_query": state.get("query", ""),
                                    "is_meta_question": True  # Nouveau flag
                                }
                                
                                answer = agent.invoke(agent_input)
                                self.logger.info(f"[MetaAgent] Réponse de '{agent_key}': {json.dumps(answer, indent=2)}")

                                # Mise à jour incrémentale de l'état
                                if agent_key in state:
                                    if isinstance(state[agent_key], dict) and isinstance(answer, dict):
                                        state[agent_key].update(answer)
                                    else:
                                        state[agent_key] = answer
                                else:
                                    state[agent_key] = answer
                    except Exception as e:
                        self.logger.error(f"[MetaAgent] Erreur critique : {str(e)}", exc_info=True)
                        break
                
                return state

        meta_agent = MetaAgentWithLoop(llm=llm, agents_map=agents_map)

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
                return {key: {"error": str(e), "status": "failed"}}
        return node

    def meta_agent_node(state: GraphState) -> Dict[str, Any]:
        try:
            # Extraction des réponses avec vérification de complétude
            required_keys = ["data_analytics", "recruiter", "rh", "talent", "onboarding", "payroll"]
            agent_responses = {k: state.get(k, {}) for k in required_keys}
            
            # Log avant analyse
            logger.info(f"Données envoyées au méta-agent: {json.dumps(agent_responses, indent=2)}")
            
            updated_state = meta_agent.invoke({"state": agent_responses})
            
            # Log après analyse
            logger.info(f"État mis à jour par méta-agent: {json.dumps(updated_state, indent=2)}")
            
            # Vérification des modifications
            has_diff = any(
                agent_responses.get(k) != updated_state.get(k)
                for k in required_keys
            )
            
            new_state = dict(state)
            if has_diff:
                for k in required_keys:
                    if k in updated_state:
                        new_state[k] = updated_state[k]
            
            # Journalisation des modifications
            if "meta_agent_trace" not in new_state:
                new_state["meta_agent_trace"] = []
                
            new_state["meta_agent_trace"].append({
                "timestamp": datetime.now().isoformat(),
                "iteration": len(new_state["meta_agent_trace"]) + 1,
                "incoherences_detected": has_diff,
                "updated_agents": list(updated_state.keys()) if has_diff else []
            })
            
            return new_state
            
        except Exception as e:
            logger.error(f"ERREUR méta-agent: {str(e)}")
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
            final_result = final.invoke({
                "answers": agent_responses,
                "critiques": state.get("critique", {}),
                "validations": state.get("validation", {})
            })
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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

    # Correction de la configuration des edges
    graph.set_entry_point("dataanalyst")

    # Nouvelle structure avec synchronisation
    graph.add_node("parallel_start", lambda state: state)  # Nœud de synchronisation
    graph.add_node("collect", lambda state: state)         # Nœud de collecte

    main_nodes = ["recruiter", "rh", "talent", "onboarding", "payroll"]

    # Configuration des edges
    graph.add_edge("dataanalyst", "parallel_start")
    for node in main_nodes:
        graph.add_edge("parallel_start", node)
        graph.add_edge(node, "collect")  # Tous les agents pointent vers collect

    graph.add_edge("collect", "meta_agent")  # Meta-agent après tous les agents
    graph.add_edge("meta_agent", "critique")
    graph.add_edge("critique", "validation")
    graph.add_edge("validation", "final")
    graph.add_edge("final", END)


    return graph
