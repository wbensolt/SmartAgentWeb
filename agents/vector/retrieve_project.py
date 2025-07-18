# retrieve_project_corrige.py
from typing import Any, Dict, Optional
from langgraph.graph import StateGraph, END
from agents.vector.state_schema import GraphState
from core.llm_providers import LLMManager
from agents.nodes.hr_agents.tools.data_analyst import analyse_data_analyst
from agents.nodes.hr_agents.recruiter_agent import recruit_candidates  
from agents.nodes.hr_agents.rhagent import check_labor_law
from agents.nodes.hr_agents.payroll_agent import payroll_agent_tool
from agents.nodes.hr_agents.onboarding_agent import onboarding_agent_tool
from agents.nodes.hr_agents.talentManagerrh import talent_manager_tool
from agents.nodes.hr_agents.critique_rh_agent import critique_agent_rh_tool
from agents.nodes.hr_agents.validation_rh_agent import validation_agent_rh_tool
from agents.nodes.hr_agents.final_rh_agent import final_agent_rh_tool
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from utils.json_utils import JSONRepairer
from datetime import datetime
import logging
import os
import json
import re
from agents.nodes.hr_agents.tools_wrapper.recruiter_tool_wrapper import recruiter_tool_wrapper
from agents.nodes.hr_agents.tools_wrapper.rh_tool_wrapper import rh_tool_wrapper

from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser

from core.formation_database import FormationDatabase
from agents.nodes.hr_agents.schema import TalentPlan, OnboardingPlan

logger = logging.getLogger(__name__)
CHROMA_PATH = "indexes/northwind_chroma"

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
parser_talent = PydanticOutputParser(pydantic_object=TalentPlan)
fixing_parser_talent = OutputFixingParser.from_llm(parser=parser_talent, llm=llm)
parser_onboarding = PydanticOutputParser(pydantic_object=OnboardingPlan)
fixing_parser_onboarding = OutputFixingParser.from_llm(parser=parser_onboarding, llm=llm)

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

def create_project_graph() -> StateGraph:
    graph = StateGraph(GraphState)
    try:
        global llm  
        retriever = get_local_retriever()

        # Création des outils LangChain
        analyse_rh_tool = Tool.from_function(
            func=analyse_data_analyst,
            name="AnalyseRH",
            description="Analyse les besoins RH de l'entreprise"
        )
        recruiter_tool = Tool.from_function(
            func=recruiter_tool_wrapper,
            name="Recruiter",
            description="Recherche des profils candidats selon la requête et données RH"
        )
        rh_tool = Tool.from_function(
            func=rh_tool_wrapper,
            name="RH",
            description="Vérifie la conformité RH par rapport au droit du travail"
        )
        talent_tool = Tool.from_function(
            func=talent_manager_tool,
            name="TalentManager",
            description="Analyse la montée en compétences et propose un plan de formation"
        )
        onboarding_tool = Tool.from_function(
            func=onboarding_agent_tool,
            name="OnboardingAgent",
            description="Estime les besoins d'onboarding RH, matériel, tuteurs et planning"
        )
        payroll_tool = Tool.from_function(
            func=payroll_agent_tool,
            name="Payroll",
            description="Estime le coût de paie et la suffisance du budget"
        )
        critique_rh_tool = Tool.from_function(
            func=critique_agent_rh_tool,
            name="CritiqueRH",
            description="Fournit une critique des propositions RH"
        )
        validation_rh_tool = Tool.from_function(
            func=validation_agent_rh_tool,
            name="ValidationRH",
            description="Valide les propositions RH avant mise en œuvre"
        )
        final_rh_tool = Tool.from_function(
            func=final_agent_rh_tool,
            name="FinalRH",
            description="Fournit une synthèse finale des recommandations RH"
        )

        # Mémoire dédiée par agent
        memories = {
            "dataanalyst": ConversationBufferMemory(memory_key="chat_history", return_messages=True),
            "recruiter": ConversationBufferMemory(memory_key="chat_history", return_messages=True),
            "rh": ConversationBufferMemory(memory_key="chat_history", return_messages=True),
            "talent": ConversationBufferMemory(memory_key="chat_history", return_messages=True),
            "onboarding": ConversationBufferMemory(memory_key="chat_history", return_messages=True),
            "payroll": ConversationBufferMemory(memory_key="chat_history", return_messages=True),
            "critique": ConversationBufferMemory(memory_key="chat_history", return_messages=True),
            "validation": ConversationBufferMemory(memory_key="chat_history", return_messages=True),
            "final": ConversationBufferMemory(memory_key="chat_history", return_messages=True),
        }

        # Initialisation agents LangChain
        dataanalyst = initialize_agent(
            tools=[analyse_rh_tool], llm=llm, agent=AgentType.OPENAI_FUNCTIONS,
            memory=memories["dataanalyst"], verbose=True
        )
        recruiter_agent = initialize_agent(
            tools=[recruiter_tool], llm=llm, agent=AgentType.OPENAI_FUNCTIONS,
            memory=memories["recruiter"], verbose=True
        )
        rh_agent = initialize_agent(
            tools=[rh_tool], llm=llm, agent=AgentType.OPENAI_FUNCTIONS,
            memory=memories["rh"], verbose=True
        )
        talent_agent = initialize_agent(
            tools=[talent_tool], llm=llm, agent=AgentType.OPENAI_FUNCTIONS,
            memory=memories["talent"], verbose=True
        )
        onboarding_agent = initialize_agent(
            tools=[onboarding_tool], llm=llm, agent=AgentType.OPENAI_FUNCTIONS,
            memory=memories["onboarding"], verbose=True
        )
        payroll = initialize_agent(
            tools=[payroll_tool], llm=llm, agent=AgentType.OPENAI_FUNCTIONS,
            memory=memories["payroll"], verbose=True
        )
        critique = initialize_agent(
            tools=[critique_rh_tool], llm=llm, agent=AgentType.OPENAI_FUNCTIONS,
            memory=memories["critique"], verbose=True
        )
        validation = initialize_agent(
            tools=[validation_rh_tool], llm=llm, agent=AgentType.OPENAI_FUNCTIONS,
            memory=memories["validation"], verbose=True
        )
        final = initialize_agent(
            tools=[final_rh_tool], llm=llm, agent=AgentType.OPENAI_FUNCTIONS,
            memory=memories["final"], verbose=True
        )

    except Exception as e:
        logger.critical(f"Erreur d'initialisation des agents: {e}")
        raise

    def create_node_wrapper(node_name: str, agent, output_key: str):
        """Factory unifiée pour créer les wrappers de nœud"""
        def wrapper(state: GraphState) -> Dict[str, Any]:
            try:
                # Gestion spéciale pour le recruteur
                if node_name == "recruiter":
                    data = state.data_analytics or {}
                    if hasattr(data, "dict"):
                        data = data.dict()
                    input_data = {
                        "input": state.query,
                        "data": data
                    }
                    result = recruiter_tool_wrapper(json.dumps(input_data, ensure_ascii=False))
                
                # Gestion spéciale pour RH
                elif node_name == "rh":
                    data = state.data_analytics or {}
                    if hasattr(data, "dict"):
                        data = data.dict()
                    input_data = {
                        "query": state.query,
                        "data": {
                            "budget": str(data.get("budget", "inconnu")),
                            "delai": str(data.get("delai", "inconnu")),
                            "context": data.get("context", {})
                        }
                    }
                    result = check_labor_law(json.dumps(input_data, ensure_ascii=False))
                
                # Gestion standard pour les autres nœuds
                else:
                    result = agent.invoke({"input": state.query})
                
                # Nettoyage et formatage du résultat
                if isinstance(result, str):
                    cleaned = clean_json_response(result)
                    result = JSONRepairer.safe_parse(cleaned)
                elif hasattr(result, "dict"):
                    result = result.dict()
                
                if isinstance(result, dict) and "chat_history" in result:
                    result["chat_history"] = [
                    {"content": msg.content, "type": msg.__class__.__name__}
                    for msg in result["chat_history"]
                ]
                
                return {output_key: result}
                
            except Exception as e:
                logger.error(f"Erreur dans le nœud {node_name}: {str(e)}")
                return {output_key: {"error": str(e)}}
        
        return wrapper

    # Configuration des nœuds
    nodes_config = [
        ("dataanalyst", dataanalyst, "data_analytics"),
        ("recruiter", recruiter_agent, "recruiter"),
        ("rh", rh_agent, "rh"),
        ("talent", talent_agent, "talent"),
        ("onboarding", onboarding_agent, "onboarding"),
        ("payroll", payroll, "payroll")
    ]

    for name, agent, key in nodes_config:
        graph.add_node(name, create_node_wrapper(name, agent, key))

    # Définition des nœuds de fin de chaîne
    def critique_node(state: GraphState) -> Dict[str, Any]:
        try:
            inputs = {
                "recruiter": state.recruiter if hasattr(state, "recruiter") else [],
                "rh": state.rh if hasattr(state, "rh") else {},
                "talent": state.talent if hasattr(state, "talent") else {},
                "onboarding": state.onboarding if hasattr(state, "onboarding") else {},
                "payroll": state.payroll if hasattr(state, "payroll") else {}
            }
            
            if not any(inputs.values()):
                return {"critique": "Aucune donnée à analyser"}
                
            critique_input = json.dumps({
                "query": state.query,
                "agent_responses": inputs
            }, ensure_ascii=False)
            
            result = critique.invoke({"input": critique_input})
            return {"critique": result}
            
        except Exception as e:
            return {"critique": {"error": str(e)}}

    def validation_node(state: GraphState) -> Dict[str, Any]:
        try:
            critique_data = state.critique if hasattr(state, "critique") else {}
            if isinstance(critique_data, str):
                validation_input = {"input": critique_data}
            else:
                validation_input = {"input": json.dumps(critique_data, ensure_ascii=False)}
            
            result = validation.invoke(validation_input)
            return {"validation": result}
            
        except Exception as e:
            return {"validation": {"validation": "erreur", "justification": str(e)}}

    def final_node(state: GraphState) -> Dict[str, Any]:
        try:
            # Sérialisation des données d'entrée
            def serialize(obj):
                if hasattr(obj, "dict"):
                    return obj.dict()
                if isinstance(obj, (str, int, float, bool)):
                    return obj
                if isinstance(obj, dict):
                    return {k: serialize(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [serialize(i) for i in obj]
                if hasattr(obj, "content"):  # Pour les messages LangChain
                    return {"content": obj.content, "type": obj.__class__.__name__}
                return str(obj)
            
            inputs = {
                "answers": {k: serialize(getattr(state, k)) for k in ["recruiter", "rh", "talent", "onboarding", "payroll"] if hasattr(state, k)},
                "critique": serialize(getattr(state, "critique", {})),
                "validations": serialize(getattr(state, "validation", {}))
            }
            
            result = final.invoke({"input": json.dumps(inputs, ensure_ascii=False)})
            
            # Sauvegarde des résultats
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("outputs", exist_ok=True)
            filepath = os.path.join("outputs", f"result_{timestamp}.json")
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump({
                    "query": state.query,
                    "results": {k: serialize(v) for k, v in inputs.items()},
                    "final_answer": serialize(result)
                }, f, indent=2, ensure_ascii=False)
                
            print(f"✅ Résultats sauvegardés dans {filepath}")
            return {"final_answer": result}
            
        except Exception as e:
            logger.error(f"Erreur finale: {str(e)}")
            return {"final_answer": {"error": str(e)}}

    # Ajout des nœuds de fin de chaîne
    graph.add_node("critique", critique_node)
    graph.add_node("validation", validation_node)
    graph.add_node("final", final_node)

    # Configuration du flux
    graph.set_entry_point("dataanalyst")
    
    # Branchement principal
    main_nodes = ["recruiter", "rh", "talent", "onboarding", "payroll"]
    for node in main_nodes:
        graph.add_edge("dataanalyst", node)
    
    # Convergence vers critique
    for node in main_nodes:
        graph.add_edge(node, "critique")
    
    # Flux conditionnel après validation
    def should_continue(state: GraphState) -> str:
        validation_data = state.validation if hasattr(state, "validation") else {}
        
        if isinstance(validation_data, str):
            if "non conforme" in validation_data.lower():
                return END
            return "final"
        
        if isinstance(validation_data, dict):
            status = validation_data.get("validation", "").lower()
            if status in ["non valide", "non conforme"]:
                return END
        
        return "final"

    graph.add_conditional_edges(
        "validation",
        should_continue,
        {END: END, "final": "final"}
    )
    
    graph.add_edge("final", END)

    return graph

if __name__ == "__main__":
    graph = create_project_graph()
    print("✅ Graphe créé avec succès")