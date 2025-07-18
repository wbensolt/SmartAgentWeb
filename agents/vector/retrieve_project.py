from typing import Any, Dict
from langgraph.graph import StateGraph, END
from agents.vector.state_schema import GraphState
from core.llm_providers import LLMManager
from agents.nodes.hr_agents.tools.data_analyst import analyse_data_analyst
from agents.nodes.hr_agents.recruiter_agent import recruit_candidates  
from agents.nodes.hr_agents.rhagent import check_labor_law  # fonction tool LangChain
from agents.nodes.hr_agents.payroll_agent import  payroll_agent_tool
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
        logger.error(f"Erreur d'initialisation du retriever: {str(e)}")
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
    except:
        budget_num, delai_num = 50000, 90

    for comp in competences_manquantes:
        formation = formation_db.find_formation(comp)
        if formation:
            formations_trouvees.append({"competence": comp, "formation": formation})
            cout_total += formation["cout_moyen"]
            try:
                duree_max = int(formation["duree"].split("-")[-1].split()[0])
                duree_max_semaines = max(duree_max_semaines, duree_max)
            except:
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
        f"• {item['competence']}: {item['formation']['duree']} via {', '.join(item['formation']['organismes'][:2])} (certification: {item['formation']['certifications'][0]}, taux réussite: {item['formation']['taux_reussite']}%)"
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

        # Création des tools LangChain à partir des fonctions tools
        analyse_rh_tool = Tool.from_function(
            func=analyse_data_analyst,
            name="AnalyseRH",
            description="Analyse les besoins RH de l'entreprise"
        )
        recruiter_tool = Tool.from_function(
            func=recruit_candidates,
            name="Recruiter",
            description="Recherche des profils candidats selon la requête et données RH"
        )
        rh_tool = Tool.from_function(
            func=check_labor_law.invoke,
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

        # mémoires dédiées par agent
        memory_dataanalyst = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        memory_recruiter = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        memory_rh = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        memory_talent = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        memory_onboarding = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        memory_payroll = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        memory_critique = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        memory_validation = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        memory_final = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # initialisation des agents LangChain
        dataanalyst = initialize_agent(
            tools=[analyse_rh_tool],
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            memory=memory_dataanalyst,
            verbose=True
        )
        recruiter_agent = initialize_agent(
            tools=[recruiter_tool],
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            memory=memory_recruiter,
            verbose=True
        )
        rh_agent = initialize_agent(
            tools=[rh_tool],
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            memory=memory_rh,
            verbose=True
        )
        talent_agent = initialize_agent(
            tools=[talent_tool],
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            memory=memory_talent,
            verbose=True
        )
        onboarding_agent = initialize_agent(
            tools=[onboarding_tool],
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            memory=memory_onboarding,
            verbose=True
        )
        payroll = initialize_agent(
            tools=[payroll_tool],
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            memory=memory_payroll,
            verbose=True
        )
        critique = initialize_agent(
            tools=[critique_rh_tool],
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            memory=memory_critique,
            verbose=True
        )
        validation = initialize_agent(
            tools=[validation_rh_tool],
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            memory=memory_validation,
            verbose=True
        )
        final = initialize_agent(
            tools=[final_rh_tool],
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            memory=memory_final,
            verbose=True
        )

        agents_map = {
            "data_analytics": dataanalyst,
            "recruiter": recruiter_agent,
            "rh": rh_agent,
            "talent": talent_agent,
            "onboarding": onboarding_agent,
            "payroll": payroll
        }

    except Exception as e:
        logger.critical(f"Erreur d'initialisation des agents: {str(e)}")
        raise

    def safe_node_wrapper(agent, key: str, needs_data: bool = False):
        def node(state: GraphState) -> Dict[str, Any]:
            logger.info(f"[{key}] Etat reçu avec clés : {list(state.dict(exclude_none=True).keys())}")
            try:
                input_payload = {"input": getattr(state, "query", "") or ""}
                if needs_data and getattr(state, "data_analytics", None):
                    input_payload["data"] = getattr(state, "data_analytics")
                raw_result = agent.invoke(input_payload)
                if isinstance(raw_result, str):
                    cleaned = clean_json_response(raw_result)
                    result = JSONRepairer.safe_parse(cleaned)
                else:
                    result = raw_result
                logger.info(f"[{key}] Résultat agent (trunc): {str(result)[:300]}")
                return {key: result}
            except Exception as e:
                logger.error(f"Erreur dans le nœud {key}: {str(e)}")
                if key == "recruiter":
                    return {key: []}  # liste vide en cas d'erreur
                else:
                    return {key: ""}  # chaîne vide sinon
        return node

    def safe_recruiter_node_wrapper(recruiter_agent, dataanalyst_agent, key: str):
        def node(state: GraphState) -> Dict[str, Any]:
            logger.info(f"[{key}] Etat reçu avec clés : {list(state.dict(exclude_none=True).keys())}")
            try:
                query = getattr(state, "query", "") or ""
                data_analytics = getattr(state, "data_analytics", {}) or {}

                if not data_analytics:
                    logger.info(f"[{key}] 🚀 Appel de l'agent Data Analyst avec query : {query}")
                    da_result = dataanalyst_agent.invoke({"input": query})
                    logger.info(f"[{key}] ✅ Résultat brut de Data Analyst : {da_result}")

                    if isinstance(da_result, str):
                        cleaned = clean_json_response(da_result)
                        logger.info(f"[{key}] 🧹 Résultat nettoyé de Data Analyst : {cleaned}")
                        da_result = JSONRepairer.safe_parse(cleaned)

                    if da_result and "data_analytics" in da_result:
                        data_analytics = da_result["data_analytics"]
                    else:
                        data_analytics = da_result or {}

                    logger.info(f"[{key}] 📊 Données analytiques récupérées : {json.dumps(data_analytics, ensure_ascii=False, indent=2)}")

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
                return {key: []}  # liste vide en cas d'erreur
        return node

    def critique_node(state: GraphState) -> Dict[str, Any]:
        try:
            parts = [f"{k.upper()}:\n{str(state[k])}" for k in ["recruiter", "rh", "talent", "onboarding", "payroll"] if k in state]
            if not parts:
                return {"critique": "Aucune donnée à analyser"}

            critique_input = {"input": "\n\n".join(parts)[:8000]}  # clé "input" ici !
            raw_result = critique.invoke(critique_input)

            # Assurer que la sortie est une chaîne
            if isinstance(raw_result, dict):
                # En cas d'erreur retournée sous forme dict, on stringify
                import json
                cleaned_result = json.dumps(raw_result, ensure_ascii=False)
            else:
                cleaned_result = str(raw_result)

            return {"critique": cleaned_result}

        except Exception as e:
            logger.error(f"Erreur dans critique_node: {str(e)}")
            return {"critique": f"Erreur interne critique_node: {str(e)}"}


    def validation_node(state: GraphState) -> Dict[str, Any]:
        try:
            critique_content = getattr(state, "critique", {})
            if not critique_content or "error" in critique_content:
                return {"validation": {"validation": "non valide", "justification": "Critique invalide ou manquante"}}
            return {"validation": validation.invoke({"critique": critique_content})}
        except Exception as e:
            logger.error(f"Erreur dans validation_node: {str(e)}")
            return {"validation": {"validation": "erreur", "justification": str(e)}}

    def final_node(state: GraphState) -> Dict[str, Any]:
        try:
            agent_responses = {k: getattr(state, k) for k in agents_map if hasattr(state, k)}

            raw_final_result = final.invoke({
                "answers": agent_responses,
                "critiques": getattr(state, "critique", {}),
                "validations": getattr(state, "validation", {})
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
            slug = re.sub(r"[^a-zA-Z0-9\-]+", "_", getattr(state, "query", "no_query")).strip("_").lower()
            filename = f"smart_project_{timestamp}.json"
            filepath = os.path.join("outputs", filename)
            os.makedirs("outputs", exist_ok=True)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump({
                    "query": getattr(state, "query", ""),
                    "agent_answers": agent_responses,
                    "critique": getattr(state, "critique", {}),
                    "validation": getattr(state, "validation", {}),
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

    # Ajout des noeuds au graph
    nodes_config = [
        ("dataanalyst", dataanalyst, "data_analytics", False),
        ("recruiter", recruiter_agent, "recruiter", True),
        ("rh", rh_agent, "rh", True),
        ("talent", talent_agent, "talent", False),
        ("onboarding", onboarding_agent, "onboarding", False),
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


# Pour tester la création du graph sans erreur (optionnel)
if __name__ == "__main__":
    graph = create_project_graph()
    print("Graphe créé avec succès.")
