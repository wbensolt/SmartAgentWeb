import sys
import json
import re
from agents.vector.retrieve_project import create_project_graph
from agents.vector.retrieve import create_dynamic_rh_graph, ensure_index
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
from agents.vector.state_schema import GraphState
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


def clean_response(text: str) -> str:
    """Nettoie la réponse brute pour faciliter le parsing JSON."""
    cleaned = text.strip()
    # Supprimer les balises ```json``` ou ```
    cleaned = re.sub(r"```json|```", "", cleaned, flags=re.IGNORECASE).strip()
    # Remplacer les guillemets typographiques par des guillemets standards
    cleaned = (
        cleaned
        .replace("“", "\"")
        .replace("”", "\"")
        .replace("‘", "'")
        .replace("’", "'")
    )
    return cleaned


def run_smart_agent():
    print("🧠 Lancement du Smart RH Project Agent...")
    graph = create_project_graph().compile()
    while True:
        query = input("\nPose ta question (ou 'exit') > ").strip()
        if query.lower() == "exit":
            break

        result = graph.invoke({"query": query})
        raw_response = result.get("final_answer", result)

        print("[DEBUG] Réponse brute avant nettoyage :")
        print(raw_response)

        if isinstance(raw_response, str):
            cleaned = clean_response(raw_response)
            try:
                parsed = json.loads(cleaned)
                print("\n=== ✅ Réponse Smart Agent (JSON validé) ===")
                print(json.dumps(parsed, indent=2, ensure_ascii=False))
            except json.JSONDecodeError as e:
                print(f"❌ Réponse JSON invalide : {str(e)}")
                print("Tentative d'afficher la réponse brute :")
                print(raw_response)
        elif isinstance(raw_response, dict):
            print("\n=== ✅ Réponse Smart Agent (dict déjà parsé) ===")
            print(json.dumps(raw_response, indent=2, ensure_ascii=False))
        else:
            print(f"⚠️ Réponse inattendue de type {type(raw_response)} :")
            print(raw_response)

        print("=================================\n")


def run_rh_agent():
    print("👔 Préparation du Retriever RH dynamique...")
    ensure_index()
    graph = create_dynamic_rh_graph().compile()
    while True:
        query = input("\nPose ta question RH (ou 'exit') > ").strip()
        if query.lower() == "exit":
            break

        result = graph.invoke({"query": query})
        print("\n=== ✅ Réponse Finale de l'Agent RH ===")
        print(result.get("final_answer", "Aucune réponse générée."))
        print("=======================================\n")


def run_single_agent():
    print("🔍 Test unitaire d’un agent RH")
    agents = {
        "dataanalyst": DataAnalystAgent,
        "recruiter": RecruiterAgent,
        "rh": RHAgent,
        "talent": TalentManagerAgent,
        "onboarding": OnboardingAgent,
        "payroll": PayrollAgent,
        "critique": CritiqueRHAgent,
        "validation": ValidationRHAgent,
        "final": FinalRHAgent,
    }

    print("\nListe des agents disponibles :")
    for i, key in enumerate(agents.keys()):
        print(f"{i + 1}. {key}")

    choice = input("\nChoisis un agent (nom ou numéro, 'exit' pour quitter) > ").strip()
    if choice.lower() == "exit":
        return

    try:
        if choice.isdigit():
            key = list(agents.keys())[int(choice) - 1]
        else:
            key = choice.lower()

        AgentClass = agents[key]
        llm = LLMManager().get_llm()

        # Préparer les données d'entrée spécifiques selon l'agent
        if key == "dataanalyst":
            retriever = Chroma(
                persist_directory="indexes/northwind_chroma",
                embedding_function=OllamaEmbeddings(model="mxbai-embed-large")
            ).as_retriever()
            agent = AgentClass(retriever=retriever, llm=llm)
            input_data = {"query": input("\nTa question > ").strip()}
        elif key == "validation":
            input_data = {"critique": input("\nDonne une critique > ").strip()}
            agent = AgentClass(llm=llm)
        elif key == "final":
            input_data = {"answers": {}, "critiques": {}, "validations": {}}
            agent = AgentClass(llm=llm)
        else:
            input_data = {"query": input("\nTa question > ").strip()}
            agent = AgentClass(llm=llm)

        result = agent.invoke(input_data)

        print("\n=== ✅ Réponse de l’agent ===")
        if isinstance(result, dict):
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(result)

    except Exception as e:
        print(f"❌ Erreur : {str(e)}")


if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "project"

    if mode == "rh":
        run_rh_agent()
    elif mode == "test":
        run_single_agent()
    else:
        run_smart_agent()
