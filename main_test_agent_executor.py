from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_community.chat_models import ChatOllama

# Importe tes modules locaux
from agents.nodes.hr_agents.tools.data_analyst import analyse_data_analyst
from agents.nodes.hr_agents.recruiter_agent import recruit_candidates
from core.llm_providers import LLMManager

def main():
    # 1. Création des outils
    tools = [
        Tool(
            name="DataAnalyst",
            func=lambda query: analyse_data_analyst.run({"query": query}),
            description="Utilisé pour analyser des données RH ou de projet"
        ),
        Tool(
            name="RecruiterAgent",
            func=lambda query: recruit_candidates.run({"query": query}),
            description="Utilisé pour rechercher et évaluer des candidats"
        ),
    ]

    # 2. Choix du modèle LLM
    llm = LLMManager().get_llm()  # Assure-toi que ce return un objet de type ChatOllama ou OpenAI

    # 3. Initialisation de l’agent
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION ,
        verbose=True,
        handle_parsing_errors=True
    )

    # 4. Prompt de test
    prompt = "Je veux lancer un projet IA à Lille avec 30000€ et recruter une équipe."
    print(f"🎯 Prompt envoyé : {prompt}\n")

    # 5. Exécution
    response = agent_executor.run(prompt)

    # 6. Affichage de la réponse
    print("\n🤖 Réponse de l'agent :")
    print(response)

    from agents.nodes.hr_agents.hr_workflow_advanced import create_workflow

    workflow_app = create_workflow()

    query = "Je veux lancer un projet IA à Lille avec 30000€ et recruter une équipe."

    print(f"🎯 Question : {query}")
    response = workflow_app.invoke({"messages": [{"role": "user", "content": query}]})
    print("\n🤖 Réponse :")
    print(response)

if __name__ == "__main__":
    main()
