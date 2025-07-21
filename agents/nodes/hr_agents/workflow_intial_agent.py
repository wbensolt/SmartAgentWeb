from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_community.chat_models import ChatOllama

from agents.nodes.hr_agents.tools.data_analyst import analyse_data_analyst
from agents.nodes.hr_agents.recruiter_agent import recruit_candidates
from core.llm_providers import LLMManager

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

# 2. Choix du modèle LLM (ici Ollama, mais tu peux mettre OpenAI, Groq, etc.)
llm = LLMManager().get_llm()

# 3. Initialisation de l’agent
agent_executor = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# 4. Exécution
response = agent_executor.run("Je veux lancer un projet IA à Lille avec 30000€ et recruter une équipe.")
print(response)
