# agent_factory.py
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import Tool
from core.llm_providers import LLMManager
from agents.nodes.hr_agents.tools.data_analyst import analyse_data_analyst
from agents.nodes.hr_agents.recruiter_agent import recruit_candidates


def make_agent():
    tools = [
        Tool.from_function(
            func=analyse_data_analyst.run,
            name="data_analyst",
            description="Analyse les besoins RH"
        ),
        Tool.from_function(
            func=recruit_candidates.run,
            name="recruiter",
            description="Recrute des candidats"
        )
    ]
    
    return create_react_agent(
        model=LLMManager().get_llm(),
        tools=tools,
        prompt="Vous DEVEZ utiliser les outils dans l'ordre: data_analyst puis recruiter",
        version="v2"
    )