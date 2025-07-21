# hr_workflow_advanced.py
from langgraph.prebuilt import ToolNode, create_react_agent
from langchain_community.tools import Tool
from core.llm_providers import LLMManager
from typing import TypedDict
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    messages: list

# Configuration des outils
def setup_tools():
    from agents.nodes.hr_agents.tools.data_analyst import analyse_data_analyst
    from agents.nodes.hr_agents.recruiter_agent import recruit_candidates
    
    return [
        Tool(
            name="data_analyst",
            func=lambda q: analyse_data_analyst.run({"query": q}),
            description="Analyse les besoins RH"
        ),
        Tool(
            name="recruiter",
            func=lambda q: recruit_candidates.run({"query": q}),
            description="Recrute des candidats"
        )
    ]

def create_workflow():
    llm = LLMManager().get_llm()
    tools = setup_tools()
    
    # Configuration spéciale pour Groq
    agent = initialize_agent(
        tools=[analyse_tool, recruiter_tool],
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,  # Ou ZERO_SHOT_REACT_DESCRIPTION selon le modèle
        verbose=True
    )

    
    # Configuration du workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", ToolNode(tools))
    
    workflow.set_entry_point("agent")
    workflow.add_edge("tools", "agent")
    
    # Ajout de la logique conditionnelle
    def route_to_tools(state: AgentState):
        last_msg = state["messages"][-1]
        if "tool_calls" in last_msg.additional_kwargs:
            return "tools"
        return END
    
    workflow.add_conditional_edges(
        "agent",
        route_to_tools,
        {"tools": "tools", END: END}
    )
    
    return workflow.compile()