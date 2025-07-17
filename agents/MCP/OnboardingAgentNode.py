from langchain_core.runnables import Runnable
from agents.nodes.hr_agents.onboarding_agent import OnboardingAgent
from agents.vector.state_schema import GraphState
from typing import Dict, Any

class OnboardingAgentNode(Runnable):
    def __init__(self, llm=None):
        self.agent = OnboardingAgent(llm=llm)

    def invoke(self, state: dict, **kwargs) -> Dict[str, Any]:
        onboarding_result = self.agent.invoke({
            "query": state.get("query", ""),
            "state": state.get("data_analytics", {}),
            "team_size": state.get("team_size", 15),
            "start_date": state.get("start_date", None)
        })
        return {
            "onboarding": onboarding_result
        }

    async def ainvoke(self, state: dict, **kwargs) -> Dict[str, Any]:
        return self.invoke(state, **kwargs)

    def __call__(self, state: dict, **kwargs) -> Dict[str, Any]:
        return self.invoke(state, **kwargs)

