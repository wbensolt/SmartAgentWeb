from langchain_core.runnables import Runnable
from typing import Dict, Any
from agents.nodes.hr_agents.recruiter_agent import RecruiterAgent
from agents.vector.state_schema import GraphState

class RecruiterAgentNode(Runnable):
    def __init__(self, llm=None):
        self.agent = RecruiterAgent(llm=llm)

    def invoke(self, state: dict, **kwargs) -> Dict[str, Any]:
        recruiter_result = self.agent.invoke({
            "query": state.get("query", ""),
            "data": state.get("data_analytics", {})
        })
        return {
            "recruiter": recruiter_result
        }

    async def ainvoke(self, state: dict, **kwargs) -> Dict[str, Any]:
        return self.invoke(state, **kwargs)
    
    def __call__(self, state: dict, **kwargs) -> Dict[str, Any]:
        return self.invoke(state, **kwargs)
