from typing import Dict, Any
from langchain_core.runnables import Runnable
from agents.nodes.hr_agents.rhagent import RHAgent
from agents.vector.state_schema import GraphState
import logging

logger = logging.getLogger(__name__)

class RHAgentNode(Runnable):
    def __init__(self, llm=None):
        self.agent = RHAgent(llm=llm)
        self.logger = logging.getLogger(__name__)

    def invoke(self, state: GraphState, **kwargs) -> Dict[str, Any]:
        try:
            input_dict = {
                "query": state.get("query", ""),
                "state": state,
                "context": {}  # tu peux ajouter des infos externes ici
            }
            self.logger.info(f"[RHAgentNode] Invocation avec query: {input_dict['query']}")
            result = self.agent.invoke(input_dict)
            return {"rh": result}
        except Exception as e:
            self.logger.error(f"[RHAgentNode] Erreur invoke: {str(e)}", exc_info=True)
            return {"rh": {"error": str(e), "status": "failed"}}

    async def ainvoke(self, state: GraphState, **kwargs) -> Dict[str, Any]:
        return self.invoke(state, **kwargs)
    
    def __call__(self, state: GraphState, **kwargs) -> Dict[str, Any]:
        return self.invoke(state, **kwargs)
