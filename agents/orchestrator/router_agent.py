# agents/orchestrator/router_agent.py
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Enumération des types d'agents disponibles"""
    DATA_ANALYST = "data_analytics"
    RECRUITER = "recruiter"
    RH = "rh"
    TALENT = "talent"
    ONBOARDING = "onboarding"
    PAYROLL = "payroll"
    CRITIQUE = "critique"
    VALIDATION = "validation"
    FINAL_ANSWER = "final_answer"

class RouterAgent:
    """Agent routeur/orchestrateur pour le système RH"""
    
    def __init__(self):
        self.agent_handlers = {
            AgentType.DATA_ANALYST: self._handle_data_analyst,
            AgentType.RECRUITER: self._handle_recruiter,
            AgentType.RH: self._handle_rh,
            AgentType.TALENT: self._handle_talent,
            AgentType.ONBOARDING: self._handle_onboarding,
            AgentType.PAYROLL: self._handle_payroll,
            AgentType.CRITIQUE: self._handle_critique,
            AgentType.VALIDATION: self._handle_validation,
            AgentType.FINAL_ANSWER: self._handle_final_answer
        }
        
        # Priorité d'exécution des agents
        self.execution_order = [
            AgentType.DATA_ANALYST,
            AgentType.RECRUITER,
            AgentType.RH,
            AgentType.TALENT,
            AgentType.ONBOARDING,
            AgentType.PAYROLL,
            AgentType.CRITIQUE,
            AgentType.VALIDATION,
            AgentType.FINAL_ANSWER
        ]
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extrait les mots-clés pertinents de la requête"""
        keywords = []
        query_lower = query.lower()
        
        # Détection des besoins RH
        if any(word in query_lower for word in ["recrut", "embauch"]):
            keywords.append("recrutement")
        if any(word in query_lower for word in ["formation", "compétence", "upskill"]):
            keywords.append("talent")
        if any(word in query_lower for word in ["paie", "salaire", "budget"]):
            keywords.append("payroll")
        if any(word in query_lower for word in ["intégration", "onboarding"]):
            keywords.append("onboarding")
        if any(word in query_lower for word in ["droit", "loi", "juridique"]):
            keywords.append("rh")
            
        return keywords or ["general"]
    
    def _select_agents(self, keywords: List[str]) -> List[AgentType]:
        """Sélectionne les agents pertinents basés sur les mots-clés"""
        priority_map = {
            "recrutement": [AgentType.DATA_ANALYST, AgentType.RECRUITER, AgentType.RH],
            "talent": [AgentType.TALENT, AgentType.ONBOARDING],
            "payroll": [AgentType.PAYROLL, AgentType.DATA_ANALYST],
            "onboarding": [AgentType.ONBOARDING, AgentType.TALENT],
            "rh": [AgentType.RH, AgentType.VALIDATION],
            "general": self.execution_order
        }
        
        selected = set()
        for keyword in keywords:
            for agent in priority_map.get(keyword, []):
                selected.add(agent)
                
        return sorted(selected, key=lambda x: self.execution_order.index(x))
    
    async def route_request(self, query: str, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Route la requête vers les agents appropriés et orchestre leur exécution
        
        Args:
            query: La requête utilisateur
            state: L'état actuel du graphe (optionnel)
            
        Returns:
            Un dictionnaire contenant les résultats de tous les agents exécutés
        """
        state = state or {}
        results = {}
        
        # Analyse initiale de la requête
        keywords = self._extract_keywords(query)
        agents_to_execute = self._select_agents(keywords)
        
        logger.info(f"Routing query '{query[:50]}...' to agents: {[a.value for a in agents_to_execute]}")
        
        # Exécution séquentielle des agents
        for agent_type in agents_to_execute:
            try:
                handler = self.agent_handlers.get(agent_type)
                if not handler:
                    continue
                    
                logger.info(f"Executing {agent_type.value} agent...")
                result = await handler(query, state)
                results[agent_type.value] = result
                state.update({agent_type.value: result})  # Mise à jour de l'état
                
            except Exception as e:
                logger.error(f"Error executing {agent_type.value} agent: {str(e)}")
                results[agent_type.value] = {"error": str(e)}
                
        return results
    
    # Handlers pour chaque type d'agent
    async def _handle_data_analyst(self, query: str, state: Dict[str, Any]) -> Dict[str, Any]:
        from tools.data_analyst import analyse_data_analyst
        return analyse_data_analyst(query)
    
    async def _handle_recruiter(self, query: str, state: Dict[str, Any]) -> Dict[str, Any]:
        from agents.nodes.hr_agents.recruiter_agent import recruit_candidates
        data = state.get("data_analytics", {})
        return recruit_candidates(query, data)
    
    async def _handle_rh(self, query: str, state: Dict[str, Any]) -> Dict[str, Any]:
        from agents.nodes.hr_agents import check_labor_law
        input_data = {"query": query, "data": state.get("data_analytics", {})}
        return check_labor_law(json.dumps(input_data))
    
    async def _handle_talent(self, query: str, state: Dict[str, Any]) -> Dict[str, Any]:
        from agents.nodes.hr_agents.talentManagerrh import talent_manager_tool
        input_data = {"query": query, "state": state}
        return talent_manager_tool(json.dumps(input_data))
    
    async def _handle_onboarding(self, query: str, state: Dict[str, Any]) -> Dict[str, Any]:
        from agents.nodes.hr_agents.onboarding_agent import onboarding_agent_tool
        input_data = {
            "query": query,
            "team_size": state.get("team_size", 5),
            "state": state
        }
        return onboarding_agent_tool(json.dumps(input_data))
    
    async def _handle_payroll(self, query: str, state: Dict[str, Any]) -> Dict[str, Any]:
        from agents.nodes.hr_agents.payroll_agent import payroll_agent_tool
        input_data = {
            "query": query,
            "state": {"data_analytics": state.get("data_analytics", {})}
        }
        return payroll_agent_tool(input_data)
    
    async def _handle_critique(self, query: str, state: Dict[str, Any]) -> Dict[str, Any]:
        from agents.nodes.hr_agents.critique_rh_agent import critique_agent_rh_tool
        content = {
            "answers": {k: v for k, v in state.items() if k != "final_answer"},
            "critiques": {},
            "validations": {}
        }
        return critique_agent_rh_tool({"content": json.dumps(content)})
    
    async def _handle_validation(self, query: str, state: Dict[str, Any]) -> Dict[str, Any]:
        from agents.nodes.hr_agents.validation_rh_agent import validation_agent_rh_tool
        critique = state.get("critique", {})
        return validation_agent_rh_tool({"critique": json.dumps(critique)})
    
    async def _handle_final_answer(self, query: str, state: Dict[str, Any]) -> Dict[str, Any]:
        from agents.nodes.hr_agents.final_rh_agent import final_agent_rh_tool
        input_data = {
            "answers": {k: v for k, v in state.items() if k != "final_answer"},
            "critiques": state.get("critique", {}),
            "validations": state.get("validation", {})
        }
        return final_agent_rh_tool(input_data)

@tool
async def route_rh_request(query: str, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Point d'entrée principal pour router une requête RH vers les agents appropriés.
    
    Args:
        query: La requête textuelle décrivant le besoin RH
        state: L'état actuel du workflow (optionnel)
        
    Returns:
        Un dictionnaire contenant les résultats de tous les agents exécutés,
        avec une clé 'final_answer' contenant la réponse synthétisée.
    """
    router = RouterAgent()
    results = await router.route_request(query, state or {})
    
    # Assure que nous avons toujours une réponse finale
    if "final_answer" not in results:
        final_answer = await router._handle_final_answer(query, results)
        results["final_answer"] = final_answer
        
    return results