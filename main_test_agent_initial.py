from typing import Any, Dict, Optional
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain.agents import initialize_agent, AgentType, Tool
from core.llm_providers import LLMManager
import logging
import json

# Classe d'état globale
class GraphState(BaseModel):
    query: str
    data_analytics: Optional[Dict[str, Any]] = None
    recruiter: Optional[Dict[str, Any]] = None

# Exemple de fonctions outils externes
def analyse_data_analyst_func(input_text: str) -> str:
    # Simule un résultat JSON en string
    return json.dumps({"analyse": f"Analyse des données sur: {input_text}"})

def recruit_candidates_func(input_data: Dict[str, Any]) -> str:
    # Simule un résultat JSON en string
    return json.dumps({"recrutement": f"Candidats trouvés pour: {input_data['input']}"})

def create_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    # Initialisation LLM
    llm = LLMManager().get_llm()

    # Création des outils LangChain
    analyse_tool = Tool.from_function(
        func=analyse_data_analyst_func,
        name="DataAnalyst",
        description="Analyse les données RH, projet ou business"
    )

    recruiter_tool = Tool.from_function(
        func=recruit_candidates_func,
        name="RecruiterAgent",
        description="Recherche et évalue des candidats selon le besoin"
    )

    # Initialisation agents LangChain
    dataanalyst_agent = initialize_agent(
        tools=[analyse_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION ,
        verbose=False,
        handle_parsing_errors=True
    )

    recruiter_agent = initialize_agent(
        tools=[recruiter_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION ,
        verbose=False,
        handle_parsing_errors=True
    )

    def make_node_wrapper(name: str, agent, output_key: str):
        def wrapper(state: GraphState) -> Dict[str, Any]:
            try:
                input_text = state.query
                logging.info(f"--- [{name.upper()}] AGENT INPUT ---\n{input_text}\n")

                result = agent.invoke({"input": input_text})

                logging.info(f"--- [{name.upper()}] AGENT RAW OUTPUT ---\n{result}\n")

                if isinstance(result, str):
                    parsed = json.loads(result)
                elif hasattr(result, "dict"):
                    parsed = result.dict()
                else:
                    parsed = dict(result)

                logging.info(f"--- [{name.upper()}] PARSED OUTPUT ---\n{json.dumps(parsed, indent=2)}\n")

                return {output_key: parsed}
            except Exception as e:
                logging.error(f"Erreur dans le noeud {name}: {e}")
                return {output_key: {"error": str(e)}}
        return wrapper

    # Ajout des noeuds
    graph.add_node("dataanalyst", make_node_wrapper("dataanalyst", dataanalyst_agent, "data_analytics"))
    graph.add_node("recruiter", make_node_wrapper("recruiter", recruiter_agent, "recruiter"))

    # Configuration des transitions
    graph.set_entry_point("dataanalyst")
    graph.add_edge("dataanalyst", "recruiter")
    graph.add_edge("recruiter", END)

    return graph

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    graph = create_graph()
    app = graph.compile()

    initial_query = "peut on construire une usine pour avion militaire a toulouse dans 3 mois pour un budget de 30000 euros?"
    initial_state = GraphState(query=initial_query)
    result = app.invoke(initial_state)

    print("\n====== ÉTAT FINAL ======")
    print(json.dumps(result, indent=2, ensure_ascii=False))


