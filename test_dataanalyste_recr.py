import json
import logging
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
from typing import Any, Dict

from core.llm_providers import LLMManager
from agents.vector.retrieve_project import clean_json_response
from utils.json_utils import JSONRepairer

from agents.nodes.hr_agents.tools.data_analyst import analyse_data_analyst
from agents.nodes.hr_agents.recruiter_agent import recruit_candidates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def deep_sanitize(obj):
    """
    Nettoie récursivement un objet en remplaçant les objets LangChain par leur contenu textuel.
    """
    if isinstance(obj, dict):
        return {k: deep_sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_sanitize(i) for i in obj]
    elif hasattr(obj, "content"):
        return obj.content
    else:
        return obj


def safe_recruiter_node_wrapper(recruiter_agent, dataanalyst_agent, key: str):
    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"[{key}] ▶ Reçu : {state}")
        try:
            query = state.get("query", "")
            data_analytics = state.get("data_analytics")

            # ⚠ Appel de data analyst si données absentes
            if not data_analytics:
                logger.info(f"[{key}] ⚠ Données analytiques absentes. Appel de Data Analyst.")
                da_result = dataanalyst_agent.invoke({"input": query})
                logger.info(f"[{key}] ✅ Résultat brut Data Analyst : {str(da_result)[:300]}")

                if isinstance(da_result, str):
                    da_result = JSONRepairer.safe_parse(clean_json_response(da_result))

                data_analytics = da_result.get("data_analytics", da_result)

            # 🔄 Sanitize pour json.dumps
            clean_data_analytics = deep_sanitize(data_analytics)

            recruiter_input = {
                "input": query,
                "data_analytics": clean_data_analytics
            }

            logger.info(f"[{key}] 🚀 Appel recruiter avec : {str(recruiter_input)[:300]}")

            raw_result = recruiter_agent.invoke({
                "input": f"{query}\n\nDonnées analytiques:\n{json.dumps(clean_data_analytics)}"
            })

            if isinstance(raw_result, str):
                result = JSONRepairer.safe_parse(clean_json_response(raw_result))
            else:
                result = raw_result

            logger.info(f"[{key}] ✅ Résultat recruiter : {str(result)[:300]}")
            return {"recruiter": result, "data_analytics": data_analytics}

        except Exception as e:
            logger.error(f"[{key}] ❌ Erreur : {str(e)}", exc_info=True)
            return {key: {"error": str(e), "status": "failed"}}
    return node


def test_recruiter_without_data():
    llm = LLMManager().get_llm()

    # 🔧 Agents initiaux
    data_analyst_agent = initialize_agent(
        tools=[Tool.from_function(func=analyse_data_analyst, name="AnalyseRH", description="Analyse RH")],
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
        verbose=True
    )

    recruiter_agent = initialize_agent(
        tools=[Tool.from_function(func=recruit_candidates, name="Recruiter", description="Recherche candidats")],
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
        verbose=True
    )

    # Création du wrapper node
    node = safe_recruiter_node_wrapper(recruiter_agent, data_analyst_agent, key="recruiter")

    # 💬 Cas test sans données analytiques
    state = {
        "query": "Nous cherchons un développeur Python à Toulouse avec un budget de 70000 euros pour 2 mois.",
        "data_analytics": {}  # 👈 volontairement vide pour forcer appel data analyst
    }

    result = node(state)

    # Nettoyer récursivement le résultat avant d'imprimer en JSON
    clean_result = deep_sanitize(result)

    print("Résultat final recruiter:\n", json.dumps(clean_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    test_recruiter_without_data()
