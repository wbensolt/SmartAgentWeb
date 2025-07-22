import chainlit as cl
from agents.orchestrator.router_agent import route_rh_request
from agents.vector.state_schema import GraphState
from agents.vector.retrieve_project import create_project_graph
from langgraph.graph import StateGraph
import json
from datetime import datetime
import os

AGENTS = [
    "data_analytics", "recruiter", "rh", "talent",
    "onboarding", "payroll", "critique", "validation", "final_answer"
]

def format_agent_output(agent_name: str, agent_data: dict) -> str:
    """Formate les données d'un agent pour l'affichage"""
    formatted = f"## 🔹 {agent_name.upper()}\n"
    for key, value in agent_data.items():
        formatted += f"- **{key}:** {json.dumps(value, ensure_ascii=False)}\n"
    return formatted + "\n"

@cl.on_chat_start
async def on_chat_start():
    graph = create_project_graph().compile()
    cl.user_session.set("graph", graph)
    
    # Message de bienvenue
    await cl.Message(content="🧠 **Smart RH Agent** - Posez votre question RH").send()
    
    # Bouton d'aide avec champ payload ajouté
    help_action = cl.Action(
        name="help",
        value="help",
        label="ℹ️ Aide",
        description="Afficher l'aide",
        payload={"action": "help"}
    )
    await cl.Message(
        content="Besoin d'assistance?",
        actions=[help_action]
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    graph = cl.user_session.get("graph")
    if not graph:
        return await cl.Message(content="❌ Erreur: Graphe non initialisé").send()

    try:
        loading_msg = await cl.Message(content="🔄 Analyse en cours...").send()
        
        #result = await graph.ainvoke(GraphState(query=message.content).dict())
        result = await route_rh_request(message.content)
        # Sauvegarde JSON
        os.makedirs("outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"outputs/result_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Préparation des résultats
        agent_results = []
        for agent in AGENTS:
            if agent_data := result.get(agent):
                agent_results.append(format_agent_output(agent, agent_data))

        if agent_results:
            await loading_msg.remove()
            
            # Bouton pour afficher les détails avec payload ajouté
            details_action = cl.Action(
                name="show_details",
                value="details",
                label="📂 Voir détails",
                description="Afficher les résultats détaillés",
                payload={"action": "show_details"}
            )
            await cl.Message(
                content=f"✅ Analyse terminée ({len(agent_results)} agents)",
                actions=[details_action]
            ).send()
            
            cl.user_session.set("agent_details", "\n".join(agent_results))
            
            # Résumé final
            if final := result.get("final_answer"):
                summary = final.get("output", "Aucun résumé disponible")
                await cl.Message(
                    content=f"📋 **Résumé:**\n\n{summary}",
                    language="markdown"
                ).send()

    except Exception as e:
        await loading_msg.remove()
        await cl.Message(content=f"❌ Erreur: {str(e)}").send()

@cl.action_callback("help")
async def on_help(action: cl.Action):
    help_content = """
## 📚 Aide - Smart RH Agent

**Fonctionnalités:**
- Analyse RH automatisée
- Gestion des recrutements
- Résolution de problèmes employés

**Exemples:**
- "Processus recrutement Data Scientist"
- "Problème de paie employé #1234"
- "Statistiques embauches 2023"
"""
    await cl.Message(content=help_content, language="markdown").send()

@cl.action_callback("show_details")
async def on_show_details(action: cl.Action):
    if details := cl.user_session.get("agent_details"):
        await cl.Message(
            content=details,
            language="markdown"
        ).send()
    else:
        await cl.Message(content="⚠️ Aucun détail disponible").send()
