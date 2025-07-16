from fastapi import FastAPI
from langserve import add_routes
from agents.vector.retrieve_project import create_project_graph
from langgraph.graph import END

# 📌 Compile le graphe multi-agents
graph = create_project_graph().compile()

# 📌 Wrappe le graphe dans un objet "Runnable"
class GraphWrapper:
    def invoke(self, input_data):
        return graph.invoke(input_data)

    async def ainvoke(self, input_data):
        return graph.ainvoke(input_data)

graph = create_project_graph().compile()
app = FastAPI()
add_routes(app, graph, path="/feasibility")
