from fastapi import FastAPI
from pydantic import BaseModel
from agents.vector.retrieve_project import create_project_graph

app = FastAPI()
graph = create_project_graph().compile()

class QueryRequest(BaseModel):
    query: str

@app.post("/feasibility/invoke")
async def invoke_endpoint(request: QueryRequest):
    result = graph.invoke({"query": request.query})
    return result 
