from agents.nodes.hr_agents.dataanalysta_agent import DataAnalystAgent
from agents.nodes.hr_agents.recruiter_agent import RecruiterAgent
from agents.vector.retrieve_project import get_local_retriever
from core.llm_providers import LLMManager


llm1 = LLMManager().get_llm()
llm2 = LLMManager().get_llm()
retriever = get_local_retriever()

dataanalyst = DataAnalystAgent(retriever=retriever, llm=llm1)
recruiter = RecruiterAgent(llm=llm2)

input_query = {"query": "Projet industriel avec Python, SQL, gestion de budget 30000 euros, Toulouse"}

res_data = dataanalyst.invoke(input_query)
print("DataAnalyst Result:", res_data)

input_recruiter = {"query": input_query["query"], "data": res_data}
res_recruiter = recruiter.invoke(input_recruiter)
print("Recruiter Result:", res_recruiter)
