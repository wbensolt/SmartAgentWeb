import json
from agents.nodes.hr_agents.recruiter_agent import recruit_candidates

def recruiter_tool_wrapper(input: str):
    try:
        payload = json.loads(input)
    except:
        payload = {"input": input}

    return recruit_candidates(payload.get("input", ""), payload.get("data", {}))
