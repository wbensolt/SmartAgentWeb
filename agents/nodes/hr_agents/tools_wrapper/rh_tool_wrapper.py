import json
from agents.nodes.hr_agents.rhagent import check_labor_law

def rh_tool_wrapper(input: str):
    try:
        payload = json.loads(input)
    except:
        payload = {"input": input}

    return check_labor_law(payload.get("input", ""), payload.get("data", {}))