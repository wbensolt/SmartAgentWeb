from agents.vector.retrieve_project import create_project_graph
from agents.vector.state_schema import GraphState

def test_graph_step_by_step():
    # Create and compile the graph
    graph = create_project_graph()
    app = graph.compile()
    
    # Initial state
    state = {
        "query": "Nous cherchons un développeur Python à Toulouse avec un budget de 70000 euros pour 2 mois.",
        "keys": {}
    }
    
    # Define the expected execution order
    execution_order = [
        "dataanalyst",
        "recruiter",
        "rh",
        "talent",
        "onboarding",
        "payroll",
        "critique",
        "validation",
        "final"
    ]
    
    # Execute nodes one by one
    for node_name in execution_order:
        print(f"\n=== Testing Node: {node_name} ===")
        
        try:
            # Execute just one step
            for output in app.stream(state, {"recursion_limit": 1}):
                if node_name in output:
                    print(f"Result from {node_name}: {output[node_name]}")
                    state.update(output)
                else:
                    print(f"Node {node_name} completed without output")
        except Exception as e:
            print(f"Error executing {node_name}: {str(e)}")
            break
    
    print("\n=== Final State ===")
    print(state)

if __name__ == "__main__":
    test_graph_step_by_step()